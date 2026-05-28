import gc
import re

import matplotlib.pyplot as plt
import optuna
import numpy as np

from datetime import datetime
import json
from pathlib import Path
import time

import pandas as pd 
import traceback
from itertools import product

import torch

from utils.train_utils import plot_val_predictions, plot_val_predictions_direction, torch_to_series

from .preprocessing import prepare_data_rolling_backtest
from .training import init_tracking_dir, load_model, fit_model, save_config, load_config, predict, predict_directional
from .config import PrepConfig, MLPConfig, XGBoostConfig
from .trading import backtest, get_positions
from utils.general_utils import create_new_dir, save_interactive_plot, timeit, format_time
from .analysis import shap_analysis



is_paral = False

DAYS_MAP = {
    "7d": [0, 1, 2, 3, 4, 5, 6],
    "5d": [0, 1, 2, 3, 4],
    "4d": [0, 1, 2, 3]
}

# --- Parameter grids ---
params = {
    "target_tenors": ["m1"], # ["m1", "m2", "q1", "q2", "y1"], 
    "prediction_days_of_week": [[0,1,2,3,4]],  # , [0,1,2,3]
    "target_wins": lambda valid_days: [len(valid_days)*i for i in range(3, 4)], # [0, win, 2*win]
    "is_target_ewms": [False], 
    "ewm_spans": lambda win: [0, win], # [0, win, 2*win]
    "model_types": ["mlp", "xgboost"], 
    "dropout_rates": [0.4],
}


def get_today_pos_from_dir(track_dir, sub_dir = "post_process"):

    new_path = create_new_dir(track_dir, sub_dir)

    prep_cfg, model_cfg = load_config(track_dir)

    _, _, features, X, y = prepare_data_rolling_backtest(prep_cfg)
    input_dim = len(features)

    model_path = track_dir / "model_2.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path}")
    model = load_model(model_path, input_dim, model_cfg)


    return get_today_pos_from_cfg(X, y, model, prep_cfg, new_path)


def get_today_pos_from_cfg(
        X, 
        y, 
        model, 
        prep_cfg, 
        track_dir, 
        ticker="C5TC",
        start=2025,
        # strats=None,
        strats=["binary_pos_4"],
    ):
    track_dir.mkdir(parents=True, exist_ok=True)

    # Get dates of interest and extend X and y_pred to it
    dates: pd.DatetimeIndex = X.index[(X.index >= pd.Timestamp(f"{start}-01-01")) & (~X.isna().any(axis=1))]
    X_test_ext = torch.tensor(X.loc[dates].values, dtype=torch.float32)
    with torch.no_grad():
        y_pred = predict(model, X_test_ext)

    pred = torch_to_series(y_pred, dates, name="y_pred")
    target = y.reindex(dates).rename("y_true")
    plot_kwargs = {
        "y_test": target,
        "y_pred": pred,
        "tracking_dir": track_dir,
    }
    plot_val_predictions(**plot_kwargs)

    pred_df = (
        pd.concat([target, pred], axis=1)
        .dropna(subset=["y_true"])
        .sort_index()
    )
    stats_df = backtest(pred_df, prep_cfg, track_dir)
    
    # SHAP
    features = X.columns.to_list()
    dates_bg: pd.DatetimeIndex = X.index[(X.index < pd.Timestamp(f"{start}-01-01")) & (~X.isna().any(axis=1))]
    X_bg = torch.tensor(X.loc[dates_bg].values, dtype=torch.float32)
    shap_analysis(model, X_bg, X_test_ext, features, track_dir, prep_cfg)

    y_pred_df = pd.DataFrame(
        y_pred.detach().cpu().numpy(),
        index=pd.to_datetime(dates),
        columns=[ticker]
    )

    pos_df = get_positions(y_pred_df, ticker, prep_cfg)
    # pos_df = pd.DataFrame(pos_df)
    pos_df = pd.concat(pos_df, axis=1)

    if strats is not None:
        pos_df = pos_df[strats]
 
    print(pos_df.tail(5))

    fig, ax = plt.subplots(figsize=(12, 6))

    pos_df.plot(ax=ax)
    ax.set_title("Positions")
    fig.savefig(track_dir / "all_pos_full_timespan.png")
    plt.close(fig)

    pos_df.to_csv(track_dir / "pos_df.csv")

    return pos_df


def prepare_backtest_data(y_pred, y_true, idxs):
    pred = y_pred.detach().cpu().flatten()
    # assert torch.isfinite(pred).all().item(), "pred contains NaN or inf after filtering"
    idxs = pd.Index(idxs)

    common_idxs = idxs.intersection(y_true.index)
    pred = pred[idxs.isin(common_idxs)]
    target = y_true.loc[common_idxs]
    valid_mask = target.notna().to_numpy()

    pred = pred[valid_mask]
    target = target[valid_mask]
    idxs_target = common_idxs[valid_mask]
    # print(idxs_target[-50:])

    return pred, target, idxs_target


@timeit
def rolling_backtest(
    prep_cfg: PrepConfig,
    model_cfg: MLPConfig | XGBoostConfig,
    metric: str,
    track_dir: Path | None = None,
    is_compute_sharpe: bool = True,
    is_compute_shap: bool = True,
    pred_std_ratio_thresh: float = 0.03,
    min_sign_fraction: float = 0.1,
) -> float | tuple[float, dict[str, float]]:
    
    # Unravel config
    is_tar_direction = prep_cfg.is_directional_target

    if metric not in {"sharpe", "normalized_rmse"}:
        raise ValueError(
            f"Metric {metric} not recognized. It must be either 'sharpe' or 'normalized_rmse'."
        )

    track_dir = save_config(prep_cfg, model_cfg, track_dir)

    fold_loaders, test_loaders, features, X, _ = prepare_data_rolling_backtest(prep_cfg)
    input_dim = len(features)

    compute_sharpe = metric == "sharpe" or is_compute_sharpe

    rmse_scores = []
    pred_frames = []

    for i, (X_test, y_test, dates_test) in enumerate(test_loaders):
        dates_test = pd.to_datetime(dates_test)

        print(
            f"\n--- Testing: "
            f"{dates_test.min().strftime('%Y-%m-%d')} to "
            f"{dates_test.max().strftime('%Y-%m-%d')}"
        )

        # Get y_pred
        model, track_dir = fit_model(
            [fold_loaders[i]],
            input_dim,
            prep_cfg,
            model_cfg,
            track_dir,
            is_save=True if is_compute_shap else False,
        )

        is_curr_year = dates_test.max().year == datetime.now().year
        dates_pred: pd.DatetimeIndex =  X.index[(X.index >= dates_test.min()) & (~X.isna().any(axis=1))] if is_curr_year else dates_test.copy()
        X_pred = (
            torch.tensor(X.loc[dates_pred].values, dtype=torch.float32)
            if is_curr_year
            else X_test.clone()
        )        
        if is_tar_direction:
            y_pred, probas = predict_directional(model, X_pred)
        else:
            y_pred = predict(model, X_pred).squeeze(-1)

        # Detach vars and adapt if directional strat        
        pred = y_pred.detach().cpu().flatten()
        target = y_test.detach().cpu().flatten()
        
        if is_tar_direction:
            pred = pred-1
            target = target.long()-1

        # --- Pruning
        # prune run if prediction is flat
        if not is_tar_direction:
            pred_std_ratio = (
                pred.std(unbiased=False) /
                target.std(unbiased=False).clamp_min(1e-12)
            ).item()

            if pred_std_ratio < pred_std_ratio_thresh:
                raise optuna.TrialPruned("flat_prediction")

        # prune run if prediction has strong bias towards long/short
        pos_frac = (pred > 0).float().mean().item()
        neg_frac = (pred < 0).float().mean().item()

        if min(pos_frac, neg_frac) < min_sign_fraction:
            raise optuna.TrialPruned("prediction_sign_bias")

        # if i == 1
        target_series = torch_to_series(target, dates_test, name="y_true")
        pred_series = torch_to_series(y_pred, dates_pred, name="y_pred")
        plot_kwargs = {
            "y_test": target_series,
            "y_pred": pred_series,
            "tracking_dir": track_dir,
        }
        if is_tar_direction:
            plot_kwargs["probas"] = probas
            plot_val_predictions_direction(**plot_kwargs)
        else:
            plot_val_predictions(**plot_kwargs)

        if not is_tar_direction:
            n = min(len(pred), len(target))
            rmse = torch.sqrt(
                torch.nn.functional.mse_loss(pred[:n], target[:n])
            ).item()
            print(f"Test Results:\n\tRMSE:\t{rmse:.3f}")

            if metric == "normalized_rmse":
                y_scale = torch.std(target).item()
                if y_scale == 0:
                    raise ValueError("Cannot compute normalized RMSE because target std is 0.")

                rmse_scores.append({
                    "score": rmse / y_scale,
                    "nb_days": len(dates_test),
                })
 
        if compute_sharpe:
            pred_df = (
                pd.concat([target_series, pred_series], axis=1)
                .dropna(subset=["y_true"])
                .sort_index()
            )

            if is_tar_direction:
                pred_df[["p_short", "p_neutral", "p_long"]] = probas.detach().cpu().numpy()

            pred_frames.append(pred_df)

        # # SHAP Analysis
        if is_compute_shap:
            dates_bg: pd.DatetimeIndex = X.index[(X.index < dates_test.min()) & (~X.isna().any(axis=1))]
            X_bg = torch.tensor(X.loc[dates_bg].values, dtype=torch.float32)
            shap_analysis(model, X_bg, X_pred, features, track_dir, prep_cfg)

    stats_df: pd.DataFrame | None = None

    if compute_sharpe:
        pred_df = pd.concat(pred_frames, axis=0).sort_index()
        stats_df = backtest(pred_df, prep_cfg, track_dir)

    if metric == "sharpe":
        if stats_df is None:
            raise ValueError("metric='sharpe' requires compute_sharpe=True")

        sharpe_per_strat = stats_df.loc["Net Sharpe"].astype(float)
        best_strat = sharpe_per_strat.idxmax()
        best_sharpe = float(sharpe_per_strat.max())
        sharpe_by_strat: dict[str, float] = {
            str(k): float(v)
            for k, v in sharpe_per_strat.to_dict().items()
        }

        print(f"Best strat: {best_strat} with weighted mean sharpe: {best_sharpe}")
        return best_sharpe, sharpe_by_strat

    rmse_df = pd.DataFrame(rmse_scores)
    weighted_score = (
        rmse_df["score"].mul(rmse_df["nb_days"]).sum()
        / rmse_df["nb_days"].sum()
    )

    print(f"\nWeighted mean score: {weighted_score}\n")
    return float(weighted_score)


"""Parameters Scan"""
if __name__ == "__main__":
    """
    imported from '2026-05-20-optuna_mlp_mean_rolling_sharpe_all_q1_new' / '2026-05-20_10-55_experiment_40'
    """
    track_dir = PrepConfig().out_dir / "data"
    get_today_pos_from_dir(track_dir)