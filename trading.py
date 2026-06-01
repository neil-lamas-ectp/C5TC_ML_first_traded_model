import math
from pathlib import Path
import matplotlib.dates as mdates
from datetime import datetime
import os

from ectp_systematic import data
import matplotlib.pyplot as plt
from typing import Optional

from .config import PrepConfig

from ectp_systematic.configs import universe
from ectp_systematic.backtest.costs import CostCalculatorSimple
from ectp_systematic.backtest import BacktestAnalyser

from utils.general_utils import align_yaxis_zero, create_new_filename, rolling_normalize, timeit
import pandas as pd
import numpy as np


@timeit
def backtest(
    pred: pd.Series,
    cfg: PrepConfig,
    track_dir,
    ticker: str = "C5TC",
    # selected_strat = "binary_pos_4",
    selected_strat = None,
) -> pd.DataFrame:
    tar_tenor = cfg.target_tenor

    pred.index = pd.to_datetime(pred.index)

    min_year = pred.index.year.min()
    max_year = pred.index.year.max()

    adj_price: pd.DataFrame = (
        universe.UNIVERSE[f"{ticker}_{tar_tenor}"]
        .price(
            what="settlement",
            start_date=str(min_year),
            end_date=str(max_year + 1),
        )
        .ts.squeeze()
        .to_frame(ticker)
    )

    if cfg.is_directional_target:
        positions_dict = get_positions_directional(pred, ticker, cfg, adj_price)
    else:
        positions_dict = get_positions(pred, cfg)

    all_stats = []
    best_name = None
    best_ba = None
    best_net_sharpe = float("-inf")
    nb_datapoints = len(pred)

    bid_offer: dict[str, float | pd.Series] = {
        ticker: 200.0,
    }
    cost_calculator = CostCalculatorSimple(
        bid_offer,
        cost_multiplier=0.35,
    )

    prev_rows_rm = None

    for name, pos in positions_dict.items():
        mask_na = ~pd.concat([adj_price, pos], axis=1).isna().any(axis=1)
        nb_removed = nb_datapoints - mask_na.sum()

        if nb_removed != prev_rows_rm:
            print(
                f"Number of NaN rows removed before backtest is {nb_removed} over "
                f"{nb_datapoints} ({nb_removed / nb_datapoints * 100:.1f} %)"
            )
            prev_rows_rm = nb_removed

        adj_price_clean = adj_price.loc[mask_na]
        pos_clean = pos.loc[mask_na]

        ba_final = BacktestAnalyser(
            adj_price_clean,
            pos_clean,
            cost_calculator=cost_calculator,
            lag=2,
        )

        net_sharpe = ba_final.statistics["Net Sharpe"]
        if name == selected_strat:
            ba_to_print = ba_final
        if net_sharpe > best_net_sharpe:
            best_name = name
            best_ba = ba_final
            best_net_sharpe = net_sharpe

        stats = pd.Series(ba_final.statistics).reset_index()
        stats.columns = ["quantity", "value"]
        stats["strategy"] = name
        all_stats.append(stats)

    stats_per_pos_df = pd.concat(all_stats, ignore_index=True)

    if best_ba is not None:
        print(f"Best strat is {best_name} with net Sharpe: {best_net_sharpe:.3f}")

    quantity_order = stats_per_pos_df["quantity"].drop_duplicates()
    out = (
        stats_per_pos_df.pivot_table(
            index="quantity",
            columns="strategy",
            values="value",
            aggfunc="first",
        )
        .reindex(quantity_order)
    )

    filename = create_new_filename(track_dir, "strat_stats_all", "csv")
    out.to_csv(track_dir / filename)

    top_5_strats = (
        stats_per_pos_df.loc[stats_per_pos_df["quantity"].eq("Net Sharpe")]
        .sort_values("value", ascending=False)
        ["strategy"]
        .head(5)
        .tolist()
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    all_colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for i, name in enumerate(top_5_strats):
        pos = positions_dict[name]
        ax1.plot(
            pos.index,
            pos[ticker],
            label=name,
            color=all_colors[i % len(all_colors)],
            zorder=i + 1,
        )

    ax1.legend()
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Position", color=all_colors[0])
    ax1.tick_params(axis="y", labelcolor=all_colors[0])
    ax1.grid(True, zorder=0)

    ax2 = ax1.twinx()
    ax2_color = all_colors[len(top_5_strats) % len(all_colors)]

    ax2.plot(
        pred,
        label="Prediction",
        color=ax2_color,
        alpha=0.3,
        linestyle="--",
        zorder=1,
    )

    ax2.set_ylabel("Prediction", color=ax2_color)
    ax2.tick_params(axis="y", labelcolor=ax2_color)

    align_yaxis_zero(ax1, ax2)

    fig.suptitle("Position vs Prediction")

    filename = create_new_filename(track_dir, "positions_all_strats", "png")
    fig.savefig(track_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if selected_strat is not None:
        save_backend_results(ba_to_print, track_dir, selected_strat)
    elif best_ba is not None:
        save_backend_results(best_ba, track_dir, best_name)
    else:
        raise ValueError("selected_strat ill-defined")  

    return out


def save_backend_results(ba, out_dir, strat_name):
    # # --- Save stats
    # print(f'Net Sharpe: {ba.statistics["Net Sharpe"]:.3f}')
    # # pd.DataFrame(ba.statistics).to_csv(track_dir / f'strat_stats_{name}.csv')
    # filename = create_new_filename(out_dir, "statistics", "csv")
    # pd.Series(ba.statistics).to_frame("value").to_csv(
    #     filename,
    #     header=True
    # )
    
    # --- Save P&L plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ba.pnl_wc.plot(ax=ax)
    ax.grid()
    ax.tick_params(axis='x', rotation=45)
    ax.set_title(f"P&L ({strat_name})")
    filename = create_new_filename(out_dir, "pnl", "png")
    fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Lead Lag plot
    fig = ba.lead_lag_plot()
    fig.update_layout(
        title=f"Lead Lag ({strat_name})",
        xaxis_tickangle=45,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    filename = create_new_filename(out_dir, "lead_lag", "png")
    fig.write_image(out_dir / filename, width=1200, height=800, scale=2)

    del fig

    # # --- Plot P&L long/short
    # plt.figure()
    # ba.long_short_pnl.plot()
    # plt.grid()
    # plt.title("Long-short P&L")
    # plt.xticks(rotation=45)
    # filename = create_new_filename(out_dir, "pnl_long_short", "png")
    # plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    # plt.close()


    # # --- Plot position ---
    # plt.figure()
    # ba.positions.plot()
    # plt.title("Positions")
    # plt.xlabel("Time")
    # plt.ylabel("Position")
    # plt.xticks(rotation=45)
    # plt.grid(True)    
    # filename = create_new_filename(out_dir, "position", "png")
    # plt.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    # plt.close()

    # # --- Save P&L seasonality
    # # def annual_seasonality(ba) -> pd.Series:
    # #     """Returns cumulative mean returns over the calendar year. More specifically,
    # #     we first compute mean return for each day of the year then we take cumsum
 
    # #     Returns:
    # #         pd.Series
    # #     """
    # #     returns = ba.returns_daily
    # #     date_group = pd.Series(returns.index.dayofyear, index=returns.index)  # type: ignore
    # #     date_group = date_group.apply(
    # #         lambda x: pd.Timestamp("1999-12-31") + pd.DateOffset(days=x)
    # #     )
    # #     return returns.groupby(date_group).mean().cumsum()

    # def annual_seasonality(ba) -> pd.Series:
    #     returns = ba.returns_daily

    #     month_group = pd.Series(
    #         returns.index.month,
    #         index=returns.index
    #     )

    #     return returns.groupby(month_group).mean().cumsum()
    
    # fig, ax = plt.subplots()
    # series = annual_seasonality(ba)
    # series.plot(ax=ax)
    # ax.grid(True)
    # ax.set_title("P&L")
    # ax.tick_params(axis="x", rotation=45)
    # filename = create_new_filename(out_dir, "seasonality", "png")
    # fig.savefig(out_dir / filename, dpi=300, bbox_inches="tight")
    # plt.close(fig)


def vol_adj_pos(adj_price: pd.DataFrame, pos: pd.DataFrame):
    vol, vn_rets = get_rol_vol(adj_price)
    adj_price_filt = vol_filt_price(adj_price, vn_rets, vol)
    valid_idxs = adj_price_filt.index
    adj_pos = (pos / vol).reindex(valid_idxs)
    
    vol_gearing_ratio = get_vol_gearing_ratio(adj_price_filt, adj_pos, vol)
    adj_pos = adj_pos.multiply(vol_gearing_ratio, axis="index")

    return adj_pos


def get_vol_gearing_ratio(
        adj_price_filt, 
        adj_pos, 
        vol,
        span = 252, # 520
        min_periods=130
    ):
    ba_gearing = BacktestAnalyser(
        adj_price_filt, adj_pos.ffill(), vol=vol.ffill(), lag=1
    )

    returns = ba_gearing.returns_daily

    vol_gearing_ratio = 1e6 / 2 * (
        16
        * returns.ewm(span=span, min_periods=min_periods)
        .std()
        .ffill()
        .bfill()
        .np.shift_skip_na(1)
        .reindex(adj_pos.index, method="ffill")
    )

    return vol_gearing_ratio
    
def get_rol_vol(
        adj_price,
        vol_span = 252, 
        min_periods=30
    ):
    """Get rolling volatility (skip NaN)"""
    vn_rets = adj_price.np.diff_skip_na()
    vol = vn_rets.ewm(span=vol_span, min_periods=min_periods).std()
    # shift 2 (to not use today's EOD which is future data + 1 from syst team)
    vol = vol.np.shift_skip_na(2)
    return vol, vn_rets


def vol_filt_price(
        adj_price, 
        vn_rets, 
        vol,
        max_vol = 20
    ) -> pd.DataFrame:
    """Filter out huge volatility (no position for those days)"""
    # Normalize returns by vol 
    vn_rets_norm = vn_rets / vol
    adj_price_filt = adj_price[vn_rets_norm.abs() < max_vol]
    # adj_price_filt = adj_prices_loop[(vn_rets_norm.abs() < min_periods).all(axis=1)]
    return adj_price_filt


def round_pos(pos: pd.DataFrame, multiple: int = 5) -> pd.DataFrame:
    """ Round position to multiples of 5 """
    return multiple * np.round(pos / multiple)



def get_positions_directional(
    df,
    ticker,
    cfg: PrepConfig,
    adj_price=None,
    max_risk=10.0,
) -> dict[str, pd.DataFrame]:

    tar_win = cfg.target_win
    threshold = cfg.direction_prob_threshold

    # 1. Best bet / argmax signal: -1, 0, 1
    signal_argmax = df["y_pred"].to_frame(name=ticker)

    # 2. Probability spread signal: [-1, 1]
    signal_proba = (df["p_long"] - df["p_short"]).to_frame(name=ticker)

    # 3. Probability spread, only when confidence clears threshold
    signal_proba_thresh = signal_proba.where(signal_proba.abs() >= threshold, 0.0)

    def build_positions(signal: pd.DataFrame) -> dict[str, pd.DataFrame]:
        position = signal.rolling(window=tar_win, min_periods=1).sum()
        position *= max_risk / tar_win

        position_halved = signal.rolling(window=math.ceil(tar_win / 2), min_periods=1).sum()
        position_halved *= max_risk / math.ceil(tar_win / 2)

        position_third = signal.rolling(window=math.ceil(tar_win / 3), min_periods=1).sum()
        position_third *= max_risk / math.ceil(tar_win / 3)

        out = {
            "pos": position,
            "halved_pos": position_halved,
            "third_pos": position_third,
        }

        if adj_price is not None:
            adj_position = vol_adj_pos(adj_price, position)
            common_idxs = position.index.intersection(adj_position.index)
            max_abs_pos = np.max(np.abs(position[ticker].loc[common_idxs]))
            adj_position = rolling_normalize(adj_position) * max_abs_pos
            out["vol-adj_pos"] = adj_position

        return out

    positions_dict = {}

    for prefix, signal in {
        "argmax": signal_argmax,
        "proba": signal_proba,
        "proba-thresh": signal_proba_thresh,
    }.items():
        for name, position in build_positions(signal).items():
            positions_dict[f"{prefix}_{name}"] = position

    return positions_dict


def get_positions(
    pred_series: pd.Series,
    cfg: PrepConfig,
    ticker="C5TC",
    max_risk=10.0,
    ternary_threshs = [0.05, 0.1, 0.2, 0.5],
) -> dict[str, pd.DataFrame]:
    """
    Build non-directional positions from analog predictions.
    """
    tar_win = cfg.target_win
    pred = pred_series.to_frame(name=ticker)

    def overlap_position(signal: pd.DataFrame, win: int) -> pd.DataFrame:
        position = signal.rolling(window=win, min_periods=1).sum()
        position *= max_risk / win
        return position

    thresholds = [t for t in ternary_threshs if t <= pred.abs().to_numpy().max()]
    signals = {
        "propto": rolling_normalize(pred),
        "binary": np.sign(pred),
        **{
            f"ternary_{thresh:g}": np.sign(pred).where(
                pred.abs() >= thresh,
                0,
            )
            for thresh in thresholds
        },
    }

    windows = {
        "pos_1": tar_win,
        "pos_2": math.ceil(tar_win / 2),
        "pos_3": math.ceil(tar_win / 3),
        "pos_4": math.ceil(tar_win / 4),
    }

    positions_dict = {
        f"{signal_name}_{position_name}": overlap_position(signal, win)
        for signal_name, signal in signals.items()
        for position_name, win in windows.items()
    }

    # if adj_price is not None:
    #     base_position = positions_dict["propto_pos"]
    #     adj_position = vol_adj_pos(adj_price, base_position)

    #     common_idxs = base_position.index.intersection(adj_position.index)
    #     max_abs_pos = np.max(np.abs(base_position[ticker].loc[common_idxs]))

    #     adj_position = rolling_normalize(adj_position) * max_abs_pos
    #     positions_dict["propto_vol_adj_pos"] = adj_position

    return positions_dict
