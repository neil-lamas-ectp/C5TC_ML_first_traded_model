from collections import Counter, abc
from filelock import FileLock
import json
import pathlib
from pprint import pprint

from typing import Tuple, Union

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

from C5TC_ML_first_traded.data_ingestion import load_C5TC_all_tenors, load_C5TC_spot, get_capes_data, load_commodities

from .config import PrepConfig
from utils.general_utils import dicts_equal_spec_keys, period_to_months, map_tenor_period, rolling_frontward_eff_ewm, rolling_frontward_eff_mean, rolling_normalize, shift_non_nan_df, timeit, plot_series, plot_multiple_series, shift_non_nan, create_new_filename

import torch
from torch.utils.data import TensorDataset, DataLoader


@timeit
def prepare_data_rolling_backtest(cfg:PrepConfig, is_load_data = True):

    raw_data = None
    if is_load_data:
        raw_data = is_raw_data_cached(cfg)
    if raw_data is None:
        print(f"{'Current config differs from saved configs. ' if is_load_data else ''}Acquiring raw data...")
        raw_data = data_acquisition(cfg)
    else:
        print("Loaded raw dataset")

    print("Formatting dataset...")
    X, y = data_transformation(*raw_data, cfg = cfg)

    fold_loaders, test_loaders = build_dataloaders_rolling_test(X, y, cfg)
    features = X.columns.to_list()
    print(f"Number of features after filtering: {len(features)}")

    return fold_loaders, test_loaders, features, X, y


# @timeit
# def prepare_data(cfg:PrepConfig, is_load_data = True):
    
#     if is_load_data:
#         raw_data = is_raw_data_cached(cfg)
#     if raw_data is None:
#         print(f"{'Current config differs from saved configs. ' if is_load_data else ''}Acquiring data...")
#         raw_data = data_acquisition(cfg)
#     else:
#         print("Loaded formatted dataset")

#     print("Formatting dataset...")
#     X, y = data_transformation(*raw_data, cfg = cfg)

#     fold_loaders, dates_test, X_test, y_test = build_dataloaders(X, y, cfg)
#     features = X.columns.to_list()

#     return fold_loaders, dates_test, X_test, y_test, features


@timeit
def data_acquisition(cfg: PrepConfig):
    """
    Get cumulated returns and tackle missing dates
    """
    # --- Cape 5TC index nominal value for multiple tenors
    # C5TC_nominal_df = pd.read_csv(in_dir / f"C5TC_nominal_per_period.csv", parse_dates=["date"], index_col="date")
    C5TC_nominal_df = load_C5TC_all_tenors()
    
    C5TC_spot = load_C5TC_spot()

    nominal_tenors_df = get_nominal_prices(C5TC_nominal_df, cfg)

    rets_df, _ = get_returns(C5TC_nominal_df, cfg)

    # other_balt_indices = get_other_balt_indices(cfg)
    
    # --- Signal Ocean
    # fixtures_df = get_fixtures_df(cal_date_range, cfg)
    capes_data = get_capes_data()
    
    signal_ocean_df = pd.concat(
        [
            series.rename(label)
            for label, series in capes_data.items()
        ],
        axis=1,
    )

    if cfg.is_plot:
        for label, series in capes_data.items():
            plot_signal_ocean_ts(label, series, cfg)

    # Chinese IO inventories
    io_inv_china = get_io_port_inv_china(cfg)

    # Commodities and bunker
    commodities_df_dict = get_commodities_df(cfg)
    commodities_df = pd.concat(commodities_df_dict.values(), axis=1)
    print(f"Number of commodities considered: {len(commodities_df.columns)} (exchange symbols: {commodities_df.columns.tolist()})")

    raw_data = C5TC_spot, nominal_tenors_df, rets_df, signal_ocean_df, io_inv_china, commodities_df

    save_raw_data_with_cfg(raw_data, cfg)

    return raw_data


@timeit
def data_transformation(
    # cal_date_range,
    C5TC_spot,
    nominal_tenors_df: pd.DataFrame,
    rets_df,
    signal_ocean_df,
    io_inv_china,
    commodities_df,
    # seasonality_df,
    cfg: PrepConfig,
) -> Tuple[
        pd.DataFrame, 
        pd.Series,
    ]:
    # Unpack values
    nominal_tenors: list[str] = cfg.nominal_tenors
    target_tenor: str = cfg.target_tenor
    prediction_days_of_week: list[int] = cfg.prediction_days_of_week

    # cal_date_range = pd.date_range(C5TC_nominal_df["date"].min(), C5TC_nominal_df["date"].max(), freq="D")   
    cal_date_range = pd.date_range(
        C5TC_spot.index.min(),
        C5TC_spot.index.max() + pd.offsets.BDay(1),
        freq="D",
    )

    # Seasonality metadate
    seasonality_df = get_seasonality_df(cal_date_range)

    # ---------------
    # Manage last day
    # ---------------
    C5TC_spot = C5TC_spot.reindex(cal_date_range)
    nominal_tenors_df = nominal_tenors_df.reindex(cal_date_range)
    rets_df = rets_df.reindex(cal_date_range)

    # ---------------
    # LAG -1 EOD data (today during trading day we dont know those values yet)
    # ---------------
    yest_spot = shift_non_nan(C5TC_spot, 1)
    yest_tenors_df = shift_non_nan_df(nominal_tenors_df, 1)
    yest_rets = shift_non_nan(rets_df, 1)
    yest_commodities_df = shift_non_nan_df(commodities_df, 1)
    yest_spot.iloc[-1] = C5TC_spot.dropna().iloc[-1]
    yest_tenors_df.iloc[-1] = nominal_tenors_df.dropna().iloc[-1]
    yest_rets.iloc[-1] = rets_df.dropna().iloc[-1]
    yest_commodities_df.iloc[-1] = commodities_df.dropna().iloc[-1]

    # ---------------
    # LAG -2 on Signal Ocean data (due to their data refinement process)
    # ---------------
    lagged_signal_ocean_df = shift_non_nan_df(signal_ocean_df, 2)

    # ---------------
    # Normalize signal ocean data by fleet size
    # ---------------
    if cfg.is_fleet_normalized:
        fleet_size_col = "fleet_size"
        if not fleet_size_col in lagged_signal_ocean_df.columns:
            raise Exception("Need fleet size in signal ocean DataFrame")
        else:
            vessels_cols = [
                c for c in lagged_signal_ocean_df.columns
                if not (c.lower().endswith("exp") or c.lower() == fleet_size_col)
            ]
            lagged_signal_ocean_df[vessels_cols] = lagged_signal_ocean_df[vessels_cols].div(
                lagged_signal_ocean_df[fleet_size_col],
                axis=0,
            )
    # print(lagged_signal_ocean_df.tail(10))

    # ---------------
    # NaN management
    # ---------------
    yest_spot_filled = yest_spot.reindex(cal_date_range).ffill()
    yest_tenors_filled_df = yest_tenors_df.reindex(cal_date_range).ffill()
    yest_rets_filled_df = yest_rets.reindex(cal_date_range).fillna(0) # today's return
    yest_commodities_filled_df = yest_commodities_df.reindex(cal_date_range).ffill()
    # io_inv_china_filled = io_inv_china.reindex(cal_date_range).ffill()

    # --------------------------------
    # Combine inputs and rename them
    # --------------------------------
    X: pd.DataFrame = yest_tenors_filled_df[nominal_tenors].rename(
        columns=lambda t: f"YEST_C5TC_NOMINAL_{t.upper()}"
    ).copy()

    X = X.reindex(cal_date_range) # To ensure join on left works afterwards

    X["YEST_C5TC_SPOT"] = yest_spot_filled

    # Returns for all tenors
    yest_rets_filled_renamed = yest_rets_filled_df.set_axis(
        [f"YEST_C5TC_RET_{c.upper()}" for c in yest_rets_filled_df.columns],
        axis="columns",
    )
    X = X.join(yest_rets_filled_renamed, how="left")

    # # Lagged returns for target tenor
    # X = X.join(prev_rets_filled_df, how="left")

    X = X.join(
        lagged_signal_ocean_df.rename(columns=lambda c: f"SIGNAL_OCEAN_CAPES_{c.upper()}"),
        how="left",
    )

    # --- cyclic time location
    seasonality_df = seasonality_df.rename(columns={
        "WEEKDAY_SIN": "WEEK_SIN",
        "WEEKDAY_COS": "WEEK_COS",
        "MONTHDAY_SIN": "MONTH_SIN",
        "MONTHDAY_COS": "MONTH_COS",
        "MONTH_SIN": "YEAR_SIN",
        "MONTH_COS": "YEAR_COS",
    })
    X = X.join(
        seasonality_df.add_prefix("SEASON_WITHIN_"),
        how="left",
    )

    # ---------------
    # Moments indicators
    # ---------------
    @timeit
    def get_all_moments():
        name_2_mom_vars = {
            "YEST_C5TC_SPOT": yest_spot,
            **{f"YEST_C5TC_RET_{t.upper()}": yest_rets[t] for t in nominal_tenors},
            **{f"SIGNAL_OCEAN_CAPES_{c.upper()}": lagged_signal_ocean_df[c] for c in lagged_signal_ocean_df.columns},
        }

        moments_df = pd.DataFrame(index=cal_date_range)
        for prefix, var in name_2_mom_vars.items():
            moments_var = get_moments(var, prefix, cfg)
            moments_df = moments_df.join(moments_var, how="left")

        return moments_df

    moments_df = get_all_moments()

    X = X.join(
        moments_df,
        how="left"
    )

    # Seasonality - Implement x[t] - x[t-1 year] for x in inputs of interest with seasonality
    # Choose transformed data to deaseasonalize
    season_ret_mom_cols = [
        make_moments_name(f"YEST_C5TC_RET_{t.upper()}", type_mom, win)
        for t in nominal_tenors
        for type_mom in cfg.moments_funcs.keys()
        for win in cfg.moments_roll_wins
    ]
    season_ocean_mom_cols = [
        make_moments_name(f"SIGNAL_OCEAN_CAPES_{c}_EXP", type_mom, win=14)
        for c in ["BRAZIL", "AUS"]
        for type_mom in cfg.moments_funcs.keys()
    ]
    @timeit
    def deseasonalize():
        name_2_deseason_vars = {
            **{f"YEST_C5TC_NOMINAL_{t.upper()}": yest_tenors_df[t] for t in nominal_tenors},
            **{col: moments_df[col] for col in season_ocean_mom_cols},
            **{f"SIGNAL_OCEAN_CAPES_{c.upper()}": lagged_signal_ocean_df[c] for c in lagged_signal_ocean_df.columns},
            **{col: moments_df[col] for col in season_ret_mom_cols}
        }

        deseasonalized_df = pd.DataFrame(index=cal_date_range)
        for prefix, var in name_2_deseason_vars.items():
            deseasonalized_var = deseasonalize_var(var, prefix, cfg)
            deseasonalized_df = deseasonalized_df.join(deseasonalized_var, how="left")
        
        return deseasonalized_df

    deseasonalized_df = deseasonalize()

    X = X.join(
        deseasonalized_df,
        how="left"
    )

    # -------------------------------
    # Some stats on X
    # -------------------------------
    # # Count NaNs per column
    # nan_counts = X.isna().sum()
    # top10_nans = nan_counts.sort_values(ascending=False)
    # print(f"Number of NaNs per col in X:\n{top10_nans.head(30)}")

    print(f"Number of raw inputs: {len(X.columns)}")

    # TARGET
    y: pd.Series = make_target(rets_df[target_tenor], nominal_tenors_df[target_tenor], cfg)

    # --------------
    # Remove some selected weekdays
    # --------------    
    if len(prediction_days_of_week) > 0:
        mask_weekdays = pd.to_datetime(X.index).dayofweek.isin(prediction_days_of_week)
        n_drop = (~mask_weekdays).sum()
        print(f"Will drop {n_drop}/{len(X)} ({n_drop/len(X)*100:.2f}%) rows containing excluded days of week for which prediction will not calculated")
        X = X.loc[mask_weekdays, :]
        y = y.reindex(X.index)
    else:
        raise ValueError("Prediction days list from config is empty")

    # ---------------------------
    # Scaling
    # ---------------------------
    # scaling_cols = X.columns[~X.columns.str.startswith('SEASON')]
    scaling_cols = X.columns
    X, _, _ = scale(X, scaling_cols, cfg) # type: ignore[assignment]

    # Log-transform y
    if cfg.is_log_target:
        y = y.apply(np.log1p)

    # ---------------------------
    # DEBUGGING AID
    # ---------------------------
    # with open(cfg.debug_dir / "X.txt", "w") as f:
    #     f.write(X.to_string())
    # with open(cfg.debug_dir / "y.txt", "w") as f:
    #     f.write(y.to_string())
    # with open(cfg.debug_dir / "X_columns.txt", "w", encoding="utf-8") as f:
    #     f.write(",\n".join(sorted(map(str, X.columns))))

    return X, y


def scale(
    x: Union[pd.Series, pd.DataFrame],
    scaling_cols: Union[str, list, None],
    cfg,
) -> Tuple[Union[pd.Series, pd.DataFrame], pd.Series, pd.Series]:
    """
    Rolling scale (std or z-score) for time series inputs or targets.

    Returns:
        x_scaled      : same type as x
        rolling_mean  : pd.Series (aligned with x_scaled)
        rolling_std   : pd.Series (aligned with x_scaled)
    """

    min_period = cfg.min_period_scale
    window = cfg.window_scale
    
    # --- Determine column(s) to scale ---
    single_col = False
    if isinstance(x, pd.Series):
        single_col = True
        x = x.to_frame()        # convert Series → DataFrame
        scaling_cols = x.columns.tolist()  # now always a list
    elif isinstance(x, pd.DataFrame):
        if scaling_cols is None:
            scaling_cols = x.columns.tolist()
    else:
        raise TypeError("x must be a pd.Series or pd.DataFrame")

    # --- Compute rolling stats ---
    rolling = x[scaling_cols].rolling(window=window, min_periods=min_period)
    rolling_mean = rolling.mean()
    rolling_std  = rolling.std(ddof=0)

    # --- Apply scaling ---
    if cfg.scaler_type == "none":
        x_scaled = x.copy()
    elif cfg.scaler_type == "std":
        x_scaled = x.copy()
        x_scaled.loc[:, scaling_cols] = x[scaling_cols] / (rolling_std + 1e-8)
    elif cfg.scaler_type == "zscore":
        global_std = x[scaling_cols].std()
        std_floor = 0.05 * global_std
        safe_std = rolling_std.clip(lower=std_floor, axis=1) # std_floor prevents unrealistic z-score explosions
        x_scaled = x.copy()
        x_scaled.loc[:, scaling_cols] = (
            (x[scaling_cols] - rolling_mean) / safe_std
        ).clip(-10, 10) # z_clip prevents any remaining extreme input from dominating the model
    else:
        raise ValueError(f"Unknown scaler_type: {cfg.scaler_type}")

    # --- Drop first rows where rolling stats are NaN ---
    x_scaled = x_scaled[min_period:]
    rolling_mean = rolling_mean[min_period:]
    rolling_std  = rolling_std[min_period:]

    # --- Back to series if y ---
    if single_col:
        x_scaled = x_scaled.iloc[:, 0]
        rolling_mean = rolling_mean.iloc[:, 0]
        rolling_std  = rolling_std.iloc[:, 0]

    return x_scaled, rolling_mean, rolling_std


def build_dataloaders_rolling_test(
    X_df:pd.DataFrame, 
    y_series:pd.Series,
    cfg:PrepConfig,
    min_train_size:int = 3,
    max_folds:int = 3,
    batch_size:int=256, 
    is_shuffle_training:bool=True
):
    """
    - Split data into: 3 (training + validation) walk-forward expanding-window folds + test set
    - Split into batches 
    - Scale/Transform inputs and outputs according to 
    - Optional: Shuffle training data (which is fine since we work with a non-sequential model, e.g. not RNN)
    Args:
    - is_shuffle_training: whether to shuffle training data
    """
    
    # Assert indexes are DatetimeIndex and sorted
    assert isinstance(X_df.index, pd.DatetimeIndex), f"X_df index is {type(X_df.index)}"
    assert isinstance(y_series.index, pd.DatetimeIndex), f"y_series index is {type(y_series.index)}"
    assert X_df.index.is_monotonic_increasing, "X_df index is not sorted!"
    assert y_series.index.is_monotonic_increasing, "y_series index is not sorted!"    

    # Import cfg
    folding_strategy = cfg.folding_strategy  
    is_tar_direct = cfg.is_directional_target

    # ----------------------------------------------------------------------------------
    # NaN management 2 + X-y alignement
    # ----------------------------------------------------------------------------------
    # Debug
    print("NaNs per X column:")
    print(X_df.isna().sum()[lambda s: s > 0].sort_values(ascending=False))
    print("NaNs in y:")
    print(y_series.isna().sum())
    # ----

    # Keep only rows where both X and y have no NaN
    mask_no_nan = X_df.notna().all(axis=1) & y_series.notna()
    X_df = X_df.loc[mask_no_nan]
    y_series = y_series.loc[mask_no_nan]
    print(f"Dropped {len(mask_no_nan) - mask_no_nan.sum()}/{len(mask_no_nan)} ({(len(mask_no_nan) - mask_no_nan.sum())/len(mask_no_nan)*100:.2f}%) rows containing NaNs")
    assert X_df.index.equals(y_series.index), "X and y indexes do not match!"
    
    # Debug
    # with open(cfg.debug_dir / "X_clean.txt", "w") as f:
    #     f.write(X_df.to_string())
    # with open(cfg.debug_dir / "y_clean.txt", "w") as f:
    #     f.write(y_series.to_string())
    # with open(cfg.debug_dir / "X_cols.txt", "w", encoding="utf-8") as f:
    #     f.write("\n".join(map(str, X_df.columns)))
    # ---

    # Define training-validation folds + test set
    dates = X_df.index.to_numpy()
    years = dates.astype("datetime64[Y]").astype(int) + 1970

    unique_years, year_start_idxs = np.unique(years, return_index=True)
    year_end_idxs = np.r_[year_start_idxs[1:] - 1, len(years) - 1]

    # (train_start, train_end, val_start, val_end, test_start, test_end)  
    if folding_strategy == "rolling":
        get_train_start_idx = lambda i: year_start_idxs[i - min_train_size + 1]
    elif folding_strategy == "expanding":
        get_train_start_idx = lambda i: year_start_idxs[0]
    else:
        raise ValueError(f"folding_strategy: {folding_strategy} unkown")
    
    folds = [
        (
            get_train_start_idx(i),
            year_end_idxs[i],
            year_start_idxs[i + 1],
            year_end_idxs[i + 1],
            year_start_idxs[i + 2],
            year_end_idxs[i + 2],
        )
        for i in range(min_train_size - 1, len(unique_years) - 2)
    ]

    # Combine last 2 test years
    curr_year = pd.Timestamp.today().year
    if (
        len(folds) >= 2
        and unique_years[-1] == curr_year
    ):
        prev_fold = folds[-2]
        last_fold = folds[-1]
        # Keep the train/val from the previous fold,
        # but extend its test through the unfinished current year.
        folds = folds[:-2] + [(
            prev_fold[0],  # train_start
            prev_fold[1],  # train_end
            prev_fold[2],  # val_start
            prev_fold[3],  # val_end
            prev_fold[4],  # test_start
            last_fold[5],  # test_end
        )]

    # Limit max number of folds
    if max_folds is not None:
        folds = folds[-max_folds:]
    
    # # ELIMINATE TEST SET TO KEEP IT ONLY FOR POST HYPERPARAM TUNING
    # folds = folds[:-1]

    # Printing
    dates_dt: np.ndarray = X_df.index.to_pydatetime()   # type: ignore
    for i, (t_start, t_end, v_start, v_end, te_start, te_end) in enumerate(folds):
        train_start_date = dates_dt[t_start].date()     # type: ignore
        train_end_date = dates_dt[t_end].date()         # type: ignore
        val_start_date = dates_dt[v_start].date()       # type: ignore
        val_end_date = dates_dt[v_end].date()           # type: ignore
        test_start_date = dates_dt[te_start].date()       # type: ignore
        test_end_date = dates_dt[te_end].date()           # type: ignore
        print(f"Fold {i+1}:")
        print(f"\tTrain:\t{train_start_date} -> {train_end_date} ({t_end - t_start + 1} days)")
        print(f"\tVal:\t{val_start_date} -> {val_end_date} ({v_end - v_start + 1} days)")
        print(f"\tTest:\t{test_start_date} -> {test_end_date} ({te_end - te_start + 1} days)")

    # Build loaders for each fold
    X = torch.tensor(X_df.to_numpy(), dtype=torch.float32, device=cfg.device)
    if is_tar_direct:
        y = torch.tensor(
            y_series.to_numpy(dtype=np.int64),
            dtype=torch.long,
            device=cfg.device,
        )
    else:
        y = torch.tensor(y_series.to_numpy(), dtype=torch.float32, device=cfg.device)

    fold_loaders = []
    test_loaders = []
    for i, (train_start, train_end, val_start, val_end, test_start, test_end) in enumerate(folds):      
        X_train = X[train_start:train_end+1]
        y_train = y[train_start:train_end+1]
        X_val = X[val_start:val_end+1]
        y_val = y[val_start:val_end+1]
        # Get fold span (train + val)
        fold_span = (dates_dt[train_start].date(), dates_dt[val_end].date()) # type: ignore
        X_test = X[test_start:test_end+1]
        y_test = y[test_start:test_end+1]
        dates_test = dates_dt[test_start:test_end+1]
        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=is_shuffle_training
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )
        fold_loaders.append((train_loader, val_loader, fold_span))
        test_loaders.append((X_test, y_test, dates_test))
    
    return fold_loaders, test_loaders


# def build_dataloaders(
#     X_df:pd.DataFrame, 
#     y_series:pd.Series,
#     cfg:PrepConfig,
#     batch_size:int=256, 
#     is_shuffle_training:bool=True
# ):
#     """
#     - Split data into: 3 (training + validation) walk-forward expanding-window folds + test set
#     - Split into batches 
#     - Scale/Transform inputs and outputs according to 
#     - Optional: Shuffle training data (which is fine since we work with a non-sequential model, e.g. not RNN)
#     Args:
#     - is_shuffle_training: whether to shuffle training data
#     """
    
#     # Assert indexes are DatetimeIndex and sorted
#     assert isinstance(X_df.index, pd.DatetimeIndex), f"X_df index is {type(X_df.index)}"
#     assert isinstance(y_series.index, pd.DatetimeIndex), f"y_series index is {type(y_series.index)}"
#     assert X_df.index.is_monotonic_increasing, "X_df index is not sorted!"
#     assert y_series.index.is_monotonic_increasing, "X_df index is not sorted!"    

#     # Assert inputs and target have same indices
#     assert X_df.index.equals(y_series.index), "X and y indexes do not match!"

#     # Define training-validation folds + test set
#     dates = X_df.index.to_numpy()
#     years = dates.astype("datetime64[Y]").astype(int) + 1970
#     dataset_start, dataset_end = years.min(), years.max()
#     year_to_last_idx = {
#         y: np.where(years <= y)[0][-1] 
#         for y in range(dataset_start, dataset_end+1)
#     }
#     # (train_start, train_end, val_start, val_end)
#     test_start_year = dataset_end-2
#     folds = [(0, year_to_last_idx[y], year_to_last_idx[y]+1, year_to_last_idx[y+1]) for y in range(dataset_start, test_start_year)]
#     folds = folds[-3:]
#     test_range = (year_to_last_idx[test_start_year]+1, year_to_last_idx[dataset_end])

#     # Printing
#     dates_dt: np.ndarray = X_df.index.to_pydatetime()
#     for i, (t_start, t_end, v_start, v_end) in enumerate(folds):
#         train_start_date = dates_dt[t_start].date()     # type: ignore
#         train_end_date = dates_dt[t_end].date()         # type: ignore
#         val_start_date = dates_dt[v_start].date()       # type: ignore
#         val_end_date = dates_dt[v_end].date()           # type: ignore
#         print(f"Fold {i+1}:")
#         print(f"\tTrain:\t{train_start_date} -> {train_end_date} ({t_end - t_start + 1} days)")
#         print(f"\tVal:\t{val_start_date} -> {val_end_date} ({v_end - v_start + 1} days)")
#     test_start_date = dates_dt[test_range[0]].date()    # type: ignore
#     test_end_date = dates_dt[test_range[1]].date()      # type: ignore
#     print(f"Test set:\n\t\t{test_start_date} -> {test_end_date} ({test_range[1] - test_range[0] + 1} days)")

#     # Build loaders for each fold
#     X = torch.tensor(X_df.to_numpy(), dtype=torch.float32, device=cfg.device)
#     y = torch.tensor(y_series.to_numpy(), dtype=torch.float32, device=cfg.device)

#     fold_loaders = []
#     for i, (train_start, train_end, val_start, val_end) in enumerate(folds):      
#         X_train = X[train_start:train_end+1]
#         y_train = y[train_start:train_end+1]
#         X_val = X[val_start:val_end+1]
#         y_val = y[val_start:val_end+1]

#         train_loader = DataLoader(
#             TensorDataset(X_train, y_train),
#             batch_size=batch_size,
#             shuffle=is_shuffle_training
#         )
        
#         val_loader = DataLoader(
#             TensorDataset(X_val, y_val),
#             batch_size=batch_size,
#             shuffle=False
#         )
        
#         # Get fold span (train + val)
#         fold_span = (dates_dt[train_start].date(), dates_dt[val_end].date()) # type: ignore

#         fold_loaders.append((train_loader, val_loader, fold_span))
    
#     # Test set
#     dates_test = dates_dt[test_range[0]:test_range[1]+1]
#     X_test = X[test_range[0]:test_range[1]+1]
#     y_test = y[test_range[0]:test_range[1]+1]
    
#     return fold_loaders, dates_test, X_test, y_test


"""
DATA ACQUISITION
"""

@timeit
def get_nominal_prices(
    C5TC_nominal_df: pd.DataFrame,
    cfg: PrepConfig,
) -> pd.DataFrame:
    """
    Cape 5TC index nominal value for multiple tenors
    """
    # Unpack config
    fig_dir = cfg.fig_dir
    nominal_tenors = cfg.nominal_tenors
    tenor_roll_days: int = cfg.tenor_roll_days

    trading_days: pd.DatetimeIndex = pd.DatetimeIndex(
        C5TC_nominal_df["date"].unique()
    ).sort_values()

    C5TC_nominal_df["tenor"] = None
    for tenor in nominal_tenors:
        tenor_periods = map_tenor_period(trading_days, tenor, tenor_roll_days)
        mask = C5TC_nominal_df["period"].eq(
            C5TC_nominal_df["date"].map(tenor_periods)
        )
        C5TC_nominal_df.loc[mask, "tenor"] = tenor

    # # --- Debug
    # first_ticker_by_tenor = (
    #     C5TC_nominal_df
    #     .dropna(subset=["tenor", "ticker_identifier"])
    #     .groupby("tenor")["ticker_identifier"]
    #     .first()
    # )
    # bad_rows = C5TC_nominal_df[
    #     C5TC_nominal_df["tenor"].notna()
    #     & C5TC_nominal_df["ticker_identifier"].ne(
    #         C5TC_nominal_df["tenor"].map(first_ticker_by_tenor)
    #     )
    # ]
    # print(
    #     C5TC_nominal_df
    #     .dropna(subset=["tenor", "ticker_identifier"])
    #     .groupby("tenor")["ticker_identifier"]
    #     .nunique()
    # )
    # print(
    #     bad_rows[["date", "period", "tenor", "ticker_identifier"]]
    #     .sort_values(["tenor", "date"])
    # )
    # print(
    #     "\n".join(
    #         sorted({
    #             pd.Timestamp(d).strftime("%m-%d")
    #             for d in bad_rows["date"].dropna().unique()
    #         })
    #     )
    # )
    # # ---

    pivoted = (
        C5TC_nominal_df.dropna(subset=["tenor"])
        .drop_duplicates(subset=["date", "tenor"])
        # .pivot(index="date", columns="tenor", values=["value", "period"])
        .pivot(index="date", columns="tenor", values="value")
        .sort_index()
    )
    # print(pivoted.head(50))
    
    # Plot all tenors in one function
    if cfg.is_plot:
        plot_multiple_series(
            series_list=[(v, k) for k, v in pivoted.items()],
            filename=fig_dir / f"C5TC_nominal_tr{tenor_roll_days}.png",
            title="C5TC Nominal",
            is_show_missing=cfg.is_show_missing,
        )

    return pivoted


# def get_nominal_prices(cal_date_range: pd.DatetimeIndex, cfg: PrepConfig):
#     """
#     Cape 5TC index nominal value for multiple tenors
#     """
#     # Unpack config
#     in_dir = cfg.in_dir
#     out_dir = cfg.out_dir
#     start = cfg.start
#     end = cfg.end
#     nominal_tenors = cfg.nominal_tenors

#     # Read all CSVs and combine into a single DataFrame
#     C5TC_nominal_df = pd.concat(
#         [
#             pd.read_csv(in_dir / f"C5TC_{tenor}_nominal.csv", parse_dates=["date"], index_col="date")["value"]
#             .rename(tenor)
#             for tenor in nominal_tenors
#         ],
#         axis=1
#     )

#     # Reindex the whole DataFrame to date range of interest
#     C5TC_nominal_df = C5TC_nominal_df.reindex(cal_date_range)

#     # Plot all tenors in one function
#     plot_multiple_series(
#         series_list=[(v, k) for k, v in C5TC_nominal_df.items()],
#         filename=out_dir / f"C5TC_nominal_{start}-{end}.png",
#         title="C5TC Nominal"
#     )

#     return C5TC_nominal_df


@timeit
def get_returns(
    C5TC_nominal_df: pd.DataFrame,
    cfg: PrepConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cape 5TC index nominal value for multiple tenors
    """
    # Unpack config
    fig_dir = cfg.fig_dir
    nominal_tenors = cfg.nominal_tenors
    tenor_roll_days: int = cfg.tenor_roll_days

    pivot_df = (
        C5TC_nominal_df
        .pivot(index="date", columns="period", values="value")
        .sort_index()
    )
    
    def lookup_prices(row_dates: pd.DatetimeIndex, periods: pd.Series) -> pd.Series:
        rows = pivot_df.loc[row_dates]
        col_idx = pivot_df.columns.get_indexer(periods.values)

        return pd.Series(
            rows.to_numpy()[range(len(periods)), col_idx],
            index=periods.index,
        )
    
    trading_days = pd.DatetimeIndex(
        pd.to_datetime(C5TC_nominal_df["date"].dropna().unique())
    ).sort_values()

    ret_dict = {}
    for tenor in nominal_tenors:
        periods = map_tenor_period(trading_days, tenor, tenor_roll_days)
        today_price = lookup_prices(periods.index, periods)
        prev_dates = pd.Series(periods.index, index=periods.index).shift(1).dropna()
        yesterday_price = lookup_prices(prev_dates, periods.loc[prev_dates.index])

        # check = pd.DataFrame({
        #     "date": periods.index,
        #     "prev_date": prev_dates,
        #     "today_price": today_price,
        #     "yesterday_price": yesterday_price,
        #     "diff": today_price.loc[prev_dates.index] - yesterday_price,
        #     "period": periods.values,
        # })
        # print(f"\n{tenor}")
        # print(check.head(35))
        # print(check.tail(35))

        ret_dict[tenor] = today_price.loc[prev_dates.index] - yesterday_price

    df_ret = pd.concat(ret_dict, axis=1)
    # date_range = pd.date_range(df_ret.index.min(), df_ret.index.max(), freq="D")
    # df_ret = df_ret.reindex(date_range)

    df_cumul_ret = df_ret.cumsum()

    if cfg.is_plot:
        # Plot all tenors in one function
        plot_multiple_series(
            series_list=[(v, k) for k, v in df_ret.items()],
            filename=fig_dir / f"C5TC_returns_tr{tenor_roll_days}.png",
            title=f"C5TC returns per tenor",
            is_show_missing=cfg.is_show_missing,
        )

        plot_multiple_series(
            series_list=[(v, k) for k, v in df_cumul_ret.items()],
            filename=fig_dir / f"C5TC_cumul_ret_tr{tenor_roll_days}.png",
            title=f"C5TC cumulated returns per tenor",
            is_show_missing=cfg.is_show_missing,
        )    

    return df_ret, df_cumul_ret


# def get_cumul_ret(cfg) -> Tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
#     """
#     Get cumulated returns (aka adjusted price) for all calendar days within time range considered.
#     """
#     # Unpack config
#     in_dir = cfg.in_dir
#     out_dir = cfg.out_dir
#     target_tenor = cfg.target_tenor
#     start = cfg.start
#     end = cfg.end

#     # Load price data for the selected contracts
#     cumul_ret_path = in_dir / f"C5TC_{target_tenor}_{start}-{end}.csv"
#     if not cumul_ret_path.exists():
#         cumul_ret = universe.UNIVERSE["C5TC_" + target_tenor].price(
#             what="settlement",
#             start_date=start,
#             end_date=end
#         ).ts

#         cumul_ret.to_csv(cumul_ret_path)

#     else:
#         cumul_ret = pd.read_csv(cumul_ret_path, index_col=0, parse_dates=True)

#     # Define time span based on available prices for C5TC
#     adj_start=cumul_ret.index.min()
#     adj_end=cumul_ret.index.max()

#     cal_date_range = pd.date_range(start=adj_start, end=adj_end, freq="D")
#     biz_date_range = pd.date_range(start=adj_start, end=adj_end, freq="B")

#     # Print number of missed business days
#     cumul_ret_miss_dates = [date_range.difference(cumul_ret.index) for date_range in [cal_date_range, biz_date_range]]
#     print(f"Business days missing in adjusted price: {len(cumul_ret_miss_dates[1])}.")
    
#     # fill target with missing cal days
#     cumul_ret = cumul_ret.reindex(cal_date_range)

#     # Plot cumulated returns
#     plot_series(
#         series=cumul_ret,
#         filename=out_dir / f"C5TC_{target_tenor}_cumul_ret_{start}-{end}.png",
#         title=f"C5TC {target_tenor} cumulated returns {start}-{end}"
#     )

#     return cumul_ret.squeeze(), adj_start, adj_end


def make_target(
        target_rets: pd.Series, 
        target_nominal: pd.Series, 
        cfg: PrepConfig, 
        y_smooth_win: int = 31
    ) -> pd.Series:

    # Unpack config
    tenor = cfg.target_tenor
    win = cfg.target_win
    is_ewm = cfg.is_ewm_target
    is_perc = cfg.is_perc_target
    ewm_max_span = cfg.ewm_max_span_target
    is_direct = cfg.is_directional_target
    direct_thresh = cfg.directional_threshold_target

    if is_ewm:
        assert ewm_max_span is not None, "Need to specify ewm_max_span_target"
    if is_direct:
        assert direct_thresh is not None, "Need to specify directional_threshold_target"

    cal_date_range = pd.date_range(
        target_rets.index.min(),
        target_rets.index.max(),
        freq="D",
        name=target_rets.index.name,
    )
    target_rets = target_rets.reindex(cal_date_range).copy()

    # Define target
    # print(target_rets.head(10))
    # print(f"target:\n{rolling_frontward_eff_mean(target_rets, 2, 1).head(10)}")
    if is_ewm:
        y:pd.Series = rolling_frontward_eff_ewm(target_rets, win, n=ewm_max_span)
    else:
        y:pd.Series = rolling_frontward_eff_mean(target_rets, win)
    # y:pd.Series = rolling_frontward_nan_mean(target_rets, 1, target_win)

    if is_perc:
        y = y / shift_non_nan(target_nominal.reindex(target_rets.index), 1).ffill()  # Series
        y *= 100

    if is_direct:
        if is_perc:
            y_dir = pd.Series(pd.NA, index=y.index, dtype="Int64")

            valid = y.notna()

            y_dir[valid & (y < -direct_thresh)] = 0
            y_dir[valid & (y >= -direct_thresh) & (y <= direct_thresh)] = 1
            y_dir[valid & (y > direct_thresh)] = 2

            y = y_dir
        else:
            raise ValueError("is_perc must be True when is_direct is True (contrary case not implemented)")

    # df = pd.concat(
    #     [
    #         target_nominal.rename("eod_nominal"),
    #         target_rets.rename("eod_returns"),
    #         y.rename(f"target_{win}")
    #     ],
    #     axis=1,
    #     join="outer"
    # )
    # print(df.tail(50))

    # Ploting
    if cfg.is_plot:
        y_clean = y.dropna()
        y_smooth = pd.Series(
            savgol_filter(y_clean.values, y_smooth_win, 2),
            index=y_clean.index
        )
        plot_multiple_series(
            series_list=[(y, "Original"), (y_smooth, f"{y_smooth_win}-day smoothed")],
            filename=cfg.fig_dir / f"Target_{tenor}_span{win}_ewm{is_ewm}_perc{is_perc}.png",
            title=f"C5TC {tenor} {win}-day averaged cumulated returns",
            is_show_missing=cfg.is_show_missing,
        )

    return y


def get_other_balt_indices(cfg):
    """
    Other capesize baltic indices
    """
    C_balt_inds=pd.DataFrame()
    for instrument in ["C2", "C3", "C5", "C7", "C8_182", "C9_182", "C10_182", "C14_182", "C16_182", "C17"]:
        # get right tenor
        pass
    return C_balt_inds


def get_moments(
        data: pd.Series, 
        data_label, 
        cfg, 
    ) -> pd.DataFrame:
    """
    Get weekly, bi-weekly, and monthly MAs, EMAs, and volatility
    """

    # Unpack config
    fig_dir = cfg.fig_dir
    moments_funcs = cfg.moments_funcs
    roll_wins = cfg.moments_roll_wins

    moments_df = pd.DataFrame(index=data.index)

    for type_mom, func in moments_funcs.items():
        df = pd.DataFrame({
            make_moments_name(data_label, type_mom, win): func(data, win) 
            for win in roll_wins
        })

        # Plot using your multi-series plotting function
        if cfg.is_plot:
            plot_multiple_series(
                series_list=[(data, "Original")] + [(df[col], col) for col in df.columns],
                filename=fig_dir / f"{data_label}_{type_mom}.png",
                title=f"{type_mom} on {data_label}",
                is_show_missing=cfg.is_show_missing,
            )

        moments_df = pd.concat([moments_df, df], axis=1)
    
    return moments_df


def make_moments_name(data_label: str, type_mom: str, win: int):
    return f"{data_label.upper()}_{win}-DAY_{type_mom.upper()}"


def deseasonalize_var(
        data: pd.Series,
        data_label,
        cfg,
    ) -> pd.DataFrame:
    """
    Get yearly deseasonalized series: x[t] - avg(x around t-i years)
    """
    # Unpack config
    fig_dir = cfg.fig_dir
    win = cfg.win_deseasonalize
    n = cfg.nb_year_seasn

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex")

    data = data.sort_index()

    if data.index.has_duplicates:
        data = data.groupby(level=0).mean()

    deseaz_df = pd.DataFrame(index=data.index)
    # seasonal_avg_df = pd.DataFrame(index=data.index)

    for i in range(1, n + 1):
        prev_year_data = data.copy()
        prev_year_data.index = prev_year_data.index + pd.DateOffset(years=i)

        # Manage Feb 28
        if prev_year_data.index.has_duplicates:
            prev_year_data = prev_year_data.groupby(level=0).mean()

        prev_avg = (
            prev_year_data
            .reindex(data.index)
            .rolling(f"{win}D", center=True, min_periods=1)
            .mean()
        )

        # avg_col = f"{data_label}_{i}-YEAR_{win}-DAY_SEASONAL_AVG"
        # seasonal_avg_df[avg_col] = prev_avg

        col = make_deseasonal_name(data_label, i, win)
        deseaz_df[col] = data - prev_avg

    if cfg.is_plot:
        plot_multiple_series(
            series_list=[
                (data, "Original"),
            ] 
            # + [(seasonal_avg_df[col], col) for col in seasonal_avg_df.columns]
            + [(deseaz_df[col], col) for col in deseaz_df.columns],
            filename=fig_dir / f"{data_label}_deseasonalized.png",
            title=f"Deseasonalized {data_label}",
            is_show_missing=cfg.is_show_missing,
        )

    return deseaz_df


def make_deseasonal_name(data_label: str, compar_year: int, win_deseasonalize: int):
    return f"{data_label.upper()}_{compar_year}-YEAR_{win_deseasonalize}-DAY_DESEASONALIZED"


# ---------------
#  SIGNAL OCEAN 
# ---------------
def get_fixtures_df(cal_date_range, cfg, filename="fixtures_2015_2026.txt"):
    """
    Signal Ocean Reported Fixtures
    """
    # Unpack config
    in_dir = cfg.in_dir
    fig_dir = cfg.fig_dir

    # Read CSV
    fixtures_df = pd.read_csv(
        in_dir / filename,
        sep="\t",
        header=0,
        parse_dates=["Laycan From", "Laycan To"],
    )

    # Clean Laycan columns
    fixtures_df = fixtures_df.dropna(subset=["Laycan From", "Laycan To"], how="all")
    fixtures_df["Laycan To"] = fixtures_df["Laycan To"].fillna(fixtures_df["Laycan From"])
    fixtures_df["Laycan range"] = (fixtures_df["Laycan To"] - fixtures_df["Laycan From"]).dt.days.fillna(1)
    fixtures_df.sort_values(by=["Laycan To", "Laycan From"], inplace=True)
    fixtures_df.set_index("Laycan To", inplace=True)

    # Keep only date indexes in range of interest
    fixtures_df = fixtures_df.reindex(cal_date_range)

    # Extract numeric rate and unit
    fixtures_df["Rate numeric"] = fixtures_df["Rate"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    fixtures_df["Rate unit"] = fixtures_df["Rate"].astype(str).str.extract(r"(\$/day|\$/ton)")[0]

    # Rows with Rate unit not $/day or $/ton
    fixt_other_units = fixtures_df[~fixtures_df["Rate unit"].isin(["$/day", "$/ton"])]["Rate unit"].drop_duplicates()
    print(f"Other units in reported fixtures: {fixt_other_units.values}")

    # Filter out crazy jumps (>50x previous) per unit, build combined DataFrame, and plot
    filt_fixt_rates = []
    for unit in ["$/day", "$/ton"]:
        rate = fixtures_df.loc[fixtures_df["Rate unit"] == unit, "Rate numeric"]
        rate = rate[rate <= 50 * rate.shift(1).fillna(rate.iloc[0])]
        filt_fixt_rates.append(pd.DataFrame({"Rate": rate, "Unit": unit}))
        
        if cfg.is_plot:
            plot_series(
                series=rate,
                filename=fig_dir / f"rates_fixtures_{unit.replace('$','').replace('/','')}.png",
                title=f"Rates of reported fixtures ({unit})",
                is_show_missing=cfg.is_show_missing,
            )
    
    return fixtures_df


# def get_congestion(cfg: PrepConfig) -> pd.DataFrame:
#     """
#     Congestion at main cape routes ports
#     """
        
#     # Unpack config
#     in_dir = cfg.in_dir
#     fig_dir = cfg.fig_dir
#     ports_of_interest = cfg.congestion_ports

#     congestion_df = pd.concat(
#         [
#             pd.read_csv(
#                 in_dir / f"congestion_capes_{country}.csv",
#                 index_col=0,
#                 sep="\t",
#                 parse_dates=True
#             ).iloc[:, 0]  # convert to Series
#             # .reindex(cal_date_range)  # align to calendar
#             .rename(country)          # name the column as the country
#             for country in ports_of_interest
#         ],
#         axis=1
#     )
#     if cfg.is_plot:        
#         plot_multiple_series(
#             series_list=[(v, k) for k, v in congestion_df.items()],
#             filename=fig_dir / "congestion_capes.png",
#             title="Congestion Capes",
#             is_show_missing=cfg.is_show_missing,
#         )
    
#     return congestion_df


def plot_signal_ocean_ts(
        label: str, 
        series: pd.Series, 
        cfg: PrepConfig, 
        smooth_win:int=30
    ):

    fig_dir = cfg.fig_dir
    smooth = pd.Series(savgol_filter(series.ffill(), smooth_win, 2), index=series.index)
    plot_multiple_series(
        series_list=[
            (series, "Original"), 
            (smooth, f"{smooth_win}-day smoothed")
        ],
        filename=fig_dir / f"{label}.png",
        title=f"{label.replace('_', ' ')}",
        is_show_missing=cfg.is_show_missing,
    )

@timeit
def get_commodities_df(cfg):
    """
    COMMODITIES and BUNKER
    df['tenor'] format is "%b%y"
    """    
    # Unpack config
    in_dir = cfg.in_dir
    fig_dir = cfg.fig_dir
    target_tenor = cfg.target_tenor
    tenor_roll_days = cfg.tenor_roll_days

    df = load_commodities()

    trading_days: pd.DatetimeIndex = pd.DatetimeIndex(
        df["date"].unique()
    ).sort_values()

    # Use 'settlement' if not NaN, otherwise 'close'
    df['value'] = df['settlement'].combine_first(df['close'])
    df=df.drop(columns=["close", "settlement", "volume"])

    tenor_periods = map_tenor_period(trading_days, target_tenor, tenor_roll_days)
    formatted_periods = tenor_periods.apply(period_to_months)
    # print(df["tenor"].unique()) # For debugging
    formatted_periods_df = formatted_periods.explode().reset_index()
    formatted_periods_df.columns = ["date", "tenor"]
    # print(formatted_periods_df)

    com_types = ["fuel_price", "dry_price", "dry2_price"]

    df_wide_dict={}
    for t in com_types:
        df_t=df[df['commodity_group'] == t]
        df_t=df_t.drop(columns=["commodity_group"])
        
        formatted_periods_df_t = formatted_periods_df.merge(
            df_t,
            on=["date", "tenor"],
            how="left"
        )

        df_avg = (
            formatted_periods_df_t
            .groupby(["date", "exchange_symbol"], as_index=False)["value"]
            .mean()
        )

        df_wide = df_avg.pivot(index="date", columns="exchange_symbol", values="value").sort_index()
        # print(df_wide)
        # print(list(df_wide.items())[0])

        df_wide_dict[t]=df_wide
        if cfg.is_plot:            
            plot_multiple_series(
                series_list = [(v, k) for k, v in df_wide.items()],
                filename=fig_dir / f"commodities_price_{t}_{target_tenor}_tr{tenor_roll_days}.png",
                title=f"Commodity Prices: {' '.join(t.split('_')[:-1])}",
                is_show_missing=cfg.is_show_missing,
            )
    
    return df_wide_dict


def get_io_port_inv_china(cfg: PrepConfig, inv_filename = "io_port_inventories_china.csv"):
    # Unpack config
    in_dir = cfg.in_dir
    fig_dir = cfg.fig_dir

    io_port_inv_china = pd.read_csv(
        in_dir / inv_filename,
        parse_dates=["Date"],
        index_col="Date"
    ).sort_index()
    io_port_inv_china = io_port_inv_china["PX_LAST"]#.reindex(cal_date_range)

    if cfg.is_plot:
        plot_series(
            series=io_port_inv_china,
            filename=fig_dir / f"io_port_inv_china.png",
            title=f"Iron Ore Inventories in China Ports",
            is_show_missing=cfg.is_show_missing,
        )

    return io_port_inv_china


def get_seasonality_df(cal_date_range: pd.DatetimeIndex):
    """Build seasonality DataFrame with cyclical calendar features."""

    weekday = cal_date_range.weekday  # Monday=0 ... Sunday=6
    monthday = cal_date_range.day - 1  # 0-based day within month
    days_in_month = cal_date_range.days_in_month
    month = cal_date_range.month - 1  # January=0 ... December=11

    seasonality_df = pd.DataFrame({
        "WEEKDAY_SIN": np.sin(2 * np.pi * weekday / 7),
        "WEEKDAY_COS": np.cos(2 * np.pi * weekday / 7),

        "MONTHDAY_SIN": np.sin(2 * np.pi * monthday / days_in_month),
        "MONTHDAY_COS": np.cos(2 * np.pi * monthday / days_in_month),

        "MONTH_SIN": np.sin(2 * np.pi * month / 12),
        "MONTH_COS": np.cos(2 * np.pi * month / 12),
    }, index=cal_date_range)

    return seasonality_df


# --- Data cache ---
def save_raw_data_with_cfg(raw_data, cfg: PrepConfig) -> None:
    equiv_raw_data = is_raw_data_cached(cfg)

    if equiv_raw_data is not None:
        print("Current config already cached. No need to save")
        return

    C5TC_spot, nominal_tenors_df, rets_df, signal_ocean_df, io_inv_china, commodities_df = raw_data

    filename = create_new_filename(cfg.cache_dir, cfg.processed_data_filename, "h5")
    filepath = cfg.cache_dir / filename
    # Ensure output folder exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save X, y, and cfg JSON in one HDF5 file (with lock for concurrent processing)
    lock = FileLock(str(filepath) + ".lock")
    with lock:
        with pd.HDFStore(filepath, mode="a") as store:
            # store["cal_date_range"] = pd.Series(cal_date_range, name="cal_date")
            store["C5TC_spot"] = C5TC_spot
            store["nominal_tenors_df"] = nominal_tenors_df
            store["rets_df"] = rets_df
            store["signal_ocean_df"] = signal_ocean_df
            store["io_inv_china"] = io_inv_china
            store["commodities_df"] = commodities_df
            # store["seasonality_df"] = seasonality_df
            store["cfg"] = pd.Series({"json": json.dumps(cfg.to_dict())})

    print("Saved to", filepath)


def load_dataset(filepath: pathlib.Path):
    lock = FileLock(str(filepath) + ".lock") # For concurrent processing
    with lock:
        with pd.HDFStore(filepath, mode="r") as store:
            # cal_date_range = pd.DatetimeIndex(store["cal_date_range"])
            C5TC_spot = store["C5TC_spot"]
            nominal_tenors_df = store["nominal_tenors_df"]
            rets_df = store["rets_df"]
            signal_ocean_df = store["signal_ocean_df"]
            io_inv_china = store["io_inv_china"]
            commodities_df = store["commodities_df"]
            # seasonality_df = store["seasonality_df"]
            cfg_json = store["cfg"]
        
    cfg_dict = json.loads(cfg_json.loc["json"])  # Deserialize back to dict

    raw_data = C5TC_spot, nominal_tenors_df, rets_df, signal_ocean_df, io_inv_china, commodities_df

    return raw_data, cfg_dict


@timeit
def is_raw_data_cached(cfg):
    # relevant_keys = ["start", "end", "nominal_tenors", "tenor_roll_days", "congestion_ports", "target_tenor"]
    relevant_keys = ["nominal_tenors", "tenor_roll_days", "congestion_ports", "target_tenor"]
    # ignore_keys = ["model_type", "is_plot"]
    for file in cfg.cache_dir.iterdir():
        if file.suffix not in [".h5", ".hdf5"]:
            continue
        if file.name.startswith(cfg.processed_data_filename):
            raw_data, loaded_config_dict = load_dataset(file)
            # if dicts_equal_ignore_keys(cfg.to_dict(), loaded_config_dict, ignore_keys):
            if dicts_equal_spec_keys(cfg.to_dict(), loaded_config_dict, relevant_keys):
                return raw_data
    return None