from collections import Counter
from pathlib import Path 
from dataclasses import asdict, dataclass, field
import json
import torch
from typing import Optional

ROOT = Path(__file__).resolve().parent

FUEL_SYMBOLS = {"CL", "BRN", "HO", "RB"}

DRY_SYMBOLS = {
    "IOE", "ZC", "ZW", "ZS", "ZL", "HG",
    "LMAHDS03", "LMCADS03", "LMNIDS03", "LMPBDS03", "LMZSDS03",
}

SEC_DRY_SYMBOLS = {"KC", "CC", "SB", "PL"}

@dataclass
class BaseConfig:
    def to_dict(self):
        return asdict(self)
        
    def save(self, path: Path):
        """Save THIS instance's config to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    # classmethod to create instance of right sub-class
    @classmethod
    def load(cls, path: Path):
        """Load config from file into this config class."""
        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)

@dataclass
class PrepConfig(BaseConfig):
    # -----------------
    # Target
    # -----------------
    target_win: int = 3
    target_tenor: str = "m2"
    is_perc_target: bool = True
    is_ewm_target:bool = True
    ewm_max_span_target: Optional[int] = 0 # 0 makes infinite span
    is_directional_target: bool = False
    directional_threshold_target: Optional[float] = 0.1 # in % 0.1 --> 0.001

    # -----------------
    # Trading
    # -----------------
    direction_prob_threshold: float = .1

    # -----------------
    # Training
    # -----------------
    model_type: str= "mlp" # mlp, xgboost
    folding_strategy: str = "rolling" # rolling, expanding

    # -----------------
    # Data processing
    # -----------------
    scaler_type: str = "zscore" # none, std, zscore
    prediction_days_of_week: list[int] = field(default_factory=lambda: [0,1,2,3,4]) # example: 0=Monday, 4=Friday
    is_log_target: bool = False
    window_scale: int = 365*3
    # window_scale: int = 30
    min_period_scale: int = 7
    win_deseasonalize: int = 7
    nb_year_seasn: int = 2
    is_fleet_normalized: bool = False
    keep_groups: Optional[list[bool]] = None
    # -----------------
    # Data acquisition
    # -----------------
    # Time range
    start: str = str(2018)
    end: str = str(2025)

    # Nominal tenors
    tenor_roll_days: int = 1 # 1 = switch on last day of the month
    nominal_tenors: list[str] = field(default_factory=lambda: ["m1", "m2", "q1"]) # ["m1", "m2", "q1", "q2", "y1"]

    # Congestion
    congestion_ports: list[str] = field(default_factory=lambda: ["china", "australia", "brazil", "guinea"])

    # Map indicator (~ moments) names to actual functions: dict[str, Callable[[pd.Series, int], pd.Series]]
    is_moments_ignore_na: bool = True # If True Fri counts as much for Mon as Mon does for Tue
    moments_roll_wins: list[int] = field(default_factory=lambda: [7, 14, 30])
    # moments_roll_wins: list[int] = field(default_factory=lambda: [3, 7, 14, 30])
    mom_rolling_min_period: Optional[int] = None

    moments_func_names: list[str] = field(default_factory=lambda: ["EMA"]) # Store JSON-friendly names in the config    
    moments_funcs: dict = field(init=False) # Runtime-only callables, not saved/loaded directly

    # --------------------
    # Directory structure
    # --------------------
    in_dir = ROOT / "inputs"
    out_dir = ROOT / "outputs"
    cache_dir = ROOT / "cache"
    debug_dir = ROOT / "debug"
    fig_dir = out_dir / "figs"
    optuna_db_dir = out_dir / "optuna"
    processed_data_filename = "processed_data"
   
    # --------------------
    # OTHERS
    # --------------------
    # Ploting parameter
    is_plot: bool = False
    is_show_missing: bool = False
    
    # Torch parameters
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    def to_dict(self) -> dict:
        cfg_dict = asdict(self)
        
        # Convert paths to strings
        for k in ["in_dir", "out_dir"]:
            if k in cfg_dict:
                cfg_dict[k] = str(cfg_dict[k])
        
        # functions/lambdas are runtime-only
        cfg_dict.pop("moments_funcs", None)
        
        return cfg_dict

    def __post_init__(self):
        # Ensures nominal of target tenor is included
        assert self.target_tenor.lower() in map(str.lower, self.nominal_tenors)

        # Ensures directories exist
        for dir in [
            self.in_dir,
            self.out_dir,
            self.cache_dir,
            self.debug_dir,
            self.fig_dir,
            self.optuna_db_dir,
        ]:
            if not dir.exists():
                dir.mkdir(exist_ok=True)

        if self.mom_rolling_min_period is None:
            self.mom_rolling_min_period = min(self.moments_roll_wins)

        available_moments_funcs = {
                "MA": lambda ret, win: ret.rolling(window=win, min_periods=self.mom_rolling_min_period).mean(),     # simple moving average
                "EMA": lambda ret, win: ret.ewm(span=win, min_periods=self.mom_rolling_min_period, ignore_na=self.is_moments_ignore_na).mean(),            # exponential moving average
                "vol": lambda ret, win: ret.rolling(window=win, min_periods=self.mom_rolling_min_period).var()        # rolling variance (volatility)
            }
        self.moments_funcs = {
            name: available_moments_funcs[name]
            for name in self.moments_func_names
        }


@dataclass
class MLPConfig(BaseConfig):
    name: str = "mlp"
    lr: float = 1e-5            # learning rate
    patience: int = 100         # early stopping patience
    is_lr_scheduler: bool = True
    epochs: int = 750
    dropout_rate: Optional[float] = 0.5
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    l1_lambda: float = 1e-4      # L1 regularization factor
    flat_lambda: float = 1e-2    # Flatness regularization factor
    weight_decay: float = 1e-4   # L2 regularization
    momentum: float = 0.8
    loss_type: str = "MSE"       # MAE, MSE
    is_val_flat_pen: bool = True



@dataclass
class XGBoostConfig(BaseConfig):
    name: str = "xgboost"
    n_estimators: int = 5000       # more trees for better fit
    max_depth: int = 4             # slightly shallower to reduce overfitting
    learning_rate: float = 0.01    # smaller LR for more stable training
    subsample: float = 0.9         # fraction of rows per tree
    colsample_bytree: float = 0.8  # fraction of features per tree
    gamma: float = 1.0             # minimum loss reduction to make a split
    reg_alpha: float = 0.5         # L1 regularization
    reg_lambda: float = 1.0        # L2 regularization
    min_child_weight: float = 3.0  # avoids learning from noise
    n_jobs: int = -1               # use all cores
    random_state: int = 42
    early_stopping_rounds:int = 200


    def get_args(self):
        d = self.to_dict()
        d.pop("name", None)
        return d

