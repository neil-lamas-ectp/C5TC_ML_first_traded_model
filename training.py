from collections import defaultdict, deque
from pprint import pprint
from dataclasses import asdict
from torchview import draw_graph
import copy
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from typing import Tuple, cast

from utils.train_utils import TrainingTracker, add_config_to_plot, plot_val_predictions
from pathlib import Path
from datetime import datetime

from utils.general_utils import create_new_dir, create_new_filename, timeit

from .config import MLPConfig, PrepConfig, XGBoostConfig
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


@timeit
def fit_model(
    fold_loaders,
    input_dim, 
    prep_cfg: PrepConfig,
    model_cfg: MLPConfig | XGBoostConfig,
    tracking_dir: Path,
    is_save: bool = False,
) -> Tuple[torch.nn.Module | xgb.XGBRegressor, Path]:
    
    model_type = prep_cfg.model_type

    MODEL_SWITCH = {
        "mlp": {
            "cfg": MLPConfig,
            "train": lambda: train_kfold_cv_nn(
                fold_loaders, input_dim, prep_cfg, cast(MLPConfig, model_cfg), tracking_dir
            ),
        },
        "xgboost": {
            "cfg": XGBoostConfig,
            "train": lambda: train_kfold_cv_xgboost(
                fold_loaders, cast(XGBoostConfig, model_cfg)
            ),
        }
    }

    if model_type not in MODEL_SWITCH:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_cfg.name != model_type:
        raise ValueError(f"Model Configuration (model_cfg: {model_cfg}) passed must match model type (model_type: {model_type})")
    
    model = MODEL_SWITCH[model_type]["train"]()

    if is_save:
        save_model(model, tracking_dir)
    
    return model, tracking_dir


def predict(
    model: torch.nn.Module | xgb.XGBRegressor,
    X_test: torch.Tensor, 
) -> torch.Tensor:
 
    # XGBoost
    if isinstance(model, xgb.XGBRegressor):
        X_test_np = X_test if isinstance(X_test, np.ndarray) else np.array(X_test)

        y_pred = model.predict(X_test_np)
        y_pred = torch.from_numpy(np.array(y_pred))
    
    # MLP
    elif isinstance(model, torch.nn.Module):
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.from_numpy(X_test).float()
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
           
    return y_pred


def predict_directional(
    model: torch.nn.Module | xgb.XGBRegressor,
    X_test: torch.Tensor, 
):
    assert isinstance(model, torch.nn.Module), f"Unsupported model type: {type(model)}"

    device = next(model.parameters()).device

    if not isinstance(X_test, torch.Tensor):
        X_test = torch.from_numpy(X_test).float()

    X_test = X_test.to(device)

    model.eval()

    with torch.no_grad():
        logits = model(X_test)
        proba = torch.softmax(logits, dim=1)
        y_pred_class = torch.argmax(logits, dim=1)

    return y_pred_class, proba


def save_config(
        prep_cfg: PrepConfig,
        model_cfg: MLPConfig | XGBoostConfig,
        tracking_dir: Path | None = None,
    ):
    """Save config into json file"""
    
    if tracking_dir is None:
        tracking_dir = init_tracking_dir(prep_cfg)

    for k,v in {
        "prep_cfg": prep_cfg,
        "model_cfg": model_cfg,
    }.items():
        filename = create_new_filename(tracking_dir, k, "json")
        path = tracking_dir / filename
        # Save config with same basename, .json
        v.save(path)

    return tracking_dir


def load_config(tracking_dir: Path) -> Tuple[PrepConfig, MLPConfig]:
    prep_cfg = PrepConfig.load(tracking_dir / "prep_cfg_0.json")     
    model_cfg = MLPConfig.load(tracking_dir / "model_cfg_0.json")     
    return prep_cfg, model_cfg


def init_tracking_dir(prep_cfg, extra_dir = None) -> Path:
    base_path =  prep_cfg.out_dir / "train_tracking"
    if extra_dir is not None:
        base_path = base_path / extra_dir
    tracking_dir = create_new_dir(
        base_path,
        f"{datetime.now().strftime('%Y-%m-%d')}",
        extra=f"{datetime.now().strftime('%H-%M')}"
    )
    print(f"Experiment Directory Created at: {tracking_dir}")
    return tracking_dir


def save_model(model, tracking_dir: Path):
    """Save model into pth file and config into json file"""
    if isinstance(model, torch.nn.Module):
        model_name = create_new_filename(tracking_dir, "model", "pth")
        model_path = tracking_dir / model_name
        torch.save(model.state_dict(), model_path)
        print(f"PyTorch model saved to {model_path}")
    elif isinstance(model, (xgb.XGBRegressor)):
        model_name = create_new_filename(tracking_dir, "model", "ubj")
        model_path = tracking_dir / model_name        
        model.save_model(model_path)
        print(f"XGBoost model saved to {model_path}")
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
    print(f"Model saved to: {model_path}")


def load_model(
        filepath:Path,
        input_dim: int,
        model_cfg: MLPConfig | XGBoostConfig,
    ):
    model_type = model_cfg.name
    if model_type == "mlp":
        assert isinstance(model_cfg, MLPConfig), (
            f"Model Configuration (model_cfg: {model_cfg}) passed must match model type (model_type: {model_type})"
        )                
        model = MLP(input_dim, model_cfg)
        model.load_state_dict(torch.load(filepath, weights_only=True))
        model.eval()  # set to evaluation mode
    elif model_type == "xgboost":
        model = xgb.XGBRegressor()
        model.load_model(filepath)
    return model


@timeit
def train_kfold_cv_nn(
    fold_loaders, 
    input_dim, 
    prep_cfg: PrepConfig, 
    model_cfg: MLPConfig,
    tracking_dir: Path,
) -> torch.nn.Module | None:
    """
    Train model using k-fold time series cross-validation.
    
    Args:
        fold_loaders: List of (train_loader, val_loader) for each fold
    
    Returns:
        cv_results: List of results for each fold
        best_model: Model with best validation performance
    """
    cv_results = []
    best_overall_model = None
    best_overall_loss = float('inf')
    
    print("SKIPPING CALCULATION OF FIRST FOLDS FOR SPEED")
    fold_loaders = [fold_loaders[-1]]

    for fold_idx, (train_loader, val_loader, fold_span) in enumerate(fold_loaders):
        print(f"Training Fold (including val) {fold_idx + 1}/{len(fold_loaders)}: {[d.strftime('%Y-%m-%d') for d in fold_span]}")        

        # Train on this fold
        fold_result = train_with_nn(
            train_loader,
            val_loader,
            fold_span,
            input_dim,
            prep_cfg,
            model_cfg,
            tracking_dir,
        )
    
        if fold_result is None:
            print("Could not fit model for this fold")
            return None

        print(f"Training complete. Lowest {model_cfg.loss_type} loss: {fold_result['best_val_loss']:.4e} (epoch {fold_result['best_epoch']})")
        
        cv_results.append(fold_result)
        
        # Track best overall model
        if fold_result['best_val_loss'] < best_overall_loss:
            best_overall_loss = fold_result['best_val_loss']
            best_overall_model = fold_result["model"]
    #         best_fold_idx = fold_idx + 1
    
    # # Report summary
    # print("--- CROSS-VALIDATION SUMMARY ---")
    
    # val_losses = [r['best_val_loss'] for r in cv_results]
    # avg_val_loss = np.mean(val_losses)
    # std_val_loss = np.std(val_losses)
    
    # for i, result in enumerate(cv_results):
    #     print(
    #         f"Fold {i+1}: Best val MSE = {result['best_val_loss']:.3f} "
    #         f"(epoch {result['best_epoch']})"
    #     )
    
    # print(f"Average validation MSE: {avg_val_loss:.3f} ± {std_val_loss:.3f}")
    # print(f"Stability Index (for hyperparameters): {avg_val_loss/std_val_loss:.3f}")
    # print(f"Figure of Merit (avg_val_loss*std_val_loss): {avg_val_loss*std_val_loss:.3f}")
    # print(f"Best model from fold {best_fold_idx} with MSE = {best_overall_loss:.3f}")
    
    return best_overall_model


def train_with_nn(
        train_loader, 
        val_loader,
        fold_span,
        input_dim, 
        prep_cfg: PrepConfig, 
        model_cfg: MLPConfig,
        tracking_dir,
        retries_left: int = 4,
        lr_threshold: float = 1e-6
    ):
    """
    Train model for a single fold in cross-validation.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_dim: Number of input features
        prep_cfg: General configuration    
        retries_left: Number of times lr can be changed for convergence
    Returns:
        results: Dict with training metrics and metadata
    """
    # --- Tune LR across multiple iterations to achieve convergence
    conv_manager = {
        "retries_left": retries_left,
        "min": None,
        "max": None,
    }

    results = []
    model_cfg = copy.deepcopy(model_cfg)
    while conv_manager["retries_left"] > 0 and model_cfg.lr >= lr_threshold:

        run_params = input_dim, model_cfg, prep_cfg, tracking_dir, train_loader, val_loader, fold_span
        if prep_cfg.is_directional_target:
            result = train_one_directional_nn_run(*run_params)
        else:
            result = train_one_nn_run(*run_params)

        results.append(result)

        # -- Manages convergence ---
        is_diverged = result["best_epoch"] <= 50
        is_not_finished_conv = get_is_not_finished_conv(result["best_epoch"], model_cfg)

        # If converged return result
        if not(is_diverged or is_not_finished_conv):
            return result

        # If change of direction (coming back) use bisection method to adapt lr
        lr_min = conv_manager["min"]
        lr_max = conv_manager["max"]
        prev_lr = model_cfg.lr
        if is_diverged:
            if lr_max is None:
                conv_manager["max"] = model_cfg.lr
            if lr_min is None:
                model_cfg.lr *= .1
            else: # Bisection method
                model_cfg.lr = (model_cfg.lr + lr_min)/2.
        elif is_not_finished_conv:
            if lr_min is None:
                conv_manager["min"] = model_cfg.lr            
            if lr_max is None:
                model_cfg.lr *= 10.
            else: # Bisection method
                model_cfg.lr = (model_cfg.lr + lr_max)/2.
        else:
            raise Exception("Error in convergence management")

        if model_cfg.lr < lr_threshold:
            print(f"Could't make it converge even with low lr, won't go below {lr_threshold}")
            conv_manager["retries_left"] = 0

        else:
            print(f"Relaunch training with adjusted LR: prev_lr={prev_lr:.1e}, new_lr={model_cfg.lr:.1e}, retries_left={conv_manager['retries_left']}")
            conv_manager["retries_left"] -= 1
    
    print("Training did not converge too many times. Rolling Back to best model.")

    if len(results) > 0:
        best_res_idx, best_res = min(
            enumerate(results),
            key=lambda x: x[1]["best_val_loss"]
        )
        print(f"Chose {best_res_idx}-th run out of {len(results)} with lowest best_val_loss")
        return best_res
    else:
        return None


@timeit
def train_one_nn_run(
    input_dim,
    model_cfg,
    prep_cfg,
    tracking_dir,
    train_loader,
    val_loader,
    fold_span,
    is_interactive_plot = False,
):
    model = MLP(input_dim, model_cfg).to(prep_cfg.device) # Init with random weights

    # Optimizer including L2 regularization via weight decay (SGD instead of Adam because generalizes better)
    # optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=model_cfg.lr, weight_decay=model_cfg.weight_decay, momentum=model_cfg.momentum)

    if model_cfg.loss_type=="MSE":
        criterion = nn.MSELoss()
    elif model_cfg.loss_type=="MAE":
        criterion = nn.L1Loss()
    else:
        raise Exception("Unknown criterion for Loss function")
    
    # Learning rate scheduler
    if model_cfg.is_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=model_cfg.epochs,
            eta_min=model_cfg.lr/50
        )

    # Init Training Params
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    # best_model_state = copy.deepcopy(model.state_dict())
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()} # no need to deepcopy since state_dict is flat dict
    tracker = TrainingTracker(model_cfg.to_dict(), tracking_dir, fold_span = fold_span)
    epochs = range(1, model_cfg.epochs + 1)
    epochs = tqdm(epochs, desc="Training", unit="epoch") if is_interactive_plot else epochs


    def compute_huber_delta(residuals, q=0.8):
        return torch.quantile(residuals.abs(), q).item()

    def get_distrib_loss(pred, yb):
        """
        Avoids flat prediction
        """
        # comparing matching quantiles.
        p_sorted = torch.sort(pred).values
        t_sorted = torch.sort(yb).values
        
        delta = compute_huber_delta((p_sorted - t_sorted).abs().detach())
        penalty = torch.nn.functional.huber_loss(
            p_sorted, t_sorted, delta=delta, reduction="mean"
        )
        return penalty

    def is_bad(*tensors):
        return any(not torch.isfinite(t).all().item() for t in tensors)


    # --- Training loop
    # For gradient clipping
    grad_norm_history = deque(maxlen=60)
    for epoch in epochs:
        # ----- TRAINING PHASE -----
        model.train()
        train_loss = 0.0
        n_train = 0
        
        loss_contribs = defaultdict(list)
        for xb, yb in train_loader:
            # Clear old gradients before this batch. set_to_none --> .grad become None instead of zero tensors
            optimizer.zero_grad(set_to_none=True)

            # Skip batches where inputs or targets already contain NaN or +/-inf.
            if is_bad(xb, yb):
                print("\n--- Current batch has NaN --> must be corrected ! ---\n")
                continue
            
            # Run the model forward
            # pred = model(xb)
            pred = model(xb).squeeze(-1)

            # If the model produced NaN or +/-inf, skip this batch.
            if is_bad(pred):
                continue

            # Compute the main supervised loss, e.g. MSE/MAE/etc.
            base_loss = criterion(pred, yb)
            
            # L1 penalty - encourages smaller weights.
            l1_penalty = sum(p.abs().mean() for p in model.parameters())
            l1_penalty *= model_cfg.l1_lambda

            # Flatness penalty - encourages similar distribution
            distrib_penalty = get_distrib_loss(pred, yb)
            distrib_penalty *= model_cfg.flat_lambda

            # Add all loss terms into the total objective used for backprop.
            loss = base_loss + l1_penalty + distrib_penalty

            # If total loss is NaN or +/-inf, skip this batch.
            if is_bad(loss):
                continue

            # Update loss contributions
            for name, value in {
                "base_loss": base_loss,
                "l1": l1_penalty,
                "distrib": distrib_penalty,
            }.items():
                loss_contribs[name].append(value.item())

            # Compute gradients of total loss w.r.t. model parameters.
            loss.backward()

            # Clip unusually large gradient norms using the recent 90th percentile - if gradient norm is larger, all gradients are scaled down proportionally.
            if len(grad_norm_history) >= 15:
                max_norm = float(np.clip(np.percentile(grad_norm_history, 90), 1.0, 100.0))
            else:
                max_norm = 10.0
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm
            )

            if is_bad(grad_norm):
                continue

            grad_norm_history.append(float(grad_norm))

            # Update model parameters using the optimizer.
            optimizer.step()          

            # Accumulate loss
            batch_size = yb.size(0)
            train_loss += loss.item() * batch_size
            n_train += batch_size
        
        avg_train_loss = train_loss / n_train if n_train > 0 else float("nan")

        # ----- VALIDATION PHASE -----
        model.eval()
        val_loss = 0.0
        n_val = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                # Forward pass
                # pred = model(xb)
                pred = model(xb).squeeze(-1)

                loss = criterion(pred, yb)

                # Flatness penalty 
                if model_cfg.is_val_flat_pen:
                    distrib_penalty = get_distrib_loss(pred, yb)
                    distrib_penalty *= model_cfg.flat_lambda
                    loss += distrib_penalty
    
                # Accumulate loss
                batch_size = yb.size(0)
                val_loss += loss.item() * batch_size
                n_val += batch_size
        
        avg_val_loss = val_loss / n_val if n_val > 0 else 0
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update training history
        tracker.add_epoch(epoch, avg_train_loss, loss_contribs, avg_val_loss, current_lr)
        if is_interactive_plot and epoch % 5 == 0:
            tracker.plot_losses()
            tracker.plot_contribs()

        # ----- EARLY STOPPING CHECK -----
        early_stop_msg = f"Early stopping triggered at epoch {epoch}"
        if get_is_not_finished_conv(best_epoch, model_cfg):
            tqdm.write(early_stop_msg) if is_interactive_plot else print(early_stop_msg)
            break
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= model_cfg.patience:
                tqdm.write(early_stop_msg) if is_interactive_plot else print(early_stop_msg)
                break
        
        # ----- UPDATE LEARNING RATE -----
        if model_cfg.is_lr_scheduler:
            scheduler.step()
    
    if not is_interactive_plot:
        tracker.plot_losses()
        tracker.plot_contribs()

    # Load best model
    model.load_state_dict(best_model_state)
    
    # Prepare results
    results = {
        'model': model,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_losses': tracker.train_losses,
        'val_losses': tracker.val_losses,
        'learning_rates': tracker.lrs,
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'stopped_early': patience_counter >= model_cfg.patience
    }

    return results



class MLP(nn.Module):
    """
    Enhanced Feedforward Network for 50-100 inputs and 1 output.

    Features:
    - Funnel architecture: wide -> medium -> small hidden layers
    - BatchNorm + Dropout for regularization
    - ReLU activations
    - Residual skip connection for stability
    """

    def __init__(self, input_dim: int, model_cfg: MLPConfig):
        super().__init__()
        
        dropout_rate = model_cfg.dropout_rate
        hidden_dims = model_cfg.hidden_dims

        assert dropout_rate is not None, "Need to specify dropout rate in configuration"

        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            # if i != 0:
            #     layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))  # output layer
        self.net = nn.Sequential(*layers)

        # Optional: small residual connection from input to last hidden
        if input_dim == hidden_dims[0]:
            self.residual = nn.Linear(input_dim, hidden_dims[-1])
        else:
            self.residual = None

    def forward(self, x):
        out = self.net[:-1](x)  # all layers except final output
        if self.residual is not None:
            out = out + self.residual(x)  # residual connection
        out = self.net[-1](out)  # final linear layer
        # return out.squeeze(-1)
        return out
        


@timeit
def train_one_directional_nn_run(
    input_dim,
    model_cfg,
    prep_cfg,
    tracking_dir,
    train_loader,
    val_loader,
    fold_span,
    is_interactive_plot=False,
):
    model = MLP_direction(input_dim, model_cfg).to(prep_cfg.device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=model_cfg.lr,
        weight_decay=model_cfg.weight_decay,
        momentum=model_cfg.momentum,
    )

    if model_cfg.loss_type=="Cross-Entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise Exception("Unknown criterion for Loss function")
    
    if model_cfg.is_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=model_cfg.epochs,
            eta_min=model_cfg.lr / 50,
        )

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    tracker = TrainingTracker(
        model_cfg.to_dict(),
        tracking_dir,
        fold_span=fold_span,
    )

    epochs = range(1, model_cfg.epochs + 1)
    epochs = tqdm(epochs, desc="Training", unit="epoch") if is_interactive_plot else epochs

    def is_bad(*tensors):
        return any(not torch.isfinite(t).all().item() for t in tensors)

    grad_norm_history = deque(maxlen=60)

    for epoch in epochs:
        # ----- TRAINING PHASE -----
        model.train()
        train_loss = 0.0
        n_train = 0

        train_correct = 0
        train_total = 0

        loss_contribs = defaultdict(list)

        for xb, yb in train_loader:
            xb = xb.to(prep_cfg.device)
            yb = yb.to(prep_cfg.device).long()

            optimizer.zero_grad(set_to_none=True)

            if is_bad(xb):
                print("\n--- Current batch has NaN/inf inputs --> must be corrected ! ---\n")
                continue

            pred = model(xb)

            if is_bad(pred):
                continue

            base_loss = criterion(pred, yb)

            l1_penalty = sum(p.abs().mean() for p in model.parameters())
            l1_penalty *= model_cfg.l1_lambda

            loss = base_loss + l1_penalty

            if is_bad(loss):
                continue

            for name, value in {
                "base_loss": base_loss,
                "l1": l1_penalty,
            }.items():
                loss_contribs[name].append(value.item())

            loss.backward()

            if len(grad_norm_history) >= 15:
                max_norm = float(np.clip(np.percentile(grad_norm_history, 90), 1.0, 100.0))
            else:
                max_norm = 10.0

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm,
            )

            if is_bad(grad_norm):
                continue

            grad_norm_history.append(float(grad_norm))

            optimizer.step()

            batch_size = yb.size(0)
            train_loss += loss.item() * batch_size
            n_train += batch_size

            pred_class = pred.argmax(dim=1)
            train_correct += (pred_class == yb).sum().item()
            train_total += batch_size

        avg_train_loss = train_loss / n_train if n_train > 0 else float("nan")
        train_acc = train_correct / train_total if train_total > 0 else float("nan")

        # ----- VALIDATION PHASE -----
        model.eval()
        val_loss = 0.0
        n_val = 0

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(prep_cfg.device)
                yb = yb.to(prep_cfg.device).long()

                if is_bad(xb):
                    continue

                pred = model(xb)

                if is_bad(pred):
                    continue

                loss = criterion(pred, yb)

                batch_size = yb.size(0)
                val_loss += loss.item() * batch_size
                n_val += batch_size

                pred_class = pred.argmax(dim=1)
                val_correct += (pred_class == yb).sum().item()
                val_total += batch_size

        avg_val_loss = val_loss / n_val if n_val > 0 else float("nan")
        val_acc = val_correct / val_total if val_total > 0 else float("nan")

        current_lr = optimizer.param_groups[0]["lr"]

        tracker.add_epoch(
            epoch,
            avg_train_loss,
            loss_contribs,
            avg_val_loss,
            current_lr,
        )

        if is_interactive_plot and epoch % 5 == 0:
            tracker.plot_losses()
            tracker.plot_contribs()

        # ----- EARLY STOPPING CHECK -----
        early_stop_msg = f"Early stopping triggered at epoch {epoch}"

        if get_is_not_finished_conv(best_epoch, model_cfg):
            tqdm.write(early_stop_msg) if is_interactive_plot else print(early_stop_msg)
            break

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= model_cfg.patience:
                tqdm.write(early_stop_msg) if is_interactive_plot else print(early_stop_msg)
                break

        if model_cfg.is_lr_scheduler:
            scheduler.step()

        if is_interactive_plot:
            tqdm.write(
                f"Epoch {epoch}: "
                f"train_loss={avg_train_loss:.6f}, "
                f"val_loss={avg_val_loss:.6f}, "
                f"train_acc={train_acc:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

    if not is_interactive_plot:
        tracker.plot_losses()
        tracker.plot_contribs()

    model.load_state_dict(best_model_state)

    results = {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_losses": tracker.train_losses,
        "val_losses": tracker.val_losses,
        "learning_rates": tracker.lrs,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "stopped_early": patience_counter >= model_cfg.patience,
    }

    return results



class MLP_direction(nn.Module):
    def __init__(self, input_dim, model_cfg, output_dim=3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in model_cfg.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(model_cfg.dropout_rate),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@timeit
def train_kfold_cv_xgboost(fold_loaders, model_cfg: XGBoostConfig) -> torch.nn.Module | None:
    """
    Train model using k-fold time series cross-validation.
    
    Args:
        fold_loaders: List of (train_loader, val_loader) for each fold
        config: Training configuration
    
    Returns:
        cv_results: List of results for each fold
        best_model: Model with best validation performance
    """
    rmse_scores = []
    fold_models = []

    for fold_idx, (train_loader, val_loader, fold_span) in enumerate(fold_loaders):
        print(f"\n--- Training Fold (including val) {fold_idx + 1}/{len(fold_loaders)}: {[d.strftime('%Y-%m-%d') for d in fold_span]}")        
        
        # Train on this fold
        model, rmse = train_xgboost(
            fold_idx,
            train_loader,
            val_loader,
            model_cfg,
        )

        fold_models.append(model)
        rmse_scores.append(rmse)
    
    best_model = fold_models[np.argmin(rmse_scores)]
    
    return best_model


@timeit
def train_xgboost(fold_idx, train_loader, val_loader, cfg: XGBoostConfig):
    # Stack batches into full arrays
    X_train = np.vstack([xb.numpy() for xb, _ in train_loader])
    y_train = np.hstack([yb.numpy() for _, yb in train_loader])
    
    X_val = np.vstack([xb.numpy() for xb, _ in val_loader])
    y_val = np.hstack([yb.numpy() for _, yb in val_loader])

    # Train XGBoost
    model = xgb.XGBRegressor(**cfg.get_args())

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Validate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    print(f"Fold {fold_idx} RMSE: {rmse:.4f}")
    return model, rmse


def get_is_not_finished_conv(best_epoch, model_cfg):
    return best_epoch >= model_cfg.epochs - model_cfg.patience

