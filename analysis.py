from .config import PrepConfig
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import shap
from utils.general_utils import timeit, create_new_filename


@timeit
def shap_analysis(
        model, 
        X_bg: torch.Tensor,
        X_test: torch.Tensor,
        features, 
        track_dir, 
        config: PrepConfig, 
        n_datapoints: int = 100,
        # n_samples: int = 100,
        # n_top_features: int = 30,
        percentile: int = 99,
        is_shorten_name: bool = False,
    ):
    
    # --- Define model wrapper ---        
    if config.model_type == "mlp":

        model.eval()

        idx = torch.randperm(len(X_bg))[:n_datapoints]
        X_bg = X_bg[idx].to(config.device)

        X_test = X_test[-n_datapoints:].to(config.device)

        explainer = shap.DeepExplainer(model, X_bg)

        shap_values = explainer.shap_values(X_test)

        X_test_np = X_test.detach().cpu().numpy()

    elif config.model_type == "xgboost":

        X_test = X_test[-n_datapoints:]

        X_test_np = X_test.detach().cpu().numpy().astype(np.float64)

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_test_np)

    
    else:
        raise Exception(f"model type {config.model_type} not configured for SHAP")
    
    if isinstance(shap_values, list):
        raise ValueError(
            "shap_analysis only supports single-output models"
        )

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    # # Compute SHAP values
    # shap_values = explainer.shap_values(
    #     X_test_np,
    #     # nsamples=n_samples,
    #     # l1_reg=f"num_features({n_top_features})",
    # )

    # --------------
    # Ploting 
    # --------------
    def shorten_name(name, max_len=45):
        name = str(name)
        return name if len(name) <= max_len else name[:max_len - 3] + "..."
    short_features = [shorten_name(f) for f in features]

    #  --- Summary
    # Clip extreme values to prevent x-axis stretching
    clip_val = np.percentile(np.abs(shap_values), percentile)
    shap_value_clipped = np.clip(shap_values, -clip_val, clip_val)

    shap.summary_plot(
        shap_value_clipped,
        X_test_np,
        feature_names=short_features if is_shorten_name else features,
        show=False,
        plot_type="dot",
        plot_size=(14, 8),
    )

    fig = plt.gcf()
    fig.suptitle(f"SHAP Feature (Clipped) to {percentile}th percentile")
    fig.subplots_adjust(left=0.35)

    filename = create_new_filename(track_dir, "shap_summary_plot", "png")
    fig.savefig(track_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Plot last value contributions
    base_value = np.array(explainer.expected_value).item()
    dist_to_last = 1 # 1 is last datapoint
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[-dist_to_last],
            base_values=base_value,
            data=X_test_np[-dist_to_last],
            feature_names=short_features if is_shorten_name else features,
        ),
        max_display=20,
    )

    fig = plt.gcf()
    fig.suptitle(f"SHAP of {'last' if dist_to_last == 1 else f'{dist_to_last}-to-last'} datapoint")
    fig.subplots_adjust(left=0.35)

    filename = create_new_filename(track_dir, "shap_last_plot", "png")
    fig.savefig(track_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # # global importance (mean absolute SHAP values)
    # importance = np.abs(shap_values).mean(axis=0)

    # plt.figure()
    # plt.clf()
    # plt.barh(features, importance)
    # plt.xlabel("mean(|SHAP value|)")
    # plt.title("Feature importance")
    # plt.tight_layout()
    # plt.savefig(track_dir / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    # plt.close()

    # expl = shap.Explanation(
    #     values=shap_values,
    #     data=X_test_np,
    #     feature_names=features
    # )

    # shap.plots.beeswarm(expl)   # modern replacement for summary_plot