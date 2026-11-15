# evaluation.py

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils import resample


# Helper function to resolve results directory paths.
def _get_results_dir(task):
    """
    Ensures the task-specific results directory exists at the project root.
    """

    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    results_dir = os.path.join(project_root, "results", task)
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


# Save cross-validation tuning summary.
def save_cv_summary(cv_summary, task="classification"):
    """
    Save the cross-validation hyperparameter tuning summary to the results folder.
    """

    results_dir = _get_results_dir(task)
    rows = []

    for model_name, info in cv_summary.items():
        rows.append({
            "Model": model_name,
            "Best CV Score": info["best_cv_score"],
            "Best Parameters": str(info["best_params"])
        })

    cv_df = pd.DataFrame(rows)
    out_path = os.path.join(results_dir, "cv_tuning_summary.xlsx")
    cv_df.to_excel(out_path, index=False)
    print(f"[{task.upper()}] Saved CV summary to: {out_path}")
    return cv_df


# Seed stability test through changing random initializations.
def evaluate_seed_stability(
    best_models,
    X_train,
    y_train,
    X_test,
    y_test,
    task="classification",
    seeds=None,
    resample_train=False
):
    """
    Run the best-found model configurations across multiple random seeds and generate
    boxplots to assess performance stability under variable initialization randomness.

    Parameters:
    - best_models: Dict of fitted best Pipeline objects from train.py.
    - X_train, y_train: Processed training data.
    - X_test, y_test: Processed test data.
    - task: 'classification' (scores ROC-AUC) or 'regression' (scores RMSE).
    - seeds: List of integer seeds (defaults to 10 seeds).
    - resample_train: If False, varies only the model's internal 'random_state' on 'fixed X_train'.
                      If True, also draws a stratified/bootstrap sample of 'X_train' per seed.
    """

    if seeds is None:
        seeds = [7, 13, 23, 42, 66, 88, 101, 123, 256, 512]

    results_dir = _get_results_dir(task)
    y_train_1d = y_train.squeeze()
    y_test_1d = y_test.squeeze()

    metric_name = "ROC-AUC" if task == "classification" else "RMSE"
    records = []

    print(f"\n{task.upper()} SEED STABILITY EVALUATION ({len(seeds)} Seeds | Metric: {metric_name})")

    for name, pipeline in best_models.items():
        valid_params = pipeline.get_params()
        has_random_state = "model__random_state" in valid_params

        for seed in seeds:
            model_clone = clone(pipeline)

            # Set model initialization seed if supported by the estimator.
            if has_random_state:
                model_clone.set_params(model__random_state=seed)

            # Optionally resample training data per seed.
            if resample_train:
                strat = y_train_1d if task == "classification" else None
                X_tr, y_tr = resample(
                    X_train, y_train_1d, replace=True, random_state=seed, stratify=strat
                )
            else:
                X_tr, y_tr = X_train, y_train_1d

            model_clone.fit(X_tr, y_tr)

            if task == "classification":
                y_prob = model_clone.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test_1d, y_prob)
            else:
                y_pred = model_clone.predict(X_test)
                score = np.sqrt(mean_squared_error(y_test_1d, y_pred))

            records.append({
                "Model": name,
                "Seed": seed,
                metric_name: score
            })

    stability_df = pd.DataFrame(records)

    # Print summary statistics.
    summary_stats = stability_df.groupby("Model", as_index=False)[metric_name].agg(
        mean="mean", std="std", min="min", max="max"
    )
    print(summary_stats.to_string(index=False))

    # Save seed stability tables to results folder.
    raw_excel_path = os.path.join(results_dir, "seed_stability_raw.xlsx")
    summary_excel_path = os.path.join(results_dir, "seed_stability_summary.xlsx")
    stability_df.to_excel(raw_excel_path, index=False)
    summary_stats.to_excel(summary_excel_path, index=False)

    # Plot boxplot.
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=stability_df,
        x="Model",
        y=metric_name,
        hue="Model",
        palette="Set2",
        width=0.5,
        legend=False,
        boxprops=dict(alpha=0.8)
    )
    sns.stripplot(
        data=stability_df,
        x="Model",
        y=metric_name,
        color="black",
        size=6,
        jitter=0.15,
        alpha=0.7
    )
    plt.title(
        f"{task.capitalize()} Model Stability Across {len(seeds)} Random Seeds ({metric_name})",
        fontsize=13,
        fontweight="bold"
    )
    plt.xlabel("Model", fontsize=11)
    plt.ylabel(f"Test {metric_name}", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save boxplot figure to results folder.
    fig_path = os.path.join(results_dir, "seed_stability_boxplot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[{task.upper()}] Saved seed stability results and boxplot to: {results_dir}")
    return stability_df, summary_stats


# Classification evaluation.
def evaluate_classification(trained_models, X_test, y_test):
    """
    Evaluate trained classification models on the test set and return a summary 
    DataFrame.
    """

    results_dir = _get_results_dir("classification")
    y_true = y_test.squeeze().astype(int)
    results = []

    print("\nCLASSIFICATION TEST SET EVALUATION (Target: pCR (outcome))")

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        results.append({
            "Model": name,
            "ROC-AUC": roc_auc_score(y_true, y_prob),
            "PR-AUC": average_precision_score(y_true, y_prob),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    print(results_df.to_string(index=False))

    # Save classification test metrics to results folder.
    excel_path = os.path.join(results_dir, "test_metrics.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"[CLASSIFICATION] Saved test metrics to: {excel_path}")

    return results_df


def plot_classification_2x2(trained_models, X_test, y_test):
    """
    Render a 2x2 classification evaluation grid:
    - (1,1) Multi-Model Confusion Matrix Heatmap (all models across TN, FP, FN, TP).
    - (1,2) ROC Curves (all models).
    - (2,1) Precision-Recall Curves (all models).
    - (2,2) Polar / Radar Plot of 6 classification metrics.
    """
    results_dir = _get_results_dir("classification")
    y_true = y_test.squeeze().astype(int)
    metrics_df = evaluate_classification(trained_models, X_test, y_test)

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("Classification Performance Evaluation (Target: pCR)", fontsize=15, fontweight="bold")

    # Top-Left: Multi-Model Confusion Matrix Heatmap.
    ax1 = fig.add_subplot(2, 2, 1)
    cm_cols = ["TN", "FP", "FN", "TP"]
    cm_matrix = metrics_df[cm_cols].to_numpy(dtype=int)
    cm_pct = cm_matrix / len(y_true) * 100

    annot_labels = np.array([
        [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
        for row_c, row_p in zip(cm_matrix, cm_pct)
    ])

    sns.heatmap(
        cm_matrix,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "TN\n(True 0, Pred 0)",
            "FP\n(True 0, Pred 1)",
            "FN\n(True 1, Pred 0)",
            "TP\n(True 1, Pred 1)"
        ],
        yticklabels=metrics_df["Model"],
        ax=ax1,
        annot_kws={"size": 10, "weight": "bold"},
        linewidths=1,
        linecolor="white"
    )
    ax1.set_title("Confusion Matrix Breakdown (All Models)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Confusion Matrix Quadrant")
    ax1.set_ylabel("Model")
    ax1.tick_params(axis="y", rotation=0)

    # Top-Right: ROC Curves.
    ax2 = fig.add_subplot(2, 2, 2)
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax2.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random (AUC = 0.500)")
    ax2.set_title("ROC Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("False Positive Rate (1 - Specificity)")
    ax2.set_ylabel("True Positive Rate (Sensitivity)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)

    # Bottom-Left: Precision-Recall Curves.
    ax3 = fig.add_subplot(2, 2, 3)
    baseline_rate = y_true.mean()
    for name, model in trained_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        ax3.plot(recall, precision, lw=2, label=f"{name} (AP = {ap_val:.3f})")

    ax3.axhline(
        baseline_rate,
        color="gray",
        linestyle="--",
        lw=1.5,
        label=f"Baseline Prevalence ({baseline_rate:.2f})"
    )
    ax3.set_title("Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Recall (Sensitivity)")
    ax3.set_ylabel("Precision (Positive Predictive Value)")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(alpha=0.3)

    # Bottom-Right: Polar / Radar Plot.
    ax4 = fig.add_subplot(2, 2, 4, polar=True)
    radar_metrics = ["ROC-AUC", "PR-AUC", "F1-Score", "Balanced Accuracy", "Precision", "Recall"]
    num_vars = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]   # close circular polygon.

    for _, row in metrics_df.iterrows():
        values = [row[m] for m in radar_metrics]
        values += values[:1]
        ax4.plot(angles, values, lw=2, label=row["Model"])
        ax4.fill(angles, values, alpha=0.08)

    ax4.set_theta_offset(np.pi / 2)
    ax4.set_theta_direction(-1)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_metrics, fontsize=9, fontweight="bold")
    ax4.set_ylim(0, 1.0)
    ax4.set_title("Multi-Metric Radar Comparison", fontsize=12, fontweight="bold", pad=20)
    ax4.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save 2x2 classification grid figure to results folder.
    fig_path = os.path.join(results_dir, "evaluation_2x2_grid.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[CLASSIFICATION] Saved 2x2 evaluation grid to: {fig_path}")
    return metrics_df

# Regression evaluation.
def evaluate_regression(trained_models, X_test, y_test):
    """
    Evaluate trained regression models on the test set and return a summary DataFrame.
    """

    results_dir = _get_results_dir("regression")
    y_true = y_test.squeeze().astype(float)
    results = []

    print("\nREGRESSION TEST SET EVALUATION (Target: RelapseFreeSurvival (outcome))")

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_true, y_pred)

        results.append({
            "Model": name,
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MedAE": median_absolute_error(y_true, y_pred),
            "MSE": mse,
            "R2": r2_score(y_true, y_pred)
        })

    results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    print(results_df.to_string(index=False))

    # Save regression test metrics to results folder.
    excel_path = os.path.join(results_dir, "test_metrics.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"[REGRESSION] Saved test metrics to: {excel_path}")

    return results_df


def plot_regression_2x2(trained_models, X_test, y_test):
    """
    Render a 2x2 regression evaluation grid:
    - (1,1) Nested 2x2 Grid of Actual vs. Predicted RFS Scatter Plots (per model).
    - (1,2) Nested 2x2 Grid of Residuals vs. Predicted Values Scatter Plots (per model).
    - (2,1) Residual Error Distribution (Histogram / KDE).
    - (2,2) Regression Error Characteristic (REC) Curve.
    """

    results_dir = _get_results_dir("regression")
    y_true = y_test.squeeze().astype(float)
    metrics_df = evaluate_regression(trained_models, X_test, y_test)

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        "Regression Performance Evaluation (Target: RelapseFreeSurvival)",
        fontsize=15,
        fontweight="bold"
    )
    outer_gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.28)
    colors = plt.cm.tab10.colors

    # Precompute predictions, residuals, and global axis limits for consistent comparison.
    preds_dict = {name: model.predict(X_test) for name, model in trained_models.items()}
    res_dict = {name: y_true - preds for name, preds in preds_dict.items()}

    min_actual, max_actual = y_true.min(), y_true.max()
    all_preds = np.concatenate(list(preds_dict.values()))
    all_res = np.concatenate(list(res_dict.values()))
    pred_min, pred_max = all_preds.min() - 3, all_preds.max() + 3
    res_min, res_max = all_res.min() - 5, all_res.max() + 5

    # Top-Left: Nested 2x2 Grid of Actual vs. Predicted Scatter Plots.
    ax_tl_title = fig.add_subplot(outer_gs[0, 0])
    ax_tl_title.set_title("Actual vs. Predicted RFS", fontsize=12, fontweight="bold", pad=22)
    ax_tl_title.axis("off")

    avp_gs = outer_gs[0, 0].subgridspec(2, 2, wspace=0.25, hspace=0.38)
    for idx, (name, y_pred) in enumerate(preds_dict.items()):
        r, c = divmod(idx, 2)
        ax_sub = fig.add_subplot(avp_gs[r, c])
        ax_sub.scatter(y_true, y_pred, color=colors[idx], alpha=0.7, s=28, edgecolors="none")
        ax_sub.plot([min_actual, max_actual], [min_actual, max_actual], "k--", lw=1.5)
        ax_sub.set_title(name, fontsize=10, fontweight="bold")
        ax_sub.set_xlim(min_actual - 5, max_actual + 5)
        ax_sub.set_ylim(min_actual - 5, max_actual + 5)
        ax_sub.grid(alpha=0.3)

        if r == 1:
            ax_sub.set_xlabel("Actual RFS", fontsize=9)
        if c == 0:
            ax_sub.set_ylabel("Predicted RFS", fontsize=9)
        ax_sub.tick_params(labelsize=8)

    # Top-Right: Nested 2x2 Grid of Residuals vs. Predicted Scatter Plots.
    ax_tr_title = fig.add_subplot(outer_gs[0, 1])
    ax_tr_title.set_title("Residuals vs. Predicted Values", fontsize=12, fontweight="bold", pad=22)
    ax_tr_title.axis("off")

    res_gs = outer_gs[0, 1].subgridspec(2, 2, wspace=0.25, hspace=0.38)
    for idx, (name, y_pred) in enumerate(preds_dict.items()):
        r, c = divmod(idx, 2)
        ax_sub = fig.add_subplot(res_gs[r, c])
        residuals = res_dict[name]
        ax_sub.scatter(y_pred, residuals, color=colors[idx], alpha=0.7, s=28, edgecolors="none")
        ax_sub.axhline(0, color="black", linestyle="--", lw=1.5)
        ax_sub.set_title(name, fontsize=10, fontweight="bold")
        ax_sub.set_xlim(pred_min, pred_max)
        ax_sub.set_ylim(res_min, res_max)
        ax_sub.grid(alpha=0.3)

        if r == 1:
            ax_sub.set_xlabel("Predicted RFS", fontsize=9)
        if c == 0:
            ax_sub.set_ylabel("Residual", fontsize=9)
        ax_sub.tick_params(labelsize=8)

    # Bottom-Left: Residual Error Distribution (KDE).
    ax3 = fig.add_subplot(outer_gs[1, 0])
    for idx, (name, residuals) in enumerate(res_dict.items()):
        sns.kdeplot(
            residuals,
            ax=ax3,
            color=colors[idx],
            lw=2,
            fill=True,
            alpha=0.12,
            label=name
        )

    ax3.axvline(0, color="black", linestyle="--", lw=1.5)
    ax3.set_title("Residual Error Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Residual Error (Actual - Predicted)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Bottom-Right: Regression Error Characteristic (REC) Curve.
    ax4 = fig.add_subplot(outer_gs[1, 1])
    max_tol = np.max([np.abs(res).max() for res in res_dict.values()])
    tolerances = np.linspace(0, max_tol, 200)

    for idx, (name, residuals) in enumerate(res_dict.items()):
        abs_errors = np.abs(residuals)
        accuracy_within_tol = [np.mean(abs_errors <= tol) * 100 for tol in tolerances]
        mae_val = mean_absolute_error(y_true, preds_dict[name])
        ax4.plot(
            tolerances,
            accuracy_within_tol,
            color=colors[idx],
            lw=2,
            label=f"{name} (MAE = {mae_val:.2f})"
        )

    ax4.set_title("Regression Error Characteristic (REC) Curve", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Absolute Error Tolerance")
    ax4.set_ylabel("Test Samples Within Tolerance (%)")
    ax4.set_ylim(0, 105)
    ax4.legend(loc="lower right", fontsize=9)
    ax4.grid(alpha=0.3)

    # Save 2x2 regression grid figure to results folder.
    fig_path = os.path.join(results_dir, "evaluation_2x2_grid.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[REGRESSION] Saved 2x2 evaluation grid to: {fig_path}")
    return metrics_df