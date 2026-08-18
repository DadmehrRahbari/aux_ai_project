# aux_ai_project/plot_utils.py
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from evaluation import compute_metrics
from params import DATASET_NAME
import pandas as pd
import seaborn as sns

# ----------------------
# Academic Font Settings
# ----------------------
plt.rcParams["font.family"]       = "serif"
plt.rcParams["font.serif"]        = ["Times New Roman"]
plt.rcParams["font.size"]         = 18
plt.rcParams["axes.titlesize"]    = 20
plt.rcParams["axes.labelsize"]    = 18
plt.rcParams["xtick.labelsize"]   = 16
plt.rcParams["ytick.labelsize"]   = 16
plt.rcParams["legend.fontsize"]   = 16
plt.rcParams["figure.titlesize"]  = 22


def export_to_file(name):
    folder = os.path.join("plots", DATASET_NAME)
    os.makedirs(folder, exist_ok=True)

    plt.tight_layout()

    # Only add a figure-level legend for single-axis figures
    # (multi-panel figures handle their own legends per subplot)
    try:
        fig = plt.gcf()
        if len(fig.axes) == 1:
            ax = plt.gca()
            handles, labels = ax.get_legend_handles_labels()
            # FIX: this used to unconditionally call plt.legend(loc="best")
            # for any single-axis figure, silently overwriting any legend a
            # plot function had already explicitly positioned (e.g.
            # resource_radar's bbox_to_anchor placement outside the polar
            # axes) -- "best" places it back inside the axes bounds, right
            # on top of the data, which is exactly the overlap that
            # positioning was meant to avoid. Only auto-add a "best" legend
            # if the axes doesn't already have one.
            if handles and ax.get_legend() is None:
                plt.legend(loc="best")
    except Exception:
        pass

    png_path = os.path.join(folder, f"Fig_{DATASET_NAME}_plot_{name}.png")
    pdf_path = os.path.join(folder, f"Fig_{DATASET_NAME}_plot_{name}.pdf")

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()   # prevent figure accumulation in memory
    print(f"Exported: {png_path} and {pdf_path}")


# ------------------------------------------------------------------ #
#  Individual plot functions
# ------------------------------------------------------------------ #

def plot_anomaly_scores(scores, y_true, anomaly_types=None):
    frames = list(range(len(scores)))
    plt.figure(figsize=(14, 5))
    plt.plot(frames, scores, label="Proposed Anomaly Score", color="#1f77b4", linewidth=1.5)

    if anomaly_types:
        type_colors = {
            "novel":          "red",
            "asd":            "orange",
            "sensor_failure": "purple",
            "normal":         "green",
        }
        label_map = {
            "novel":          "Novelty",
            "asd":            "Adv. Semantic Drift (ASD)",
            "sensor_failure": "Sensor Fail",
            "normal":         "Normal",
        }
        for t, c in type_colors.items():
            idxs = [i for i, ty in enumerate(anomaly_types) if ty == t]
            valid_idxs = [i for i in idxs if i < len(scores)]
            if valid_idxs:
                plt.scatter(
                    valid_idxs,
                    [scores[i] for i in valid_idxs],
                    color=c,
                    label=label_map.get(t, t),
                    s=10,
                    alpha=0.6,
                )

    plt.xlabel("Simulation Frame Index")
    plt.ylabel("Anomaly Score")
    plt.title(f"Temporal Anomaly Scores: {DATASET_NAME}")
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.7)
    export_to_file("anomaly_scores")


def plot_method_comparison(all_scores, y_true):
    methods   = list(all_scores.keys())
    metrics   = ["AUROC", "PR_AUC", "F1", "Accuracy", "TPR", "FPR"]
    results   = {m: compute_metrics(y_true, all_scores[m]) for m in methods}
    color_map = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#e377c2"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    fig, axes = plt.subplots(2, 3, figsize=(26, 14))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [results[m][metric] for m in methods]
        # Draw bars individually so Big-AI gets a distinct hatch pattern.
        # Reduced alpha + dense hatch makes it clearly distinguishable
        # even in black-and-white print.
        for i, (m, v) in enumerate(zip(methods, values)):
            is_bigai = "big" in m.lower() or "big-ai" in m.lower()
            axes[idx].bar(
                i, v,
                color=color_map[i],
                alpha=0.35 if is_bigai else 0.85,
                hatch="////" if is_bigai else "",
                edgecolor="black",
                linewidth=1.5 if is_bigai else 0.8,
            )
        axes[idx].set_xticks(range(len(methods)))
        axes[idx].set_xticklabels(
            [f"{m} $\\dagger$" if "big" in m.lower() else m for m in methods],
            fontsize=13,
        )
        axes[idx].set_title(f"({chr(97 + idx)}) Metric: {metric}", fontweight="bold")
        axes[idx].set_ylim(0, 1.15)
        axes[idx].grid(axis="y", linestyle="--", alpha=0.6)

        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold", fontsize=15)

    # Add footnote explaining Big-AI hatching
    fig.text(
        0.01, 0.01,
        r"$\dagger$ Big-AI: trivial always-alarm lower bound, not a competing method.",
        fontsize=13,
        fontstyle="italic",
        color="gray",
    )

    plt.suptitle(f"Quantitative Performance Comparison - {DATASET_NAME}", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    export_to_file("method_comparison")


def plot_roc_curves(all_scores, y_true):
    plt.figure(figsize=(11, 9))
    for method, scores in all_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr, label=method, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    export_to_file("roc_curves")


def plot_pr_curves(all_scores, y_true):
    plt.figure(figsize=(11, 9))
    for method, scores in all_scores.items():
        p, r, _ = precision_recall_curve(y_true, scores)
        plt.plot(r, p, label=method, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curves")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    export_to_file("pr_curves")


def plot_score_distributions(all_scores, y_true):
    num_methods = len(all_scores)
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    plt.figure(figsize=(7 * num_methods, 8))
    
    for i, (method, scores) in enumerate(all_scores.items()):
        ax = plt.subplot(1, num_methods, i + 1)
        s = np.array(scores)
        yt = np.array(y_true)
        plt.hist(s[yt == 0], bins=30, alpha=0.5, label="Normal",  color="green")
        plt.hist(s[yt == 1], bins=30, alpha=0.5, label="Anomaly", color="red")
        plt.title(method)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        
        # Add subpanel label
        ax.text(-0.05, 1.05, panel_labels[i], transform=ax.transAxes,
                fontsize=20, fontweight="bold", va="bottom", ha="right")
        
    plt.tight_layout()
    export_to_file("score_distributions")


def plot_f1_vs_threshold(all_scores, y_true):
    thresholds = np.linspace(0.0, 1.0, 100)
    plt.figure(figsize=(14, 8))
    for method, scores in all_scores.items():
        f1s = [compute_metrics(y_true, scores, threshold=t)["F1"] for t in thresholds]
        plt.plot(thresholds, f1s, label=method, linewidth=2)
    plt.xlabel("Detection Threshold")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Sensitivity to Threshold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    export_to_file("f1_vs_threshold")


def plot_communication_overhead(comm_histories_kb):
    plt.figure(figsize=(14, 8))
    for method, history in comm_histories_kb.items():
        label_name = (
            "Proposed (Aux-AI Monitor)"
            if method == "proposed"
            else method.replace("_", " ").title()
        )
        is_proposed = "proposed" in method.lower() and "non_fl" not in method.lower()
        plt.plot(history, label=label_name, linewidth=3.0 if is_proposed else 1.5)

    plt.title("Resource Consumption: Cumulative Communication Overhead")
    plt.xlabel("Simulation Frame Index")
    plt.ylabel("Total Data Transferred (KB)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.5)
    export_to_file("communication_overhead")


def plot_resource_radar(results_for_latex):
    filtered = [r for r in results_for_latex if "cloud" not in r[0].lower()]
    labels   = ["Latency", "Memory", "Energy", "FPR95", "Detection Delay"]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for res in filtered:
        name = res[0]
        display_values = [
            np.log1p(res[6]) / 2,
            np.log1p(res[8]) / 4,
            np.log1p(res[9] * 1000) / 10,
            res[4],
            np.clip(res[5] / 20, 0, 1),
        ]
        display_values += display_values[:1]
        ax.plot(angles, display_values, linewidth=2, label=name.replace("_", " ").title())
        ax.fill(angles, display_values, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(f"Edge Resource Profile: {DATASET_NAME}", y=1.08)
    # FIX: legend was overlapping the radar's data area at the top-right --
    # exactly where the Latency/Detection Delay axes commonly extend to.
    # Anchored further outside the axes and reserved layout space for it so
    # it renders clear of the chart entirely instead of floating over data.
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    plt.tight_layout()
    export_to_file("resource_radar")


def plot_trust_stability_comparison(final_scores, y_true):
    plt.figure(figsize=(14, 8))
    methods    = list(final_scores.keys())
    normal_end = int(len(y_true) * 0.3)

    variances = []
    for m in methods:
        scores = np.nan_to_num(final_scores[m][:normal_end])
        variances.append(np.var(scores))

    bar_colors  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(methods)]
    bar_hatches = ["////" if "big" in m.lower() else "" for m in methods]
    bar_alphas  = [0.35   if "big" in m.lower() else 0.85 for m in methods]
    xlabels     = [m.replace("_", " ").title() for m in methods]
    # FIX: a genuinely ~0 variance (e.g. Big-AI, a trivial always-alarm stub
    # with no learned scoring and therefore near-constant output) rendered as
    # a fully invisible bar -- indistinguishable from missing data to a
    # reader. This is a real, correct value (not a bug), but needs to be
    # visually legible as "deliberately near-zero" rather than "absent".
    # Floor at 1% of the max observed variance so it's visible without
    # visually overstating an essentially-zero value.
    max_var = max(variances) if variances else 0
    floor = max_var * 0.01 if max_var > 0 else 0
    display_variances = [v if v > floor else floor for v in variances]
    for i, (v, c, h, a) in enumerate(zip(display_variances, bar_colors, bar_hatches, bar_alphas)):
        plt.bar(i, v, color=c, alpha=a, hatch=h,
                edgecolor="black", linewidth=1.5 if h else 0.8)
    plt.xticks(range(len(methods)), xlabels)
    plt.ylabel(r"Trust Score Variance ($\sigma^2$)")
    plt.title("Noise Resilience: Trust Stability (Normal Phase)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    export_to_file("trust_stability")

def plot_hardware_performance_group(results_list):
    edge_results = [r for r in results_list if "cloud" not in r[0].lower()]
    methods  = [r[0] for r in edge_results]
    latency  = [r[6] for r in edge_results]
    fps      = [r[7] for r in edge_results]
    memory   = [r[8] for r in edge_results]
    energy   = [r[9] for r in edge_results]
    colors   = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    hatches  = ["////" if "big" in m.lower() else "" for m in methods]
    alphas   = [0.35   if "big" in m.lower() else 0.85 for m in methods]

    def _bars(ax, vals):
        for i, (m, v, c, h, a) in enumerate(zip(methods, vals, colors, hatches, alphas)):
            ax.bar(i, v, color=c, alpha=a, hatch=h,
                   edgecolor="black", linewidth=1.5 if h else 0.8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            [f"{m} $\\dagger$" if "big" in m.lower() else m for m in methods],
            fontsize=12,
        )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))

    # Ax 1: Latency
    _bars(ax1, latency)
    ax1.set_yscale("log")
    ax1.set_title("(a) Inference Latency [Log Scale] (Lower is Better)", fontweight="bold")
    ax1.set_ylabel("ms / frame (log)")
    ax1.grid(axis="y", linestyle="--", alpha=0.6, which="both")

    # Ax 2: Throughput
    _bars(ax2, fps)
    ax2.set_title("(b) System Throughput (Higher is Better)", fontweight="bold")
    ax2.set_ylabel("Frames Per Second (FPS)")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    # Ax 3: RAM Usage
    _bars(ax3, memory)
    ax3.set_title("(c) RAM Usage (Lower is Better)", fontweight="bold")
    ax3.set_ylabel("MB")
    ax3.set_yscale("log")
    ax3.grid(axis="y", linestyle="--", alpha=0.6)

    # Ax 4: Energy Drain
    _bars(ax4, energy)
    ax4.set_title("(d) Total Energy Drain (Lower is Better)", fontweight="bold")
    ax4.set_ylabel("Joules (J)")
    ax4.set_yscale("log")
    ax4.grid(axis="y", linestyle="--", alpha=0.6)

    plt.suptitle(f"Edge Resource Profiling: {DATASET_NAME}", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    export_to_file("hardware_profile")


def plot_module_configuration_matrix(dataset_name):
    data = {
        "Feature Extractor": [1, 1, 1, 1],
        "Attention Head":    [0, 1, 1, 1],
        "Temporal Buffer":   [0, 0, 1, 1],
        "FL Sync":           [0, 0, 0, 1],
    }
    methods = ["Original (Big-AI)", "Attention-Only", "Proposed (Non-FL)", "Proposed (Full)"]
    df      = pd.DataFrame(data, index=methods)

    plt.figure(figsize=(14, 8))
    sns.heatmap(df, annot=True, cbar=False, cmap="YlGnBu", linewidths=1.5, linecolor="white")
    plt.title(f"Experimental Configuration Matrix ({dataset_name})", pad=20)
    plt.xlabel("System Modules")
    plt.ylabel("Ablation Configurations")
    export_to_file("module_matrix")


def plot_sfc_reliability(logs, dataset_name):
    """APSS Spatial Consistency & Fault Detection (Turco et al. 2025)."""
    frames     = [l["frame"]                    for l in logs]
    sfc_scores = [l.get("sfc_fault_detected", 0) for l in logs]
    is_anomaly = [l.get("is_anomaly", 0)         for l in logs]

    plt.figure(figsize=(14, 5))
    plt.plot(frames, sfc_scores, label="SFC (Spatial) Score", color="#9467bd", linewidth=1.5)
    plt.fill_between(frames, 0, is_anomaly, color="red", alpha=0.15, label="True Anomaly Region")

    try:
        first_anom = next(i for i, val in enumerate(is_anomaly) if val > 0)
        first_det  = next(i for i, val in enumerate(sfc_scores) if val > 0 and i >= first_anom)
        latency    = first_det - first_anom
        plt.annotate(
            f"Latency: {latency} frames",
            xy=(first_det, 0.2),
            xytext=(first_det + 15, 0.5),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1),
        )
    except (StopIteration, ValueError):
        pass

    plt.title(f"APSS Reliability: Spatial Consistency & Fault Detection ({dataset_name})")
    plt.xlabel("Frame Index")
    plt.ylabel("Fault Detection Score")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", frameon=True)
    export_to_file("apss_reliability")


def plot_trust_calibration(y_true, y_prob, dataset_name):
    """
    Trust Calibration — Brier Score & ECE.

    Uses quantile binning (strategy='quantile') so every bin contains an equal
    number of samples.  Uniform bins produce a V-shape artifact when scores are
    concentrated in a narrow range, because a handful of outlier frames can
    dominate a bin with only 2-3 samples.  Quantile bins eliminate this.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    y_prob = np.array(y_prob, dtype=float)
    y_true = np.array(y_true, dtype=int)

    # Clip to [0,1] — scores should already be in range, but guard against
    # the entropy-dominated monitor output leaking through unnormalised.
    if y_prob.max() > 1.0 or y_prob.min() < 0.0:
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-6)

    # Quantile binning: each bin has equal number of samples, avoiding
    # the sparse-bin V-shape caused by scores that don't cover [0,1] uniformly.
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=8, strategy="quantile"
    )
    brier = brier_score_loss(y_true, y_prob)

    # ECE: use the same quantile bins for consistency
    ece = float(np.mean(np.abs(prob_true - prob_pred)))

    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal Calibration")
    plt.plot(prob_pred, prob_true, marker="s", color="#1f77b4", linewidth=2,
             label=f"Aux-AI (Brier: {brier:.4f}, ECE: {ece:.4f})")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"Trust Calibration Analysis (ECE) - {dataset_name}")
    plt.xlabel("Predicted Anomaly Probability")
    plt.ylabel("Actual Anomaly Fraction")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left")
    export_to_file("trust_calibration")


def plot_segmentation_quality(logs, dataset_name):
    """mIoU & Pixel Accuracy temporal consistency."""
    frames = [l["frame"]               for l in logs]
    miou   = [l.get("miou", 0)         for l in logs]

    plt.figure(figsize=(14, 5))
    plt.plot(frames, miou, color="#2ca02c",
             label=f"TDC Consistency (Avg: {np.mean(miou):.3f})", alpha=0.8)

    plt.title(f"TDC Decision Consistency ({dataset_name})")
    plt.xlabel("Frame Index")
    plt.ylabel("Consistency Score")
    plt.ylim([0, 1.1])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower left")
    export_to_file("miou_quality")


def plot_fl_convergence(history_proposed, history_local_only):
    """FL benefit: global federated AUROC vs. local-only monitor."""
    if not history_proposed:
        print("Warning: fl_history is empty, skipping fl_convergence plot.")
        return

    plt.figure(figsize=(11, 9))
    rounds = np.arange(len(history_proposed))
    # FIX: colors were swapped relative to the ablation plot, which shows the
    # exact same fl_history/local_history data -- Federated was blue here but
    # red there, Local-Only was red here but blue there. Same underlying data,
    # inconsistent legend colors between the two figures is confusing for a
    # reviewer. Standardized to match: Federated=red solid, Local-Only=blue
    # dashed, in both places.
    plt.plot(rounds, history_proposed,  label="Proposed (Federated)",
             linewidth=2.5, color="#d62728", marker="o", markersize=5)
    plt.plot(rounds, history_local_only, label="Local-Only (No FL)",
             linestyle="--", color="#1f77b4", marker="s", markersize=5)
    plt.xlabel("FL Round")
    # FIX: ylim(0.45, 1.0) was clipping real, genuinely-computed early-round
    # AUROC values (rounds with very few positive examples are noisy and can
    # legitimately fall below 0.45) off the bottom of the chart, making them
    # look like missing data rather than real early volatility. 0.0-1.0 shows
    # every real point; genuinely undefined rounds (only one class seen so
    # far) are NaN at the source and correctly render as a gap regardless.
    plt.ylim(0.0, 1.0)
    plt.ylabel("Global Detection AUROC")
    plt.title("Federated Learning Gain: Collaborative Convergence")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    export_to_file("fl_convergence")
