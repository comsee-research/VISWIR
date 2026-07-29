"""
Optuna optimization result visualization and plotting.
"""

# src/optimization/visualization.py

# import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

from optuna import Study
import optuna.visualization as vis
from plotly.io import show
import plotly.graph_objects as go

from common.logger import logger


def visualize_study(study: Study, best_trial, output_dir: Path, run_detection: bool = False):
    """
    Generate and save visualizations of an Optuna study.

    Depending on the optimization mode, this function produces:

    - **NR-IQA mode**:
      
      * Pareto front (Entropy, BRISQUE, PIQE)
      * Slice plots for each parameter vs. metric
      * Contour plots (interpolated surfaces) for parameter pairs

    - **Detection mode**:
      
      * Optimization history (F1-score)
      * Contour plots for parameter pairs vs. F1-score

    Parameters
    ----------
    study : optuna.study.Study
        The Optuna study object containing all trials.
    best_trial : optuna.trial.FrozenTrial
        The best trial selected from the study.
    output_dir : Path
        Directory where contour plots will be saved as PNG files.
    run_detection : bool, default=False
        If True, generate visualizations for detection (F1-score).
        If False, generate visualizations for NR-IQA metrics.

    Notes
    -----
    - Uses both Plotly (interactive) and Matplotlib (static) visualizations.
    - Contour plots are saved to disk in ``output_dir``.
    - Interactive plots may fail to display in some environments; warnings are logged.
    - Metrics visualized in NR-IQA mode: Entropy, BRISQUE, NIQE, PIQE.
    - In detection mode, only F1-score is visualized.
    """
    best_params = best_trial.params
    best_values = best_trial.values

    params = ["facteur_swir", "beta", "level", "gamma_value"]
    metrics = ["Entropy", "BRISQUE", "NIQE", "PIQE"]

    try:
        if not run_detection:
            # ➤ Pareto front
            pareto_fig = vis.plot_pareto_front(
                study,
                targets=lambda t: (t.values[0], t.values[1], t.values[3]),
                target_names=["Entropy", "BRISQUE", "PIQE"]
            )
            try:
                show(pareto_fig)
            except Exception as e:
                logger.warning(f"🔕 Affichage Pareto désactivé : {e}")

            trace_best = go.Scatter3d(
                x=[best_values[0]],
                y=[best_values[1]],
                z=[best_values[3]],
                mode='markers+text',
                marker=dict(size=6, color='red'),
                text=["Best (Tchebycheff)"],
                name="Tchebycheff"
            )
            pareto_fig.add_trace(trace_best)
            pareto_fig.show()

            # ➤ Courbes slice
            for param in params:
                for i, metric in enumerate(metrics):
                    fig = vis.plot_slice(
                        study,
                        params=[param],
                        target=lambda t: t.values[i],
                        target_name=metric
                    )
                    fig.show()

            # ➤ Courbes interpolées (contourf)
            df = study.trials_dataframe(attrs=("params", "values"))
            for (p1, p2) in combinations(params, 2):
                for idx, metric in enumerate(metrics):
                    df_sub = df[[f"params_{p1}", f"params_{p2}", f"values_{idx}"]].dropna()
                    df_sub = df_sub.rename(columns={
                        f"params_{p1}": p1,
                        f"params_{p2}": p2,
                        f"values_{idx}": metric
                    })

                    x, y, z = df_sub[p1].values, df_sub[p2].values, df_sub[metric].values
                    xi = np.linspace(x.min(), x.max(), 100)
                    yi = np.linspace(y.min(), y.max(), 100)
                    xi, yi = np.meshgrid(xi, yi)
                    from scipy.interpolate import griddata
                    zi = griddata((x, y), z, (xi, yi), method="linear")

                    if metric == "Entropy":
                        zi = np.clip(zi, 0, 8)
                        cmap = "viridis"
                    else:
                        zi = np.clip(zi, 0, 100)
                        cmap = "viridis_r"

                    plt.figure(figsize=(6, 4))
                    contourf = plt.contourf(xi, yi, zi, levels=20, cmap=cmap)
                    contours = plt.contour(xi, yi, zi, levels=10, colors='black', linewidths=0.8)
                    plt.clabel(contours, inline=True, fontsize=8)
                    plt.colorbar(contourf, label=metric)
                    plt.title(f"{metric} — {p1} vs {p2}")
                    plt.xlabel(p1)
                    plt.ylabel(p2)
                    plt.tight_layout()

                    filepath = output_dir / f"contour_{metric}_{p1}_vs_{p2}.png"
                    plt.savefig(filepath)
                    plt.close()

        else:
            # ➤ Historique optimisation F1
            fig = vis.plot_optimization_history(study)
            fig.show()

            df = study.trials_dataframe(attrs=("params", "values"))
            for (p1, p2) in combinations(params, 2):
                df_sub = df[[f"params_{p1}", f"params_{p2}", "values_0"]].dropna()
                df_sub = df_sub.rename(columns={
                    f"params_{p1}": p1,
                    f"params_{p2}": p2,
                    "values_0": "f1_score"
                })

                x, y, z = df_sub[p1].values, df_sub[p2].values, df_sub["f1_score"].values
                xi = np.linspace(x.min(), x.max(), 100)
                yi = np.linspace(y.min(), y.max(), 100)
                xi, yi = np.meshgrid(xi, yi)
                from scipy.interpolate import griddata
                zi = griddata((x, y), z, (xi, yi), method="linear")
                zi = np.clip(zi, 0, 1)

                plt.figure(figsize=(6, 4))
                contourf = plt.contourf(xi, yi, zi, levels=20, cmap="magma", vmin=0, vmax=1)
                contours = plt.contour(xi, yi, zi, levels=10, colors='black', linewidths=0.8)
                plt.clabel(contours, inline=True, fontsize=8)
                plt.colorbar(contourf, label="f1_score")
                plt.title(f"f1_score — {p1} vs {p2}")
                plt.xlabel(p1)
                plt.ylabel(p2)
                plt.tight_layout()

                filepath = output_dir / f"contour_f1score_{p1}_vs_{p2}.png"
                plt.savefig(filepath)
                plt.close()

    except Exception as e:
        logger.warning(f"🔕 Visualisation désactivée ou échouée : {e}")
