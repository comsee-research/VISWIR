import os
import numpy as np
import pandas as pd
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.interpolate import griddata
import optuna.visualization as vis
from plotly.io import show

# 🔹 Configuration
study_name = "viswir_f1_optimization_study"
storage_url = "sqlite:///C:/Users/Riffard/Documents/__WSL/Resultat_HPC/optuna_study.sqlite3"
output_dir = "optuna_f1score_plots"
os.makedirs(output_dir, exist_ok=True)

# 🔹 Chargement de l'étude
study = optuna.load_study(study_name=study_name, storage=storage_url)

# 🔹 Historique d'optimisation du F1-score
fig = vis.plot_optimization_history(study, target=lambda t: t.values[0], target_name="f1_score")
# fig.write_image(os.path.join(output_dir, "f1score_optimization_history.png"))
fig.show()

# 🔹 Données de l'étude
df = study.trials_dataframe(attrs=("params", "values"))

# 🔹 Paramètres considérés
params = ["facteur_swir", "beta", "level", "gamma_value"]

# 🔹 Graphiques de type contour (F1-score en fonction de p1 vs p2)
for (p1, p2) in combinations(params, 2):
    df_sub = df[[f"params_{p1}", f"params_{p2}", "values_0"]].dropna()
    df_sub = df_sub.rename(columns={
        f"params_{p1}": p1,
        f"params_{p2}": p2,
        "values_0": "f1_score"
    })

    x = df_sub[p1].astype(float).values
    y = df_sub[p2].astype(float).values
    z = df_sub["f1_score"].astype(float).values

    # Grille régulière pour l'interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi, yi), method="linear")
    zi = np.clip(zi, 0, 1)  # pour le F1-score

    # ➤ Contour plot
    plt.figure(figsize=(6, 4))
    contourf = plt.contourf(xi, yi, zi, levels=20, cmap="magma", vmin=0, vmax=1)
    contours = plt.contour(xi, yi, zi, levels=10, colors='black', linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.colorbar(contourf, label="f1_score")
    plt.title(f"f1_score — {p1} vs {p2}")
    plt.xlabel(p1)
    plt.ylabel(p2)
    plt.tight_layout()

    filename = f"contour_f1score_{p1}_vs_{p2}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# 🔹 Pairplot pour visualiser les combinaisons de paramètres testées
df_params = df[[f"params_{p}" for p in params]].copy()
df_params.columns = params
sns.pairplot(df_params.astype(float), kind="scatter", plot_kws={"s": 30}, corner=True)
plt.suptitle("Répartition des combinaisons testées", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pairplot_param_distributions.png"))
plt.close()

print(f"✅ Graphiques enregistrés dans '{output_dir}'")
