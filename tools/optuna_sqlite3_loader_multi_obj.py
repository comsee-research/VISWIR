import optuna
import optuna.visualization as vis
import os
from plotly.io import show

from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from scipy.interpolate import griddata

from mpl_toolkits.mplot3d import Axes3D  # pour la vue 3D

# 🔹 Chemin vers la base de données SQLite
# storage_url = "sqlite:///../results//optuna_study.sqlite3"
storage_url = "sqlite:///C:/Users/Riffard/Documents/__WSL/Resultats_HPC/RASMD/RASMD_sunny/optuna_study.sqlite3"
output_dir = r"C:\Users\Riffard\Documents\__WSL\Resultats_HPC\RASMD\RASMD_sunny\optuna_plots"
os.makedirs(output_dir, exist_ok=True)

# 🔹 Chargement de l'étude depuis le fichier SQLite
study = optuna.load_study(study_name="viswir_quality_optimization_study", storage=storage_url)

# 🔹 Liste des hyperparamètres et des métriques
params = ["facteur_swir", "beta", "level", "gamma_value"]
metrics = ["Entropy", "BRISQUE", "NIQE", "PIQE"]

# 🔹 Génération des graphiques pour chaque paramètre et chaque métrique
for param in params:
    for i, metric in enumerate(metrics):
        fig = vis.plot_slice(study, params=[param], target=lambda t: t.values[i], target_name=metric)
        fig.show()
        # fig.write_image(os.path.join(output_dir, f"{param}_vs_{metric}.png"))

# 🔹 Génération du front de Pareto (3 objectifs max)
pareto_fig = vis.plot_pareto_front(
    study,
    targets=lambda t: (t.values[0], t.values[1], t.values[2]),
    target_names=["Entropy", "BRISQUE", "NIQE"]
)
show(pareto_fig)
# pareto_fig.write_image(os.path.join(output_dir, "pareto_front.png"))

# print(f"✅ Graphiques enregistrés dans '{output_dir}'")


df = study.trials_dataframe(attrs=("params", "values"))

# for (p1, p2) in combinations(params, 2):
#     for idx, metric in enumerate(metrics):
#         df_sub = df[[f"params_{p1}", f"params_{p2}", f"values_{idx}"]].dropna()

#         df_sub = df_sub.rename(columns={
#             f"params_{p1}": p1,
#             f"params_{p2}": p2,
#             f"values_{idx}": metric
#         })

#         # Conversion et tri
#         df_sub[p1] = pd.to_numeric(df_sub[p1], errors='coerce')
#         df_sub[p2] = pd.to_numeric(df_sub[p2], errors='coerce')
#         df_sub = df_sub.dropna()

#         # Pivot + tri
#         pivot = df_sub.pivot_table(index=p2, columns=p1, values=metric, aggfunc="mean")
#         pivot = pivot.sort_index().sort_index(axis=1)

#         # Affichage
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f", mask=pivot.isnull())
#         plt.title(f"{metric} en fonction de {p1} et {p2}")
#         plt.xlabel(p1)
#         plt.ylabel(p2)
#         plt.tight_layout()

#         filename = f"heatmap_{metric}_{p1}_vs_{p2}.png"
#         filepath = os.path.join(output_dir, filename)
#         plt.savefig(filepath)
#         plt.close()

for (p1, p2) in combinations(params, 2):
    for idx, metric in enumerate(metrics):
        df_sub = df[[f"params_{p1}", f"params_{p2}", f"values_{idx}"]].dropna()
        df_sub = df_sub.rename(columns={
            f"params_{p1}": p1,
            f"params_{p2}": p2,
            f"values_{idx}": metric
        })

        # Données
        x = df_sub[p1].values
        y = df_sub[p2].values
        z = df_sub[metric].values

        # Création de la grille régulière
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolation des points irréguliers
        # zi = griddata((x, y), z, (xi, yi), method='cubic')
        zi = griddata((x, y), z, (xi, yi), method="linear") # cubic

        # Bornes par métrique
        if metric == "Entropy":
            zi = np.clip(zi, 0, 8)  # pour Entropy
        else:
            zi = np.clip(zi, 0, 100)  # pour BRISQUE/NIQE/PIQE

        # Affichage
        plt.figure(figsize=(6, 4))

        # Définir les bornes pour chaque métrique (si tu veux une échelle fixe)
        metric_bounds = {
            "Entropy": (0, 8),
            "BRISQUE": (0, 100),
            "NIQE": (0, 100),
            "PIQE": (0, 100)
        }

        # Choix de la colormap selon la métrique
        if metric == "Entropy":
            cmap = "viridis"  # Entropy : plus c'est élevé, plus c'est riche
        else:
            cmap = "viridis_r"  # Pour BRISQUE, NIQE, PIQE : inversé

        # Et dans le plot :
        vmin, vmax = metric_bounds[metric]
        # contourf = plt.contourf(xi, yi, zi, levels=20, cmap=cmap, vmin=vmin, vmax=vmax) # Affichage avec palette inversée si nécessaire
        contourf = plt.contourf(xi, yi, zi, levels=20, cmap=cmap) # Affichage avec palette inversée si nécessaire
        # contourf = plt.contourf(xi, yi, zi, levels=20, cmap="viridis")
        contours = plt.contour(xi, yi, zi, levels=10, colors='black', linewidths=0.8)  # lignes de niveau
        plt.clabel(contours, inline=True, fontsize=8)  # étiquettes sur les courbes
        plt.colorbar(contourf, label=metric)
        plt.title(f"{metric} — {p1} vs {p2}")
        plt.xlabel(p1)
        plt.ylabel(p2)
        plt.tight_layout()

        filename = f"contour_{metric}_{p1}_vs_{p2}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()

######################################## Répartition des combinaisons

df_params = df[[f"params_{p}" for p in params]].copy() #.dropna()
df_params.columns = params

# sns.pairplot(df_params, plot_kws={"s": 30})
sns.pairplot(df_params, kind="scatter", plot_kws={"s": 30}, corner=True)
plt.suptitle("Répartition des combinaisons testées", y=1.02)
plt.tight_layout()

filename = "pairplot_param_distributions.png"
filepath = os.path.join(output_dir, filename)
plt.savefig(filepath)
plt.close()


# Charger les données depuis l'étude
# study = optuna.load_study(study_name="viswir_quality_optimization_study", storage="sqlite:///../results/optuna_study.sqlite3")
# trials_df = study.trials_dataframe()

# # Scatter 3D : facteur_swir, beta, gamma_value + couleur = level
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# scatter = ax.scatter(
#     trials_df['params_facteur_swir'],
#     trials_df['params_beta'],
#     trials_df['params_gamma_value'],
#     c=trials_df['params_level'],
#     cmap='viridis',
#     s=40
# )
# ax.set_xlabel("facteur_swir")
# ax.set_ylabel("beta")
# ax.set_zlabel("gamma_value")
# fig.colorbar(scatter, label="level")
# plt.title("Distribution des paramètres")
# # plt.show()
# filename = "param_distributions.png"
# filepath = os.path.join(output_dir, filename)
# plt.savefig(filepath)
# plt.close()