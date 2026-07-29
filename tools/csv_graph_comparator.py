# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Charger les fichiers CSV
# csv_file_1 = r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\export.csv"  # (Original) Premier fichier CSV
# csv_file_2 = r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\export.csv"  # (Optuna ou v2) Deuxième fichier CSV (à adapter)

# # Lire les fichiers CSV
# df1 = pd.read_csv(csv_file_1, sep=",")
# df2 = pd.read_csv(csv_file_2, sep=",")

# # Choisir la métrique à comparer (exemple : BRISQUE)
# metric = "entropy"  # Peut être changé en "entropy", "brisque", "niqe" ou "piqe" ou une autre du fichier CSV.
# metric = "entropy"  # Peut être changé en "entropy", "brisque", "niqe" ou "piqe" ou une autre du fichier CSV.

# # Vérifier si la métrique existe dans les fichiers
# if metric not in df1.columns or metric not in df2.columns:
#     raise ValueError(f"La métrique '{metric}' n'existe pas dans les fichiers CSV.")

# # Créer un dossier "results" s'il n'existe pas
# # os.makedirs("results", exist_ok=True)

# # Tracer le graphique
# plt.figure(figsize=(10, 6))
# plt.plot(df1[metric], label="Fichier 1 (original params)", marker="o")
# plt.plot(df2[metric], label="Fichier 2 (optuna param)", marker="s")
# plt.xlabel("Index")
# plt.ylabel(metric)
# # plt.title(f"Comparaison de la métrique {metric} entre deux fichiers CSV")
# plt.title(f"Comparaison de la métrique {metric} entre deux série de paramètres pour la série Global (full dataset)")
# plt.legend()
# plt.grid(True)

# # Sauvegarder le graphique
# graph_path = os.path.join("..", "results", f"comparaison_{metric}.png")
# plt.savefig(graph_path, dpi=300)
# plt.show()

# print(f"✅ Graphique enregistré sous {graph_path}")



####################################################################################################################################################################

# import pandas as pd
# import matplotlib.pyplot as plt

# # Charger les fichiers
# df_mono = pd.read_csv(r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\_Fusion_result\23-24_detection_param_f1\export.csv")
# df_multi = pd.read_csv(r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\export.csv")

# # Vérifier que les colonnes existent
# required_cols = ["f_f1_score", "v_f1_score", "s_f1_score"]
# for col in required_cols:
#     if col not in df_mono.columns or col not in df_multi.columns:
#         raise ValueError(f"Colonne manquante : {col}")

# # Lissage (optionnel)
# window_size = 3 #1 #5  # Peut être augmenté selon la fluctuation
# f_mono_smooth = df_mono["f_f1_score"].rolling(window=window_size, center=True).mean()
# f_multi_smooth = df_multi["f_f1_score"].rolling(window=window_size, center=True).mean()
# v_smooth = df_mono["v_f1_score"].rolling(window=window_size, center=True).mean()
# s_smooth = df_mono["s_f1_score"].rolling(window=window_size, center=True).mean()

# # Tracer
# x = range(len(df_mono))
# plt.figure(figsize=(10, 6))
# # plt.plot(x, f_mono_smooth, label="Fusion (mono - F1)", color="blue")
# plt.plot(x, f_mono_smooth, label="Fusion (mono - Optuna)", color="blue")
# # plt.plot(x, f_multi_smooth, label="Fusion (multi - NR-IQA)", color="purple")#, linestyle="--")
# plt.plot(x, f_multi_smooth, label="Fusion (mono - Default)", color="purple")#, linestyle="--")
# plt.plot(x, v_smooth, label="Visible", color="green", linestyle="-.")
# plt.plot(x, s_smooth, label="SWIR", color="red", linestyle=":")

# # Calcul des moyennes
# mean_f_mono = f_mono_smooth.mean()
# mean_f_multi = f_multi_smooth.mean()
# mean_v = v_smooth.mean()
# mean_s = s_smooth.mean()

# # Ajouter les annotations sur le graphique
# plt.text(0.98, 0.75, f"Mean F1 score :", color="black", fontsize=9, ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())
# # plt.text(0.98, 0.7, f"Mono: {mean_f_mono:.3f}", color="blue", fontsize=9, ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())
# plt.text(0.98, 0.7, f"Optuna: {mean_f_mono:.3f}", color="blue", fontsize=9, ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())
# # plt.text(0.98, 0.65, f"Multi: {mean_f_multi:.3f}", color="purple", fontsize=9, ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())
# plt.text(0.98, 0.65, f"Default: {mean_f_multi:.3f}", color="purple", fontsize=9, ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())
# plt.text(0.98, 0.6, f"Visible: {mean_v:.3f}", color="green", fontsize=9, ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())
# plt.text(0.98, 0.55, f"SWIR: {mean_s:.3f}", color="red", fontsize=9, ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())

# # Mise en forme
# # plt.title("Comparaison F1-score : Mono vs Multi objectif")# + Visible & SWIR")
# plt.title("Comparaison F1-score : Optuna vs default parameters")# + Visible & SWIR")
# plt.xlabel("Échantillon (index)")
# plt.ylabel("F1-score")
# plt.ylim(0, 1.05)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("f1score_comparison_optuna_vs_default_l3_annot.png", dpi=300)
# plt.close()


# # # Calcul des moyennes
# # mean_f_mono = f_mono_smooth.mean()
# # mean_f_multi = f_multi_smooth.mean()
# # mean_v = v_smooth.mean()
# # mean_s = s_smooth.mean()

# # Affichage des résultats
# # print("=== Moyenne des F1-scores ===")
# # print(f"Fusion (mono-objectif) : {mean_f_mono:.4f}")
# # print(f"Fusion (multi-objectif) : {mean_f_multi:.4f}")
# # print(f"Visible : {mean_v:.4f}")
# # print(f"SWIR : {mean_s:.4f}")

####################################################################################################################################################################

import pandas as pd
import matplotlib.pyplot as plt

# === Chemins vers les fichiers ===
f1_csv = r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\_Fusion_result\21_detection_param_f1\export.csv"
nriqa_csv = r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\_Fusion_result\21_detection_param_NR-IQA\export.csv"
default_csv = r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\_Fusion_result\21_detection_param_default\export.csv"

# === Chargement des fichiers ===
df_f1 = pd.read_csv(f1_csv)
df_nriqa = pd.read_csv(nriqa_csv)
df_default = pd.read_csv(default_csv)

# === Paramètres de base ===
window_size = 5 #3
x = range(len(df_f1))  # Même index pour tous

# === Pour enregistrer les moyennes ===
summary_data = []

# === Métriques à comparer ===
metrics = {
    "f1_score": "F1-Score",
    "niqe": "NIQE",
    "piqe": "PIQE",
    "brisque": "BRISQUE",
    "entropy": "Entropie"
}

# === Palette de couleurs ===
colors = {
    "visible": "green",
    "swir": "red",
    "f1_opt": "blue",
    "nriqa_opt": "orange",
    "default": "purple"
}

# === Boucle sur les métriques ===
for metric_key, metric_label in metrics.items():
    plt.figure(figsize=(10, 6))

    # Récupération et lissage des courbes
    v = df_f1[f"v_{metric_key}"].rolling(window=window_size, center=True).mean()
    s = df_f1[f"s_{metric_key}"].rolling(window=window_size, center=True).mean()
    f1 = df_f1[f"f_{metric_key}"].rolling(window=window_size, center=True).mean()
    nriqa = df_nriqa[f"f_{metric_key}"].rolling(window=window_size, center=True).mean()
    default = df_default[f"f_{metric_key}"].rolling(window=window_size, center=True).mean()

    # Tracés
    plt.plot(x, f1, label="Fusion (Optuna F1)", color=colors["f1_opt"])
    plt.plot(x, nriqa, label="Fusion (Optuna NR-IQA)", color=colors["nriqa_opt"])
    plt.plot(x, default, label="Fusion (Default)", color=colors["default"])
    plt.plot(x, v, label="Visible", color=colors["visible"], linestyle="-.")
    plt.plot(x, s, label="SWIR", color=colors["swir"], linestyle=":")

    # Moyennes
    mean_f1 = f1.mean()
    mean_nriqa = nriqa.mean()
    mean_def = default.mean()
    mean_v = v.mean()
    mean_s = s.mean()

    # Annotations des moyennes
    # plt.text(0.98, 0.85, f"Mean {metric_label}:", fontsize=9, ha='right', va='bottom', transform=plt.gca().get_yaxis_transform())
    # plt.text(0.98, 0.80, f"F1 Opt.: {mean_f1:.3f}", color=colors["f1_opt"], fontsize=9, ha='right', transform=plt.gca().get_yaxis_transform())
    # plt.text(0.98, 0.75, f"NR-IQA Opt.: {mean_nriqa:.3f}", color=colors["nriqa_opt"], fontsize=9, ha='right', transform=plt.gca().get_yaxis_transform())
    # plt.text(0.98, 0.70, f"Default: {mean_def:.3f}", color=colors["default"], fontsize=9, ha='right', transform=plt.gca().get_yaxis_transform())
    # plt.text(0.98, 0.65, f"Visible: {mean_v:.3f}", color=colors["visible"], fontsize=9, ha='right', transform=plt.gca().get_yaxis_transform())
    # plt.text(0.98, 0.60, f"SWIR: {mean_s:.3f}", color=colors["swir"], fontsize=9, ha='right', transform=plt.gca().get_yaxis_transform())
    summary_data.append({
        "Métrique": metric_label,
        "Fusion (Optuna F1)": mean_f1,
        "Fusion (Optuna NR-IQA)": mean_nriqa,
        "Fusion (Default)": mean_def,
        "Visible": mean_v,
        "SWIR": mean_s
    })


    # Mise en forme
    plt.title(f"Comparaison des scores : {metric_label}")
    plt.xlabel("Échantillon (index)")
    plt.ylabel(metric_label)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"comparison_{metric_key}_l5.png", dpi=300)
    plt.close()

# Convertir en DataFrame
df_summary = pd.DataFrame(summary_data)

# Réorganiser les colonnes si besoin
df_summary = df_summary[["Métrique", "Fusion (Optuna F1)", "Fusion (Optuna NR-IQA)", "Fusion (Default)", "Visible", "SWIR"]]

# Sauvegarder dans un CSV
df_summary.to_csv("comparaison_moyennes_scores.csv", index=False)

# Optionnel : affichage dans la console
print("=== Résumé des moyennes ===")
print(df_summary)
