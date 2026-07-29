import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
df = pd.read_csv(r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\export.csv")

# Vérifier que les colonnes de F1-score existent
# required_columns = ["f_f1_score", "v_f1_score", "s_f1_score"]
required_columns = ["f_iou_mean", "v_iou_mean", "s_iou_mean"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Colonne manquante dans le fichier CSV : {col}")
    
# Appliquer un lissage par moyenne glissante (exemple : fenêtre de 5)
window_size = 1 #1 #5 #30
f_smooth = df["f_f1_score"].rolling(window=window_size, center=True).mean()
v_smooth = df["v_f1_score"].rolling(window=window_size, center=True).mean()
s_smooth = df["s_f1_score"].rolling(window=window_size, center=True).mean()
# f_smooth = df["f_iou_mean"].rolling(window=window_size, center=True).mean()
# v_smooth = df["v_iou_mean"].rolling(window=window_size, center=True).mean()
# s_smooth = df["s_iou_mean"].rolling(window=window_size, center=True).mean()


# Créer un index x (par exemple, les lignes ou un identifiant si disponible)
x = range(len(df))  # ou bien df["nom_colonne_identifiant"] si tu as une colonne spécifique

# # Tracer les courbes F1-score
# plt.figure(figsize=(10, 6))
# plt.plot(x, df["f_f1_score"], label="Fusion", marker='o')
# plt.plot(x, df["v_f1_score"], label="Visible", marker='s')
# plt.plot(x, df["s_f1_score"], label="SWIR", marker='^')

# # Mise en forme
# plt.title("Comparaison des F1-scores")
# plt.xlabel("Échantillon (ou index)")
# plt.ylabel("F1-score")
# plt.ylim(0, 1.05)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()

# Tracer
plt.figure(figsize=(10, 6))
plt.plot(x, f_smooth, label="Fusion (lissé)", marker='', color="blue")
plt.plot(x, v_smooth, label="Visible (lissé)", marker='', color="green")
plt.plot(x, s_smooth, label="SWIR (lissé)", marker='', color="red")

# Optionnel : aussi afficher les courbes originales en pointillés
# plt.plot(x, df["f_f1_score"], linestyle='dotted', alpha=0.3, color="blue")
# plt.plot(x, df["v_f1_score"], linestyle='dotted', alpha=0.3, color="green")
# plt.plot(x, df["s_f1_score"], linestyle='dotted', alpha=0.3, color="red")

# Mise en forme
# plt.title("F1-score avec lissage (moyenne glissante)")
plt.title("F1-score")
# plt.title("IoU Moyen avec lissage (moyenne glissante)")
# plt.title("IoU Moyen")
plt.xlabel("Échantillon (index)")
plt.ylabel("F1-score")
# plt.ylabel("Mean IoU")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.show()

# Afficher le graphique
# plt.show()
plt.savefig("f1score_comparison_l1.png", dpi=300)
# plt.savefig("iou_mean_comparison_l1.png", dpi=300)
plt.close()
