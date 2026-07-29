from sqlalchemy import create_engine, text
import pandas as pd
import json

# Connexion à la base de données SQLite avec SQLAlchemy
# engine = create_engine(r'sqlite:///C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\results.db')
engine = create_engine(r'sqlite:///C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\results\_Fusion_result\21_detection_param_default\results.db')
conn = engine.connect()

# Récupérer toutes les données de la table fusion_results
result = conn.execute(text("SELECT * FROM fusion_results"))
rows = result.fetchall()

# Récupérer les noms de colonnes
column_names = result.keys()

### Old version
# # Convertir les données en une liste de dictionnaires
# data = []
# for row in rows:
#     row_dict = dict(zip(column_names, row))
#     row_dict["metrics"] = json.loads(row_dict["metrics"])  # Convertir les métriques JSON en dictionnaire
#     data.append(row_dict)

# # Convertir en DataFrame pandas
# df = pd.DataFrame(data)

# # Éclater la colonne `metrics` (JSON) en colonnes individuelles
# metrics_df = df["metrics"].apply(pd.Series)

# # Fusionner avec les autres colonnes
# df = df.drop(columns=["metrics"]).join(metrics_df)

# # Sauvegarde dans un fichier CSV
# df.to_csv("export.csv", index=False, encoding="utf-8")

# # Fermeture de la connexion
# conn.close()

# print("✅ Exportation terminée ! Les données ont été sauvegardées dans export.csv.")


### MàJ :
# Convertir les données en une liste de dictionnaires
data = []
for row in rows:
    row_dict = dict(zip(column_names, row))
    
    # Conversion des colonnes JSON en dictionnaires Python
    if "metrics_f" in row_dict and row_dict["metrics_f"]:
        row_dict["metrics_f"] = json.loads(row_dict["metrics_f"])
    else:
        row_dict["metrics_f"] = {}
    
    if "metrics_v" in row_dict and row_dict["metrics_v"]:
        row_dict["metrics_v"] = json.loads(row_dict["metrics_v"])
    else:
        row_dict["metrics_v"] = {}
    
    if "metrics_s" in row_dict and row_dict["metrics_s"]:
        row_dict["metrics_s"] = json.loads(row_dict["metrics_s"])
    else:
        row_dict["metrics_s"] = {}
    
    data.append(row_dict)

# Convertir en DataFrame pandas
df = pd.DataFrame(data)

# Éclater chaque dictionnaire de métriques en colonnes
metrics_f_df = df["metrics_f"].apply(pd.Series).add_prefix("f_")
metrics_v_df = df["metrics_v"].apply(pd.Series).add_prefix("v_")
metrics_s_df = df["metrics_s"].apply(pd.Series).add_prefix("s_")

# Supprimer les anciennes colonnes JSON
df = df.drop(columns=["metrics_f", "metrics_v", "metrics_s"])

# Fusionner avec les colonnes éclatées
df = pd.concat([df, metrics_f_df, metrics_v_df, metrics_s_df], axis=1)

# Sauvegarde dans un fichier CSV
df.to_csv("export.csv", index=False, encoding="utf-8")

# Fermeture de la connexion
conn.close()

print("✅ Exportation terminée ! Les données ont été sauvegardées dans export.csv.")
