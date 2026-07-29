from sqlalchemy import create_engine, text
import json

# Connexion à la base de données SQLite avec SQLAlchemy
engine = create_engine(r'sqlite:///C:\Users\Riffard\Documents\Datasets\Fusion\Tests_carte_de_poids_et_pyramides\_test_VISWIR\VISWIR\21\results.db')
conn = engine.connect()

# Récupérer toutes les données de la table fusion_results
result = conn.execute(text("SELECT * FROM fusion_results"))
rows = result.fetchall()

# Récupérer les noms de colonnes
column_names = result.keys()

# Convertir les données en une liste de dictionnaires
data = []
for row in rows:
    row_dict = dict(zip(column_names, row))
    row_dict["metrics"] = json.loads(row_dict["metrics"])  # Convertir les métriques JSON en dictionnaire
    data.append(row_dict)

# Sauvegarde dans un fichier JSON
with open("export.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

# Fermeture de la connexion
conn.close()

print("✅ Exportation terminée ! Les données ont été sauvegardées dans export.json.")
