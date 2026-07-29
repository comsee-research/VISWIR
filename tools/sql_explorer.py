from sqlalchemy import create_engine, text
from rich.console import Console
from rich.table import Table
import json

# Création de l'interface console Rich
console = Console()

# Connexion à la base de données
engine = create_engine(r'sqlite:///C:\Users\Riffard\Documents\Datasets\Fusion\Tests_carte_de_poids_et_pyramides\_test_VISWIR\VISWIR\21\results.db')
conn = engine.connect()

# Vérification des tables disponibles
result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
console.print("[bold cyan]Tables disponibles :[/bold cyan]")
for row in result:
    console.print(f"🔹 {row[0]}")

# # Récupération des données de fusion_results
# result = conn.execute(text("SELECT * FROM fusion_results"))

# # Création du tableau Rich
# table = Table(title="Résultats de Fusion", show_lines=True)

# Récupération **des 10 premières lignes** de fusion_results
result = conn.execute(text("SELECT * FROM fusion_results LIMIT 10"))

# Création du tableau Rich
table = Table(title="Aperçu des Résultats de Fusion (10 premières lignes)", show_lines=True)

# Ajout des colonnes
table.add_column("ID", style="bold yellow")
table.add_column("Visible Img", style="dim")
table.add_column("SWIR Img", style="dim")
table.add_column("Ref Img", style="dim")
table.add_column("Alpha", style="bold")
table.add_column("Beta", style="bold")
table.add_column("Gamma", style="bold")
table.add_column("Metrics Fusion", style="italic magenta")
table.add_column("Metrics Visible", style="italic magenta")
table.add_column("Metrics SWIR", style="italic magenta")

# Ajout des lignes avec formatage des métriques
for row in result:
    id, visible_img, swir_img, ref_img, alpha, beta, gamma, metrics_json, error = row
    metrics_dict = json.loads(metrics_json)  # Convertir JSON en dict
    metrics_str = "\n".join([f"{k}: {v:.3f}" for k, v in metrics_dict.items()])  # Format des métriques

    table.add_row(str(id), visible_img, swir_img, ref_img, f"{alpha:.2f}", f"{beta:.2f}", f"{gamma:.2f}", metrics_str)

# Affichage du tableau
console.print(table)

# Fermeture de la connexion
conn.close()

#==============================================================================================#
# from sqlalchemy import create_engine, text
# from rich.console import Console
# from rich.table import Table
# import json

# # Création de l'interface console Rich
# console = Console()

# # Connexion à la base SQLite
# engine = create_engine(r'sqlite:///C:\Users\Riffard\Documents\Datasets\Fusion\Tests_carte_de_poids_et_pyramides\_test_VISWIR\VISWIR\21\results.db')
# conn = engine.connect()

# # Vérification des tables disponibles
# result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
# console.print("[bold cyan]Tables disponibles :[/bold cyan]")
# for row in result:
#     console.print(f"🔹 {row[0]}")

# # Récupération des données de fusion_results
# result = conn.execute(text("SELECT * FROM fusion_results"))
# rows = result.fetchall()

# # Extraction des métriques depuis le premier enregistrement
# example_metrics = json.loads(rows[0][7])  # Les métriques sont à l'index 7 dans la table
# metric_names = list(example_metrics.keys())  # Liste des noms des métriques

# # Création du tableau Rich
# table = Table(title="Résultats de Fusion", show_lines=True)

# # Ajout des colonnes fixes
# table.add_column("ID", style="bold yellow")
# table.add_column("Visible Img", style="dim")
# table.add_column("SWIR Img", style="dim")
# table.add_column("Ref Img", style="dim")
# table.add_column("Alpha", style="bold")
# table.add_column("Beta", style="bold")
# table.add_column("Gamma", style="bold")

# # Ajout dynamique des colonnes de métriques
# for metric in metric_names:
#     table.add_column(metric, style="italic magenta")

# # Ajout des lignes avec les métriques réparties en colonnes
# for row in rows:
#     id, visible_img, swir_img, ref_img, alpha, beta, gamma, metrics_json, error = row
#     metrics_dict = json.loads(metrics_json)  # Convertir JSON en dict

#     # Création de la ligne avec les valeurs extraites
#     row_values = [
#         str(id), visible_img, swir_img, ref_img,
#         f"{alpha:.2f}", f"{beta:.2f}", f"{gamma:.2f}"
#     ]

#     # Ajout des valeurs de chaque métrique dans sa colonne
#     for metric in metric_names:
#         row_values.append(f"{metrics_dict.get(metric, 'N/A'):.3f}")

#     table.add_row(*row_values)

# # Affichage du tableau
# console.print(table)

# # Fermeture de la connexion
# conn.close()
