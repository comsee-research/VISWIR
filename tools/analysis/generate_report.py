import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration pour des graphiques style "Article Scientifique"
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def parse_metrics(json_str):
    """Convertit une chaîne JSON en dictionnaire, gère les NULL."""
    try:
        if not json_str:
            return {}
        return json.loads(json_str)
    except (TypeError, json.JSONDecodeError):
        return {}

def analyze_database(db_path: Path, output_folder: Path):
    print(f"🔍 Analyse de : {db_path}")
    
    # 1. Connexion et Chargement
    conn = sqlite3.connect(db_path)
    
    # On ne prend que les lignes sans erreur
    query = "SELECT * FROM fusion_results WHERE error IS NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("⚠️ La base de données est vide ou ne contient que des erreurs.")
        return

    print(f"📊 Données chargées : {len(df)} lignes.")

    # 2. Parsing des colonnes JSON (Métriques)
    # On suppose que l'on a metrics_f (Fusion), metrics_v (Visible), metrics_s (SWIR)
    # On les éclate en colonnes distinctes préfixées (ex: Fusion_PSNR, Swir_SSIM)
    
    metric_cols = {
        'metrics_f': 'Fusion',
        'metrics_v': 'Vis',
        'metrics_s': 'Swir'
    }

    for col, prefix in metric_cols.items():
        if col in df.columns:
            # On transforme la colonne de texte JSON en DataFrame
            expanded = df[col].apply(parse_metrics).apply(pd.Series)
            # On renomme les colonnes (ex: psnr -> Fusion_psnr)
            expanded = expanded.add_prefix(f"{prefix}_")
            # On colle ça au DataFrame principal
            df = pd.concat([df, expanded], axis=1)

    # On supprime les colonnes JSON d'origine et les chemins inutiles pour l'analyse
    cols_to_drop = list(metric_cols.keys()) + ['visible_img', 'swir_img', 'ref_img', 'grd_tr', 'error']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 3. Génération du Rapport Statistique (Global)
    # On sélectionne uniquement les colonnes numériques
    numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])
    
    # describe() donne : count, mean, std, min, 25%, 50%, 75%, max
    stats = numeric_df.describe().T 
    
    # Ajout de l'intervalle de confiance ou d'autres métriques custom si besoin
    # stats['IQR'] = stats['75%'] - stats['25%'] # Ecart inter-quartile

    print("\n📈 --- RÉSUMÉ GLOBAL (Extrait) ---")
    print(stats[['mean', 'std', 'min', 'max']].head(10))

    # 4. Export Excel
    output_folder.mkdir(parents=True, exist_ok=True)
    excel_path = output_folder / f"Rapport_Stats_{db_path.parent.name}.xlsx"
    
    with pd.ExcelWriter(excel_path) as writer:
        stats.to_excel(writer, sheet_name='Global_Stats')
        df_clean.to_excel(writer, sheet_name='Raw_Data', index=False)
        print(f"\n💾 Rapport Excel sauvegardé : {excel_path}")

    # 5. Visualisation
    # Exemple : Boxplot des métriques de Fusion principales
    # On cherche les colonnes qui contiennent "Fusion" et ("psnr" ou "ssim" ou "f1"...)
    fusion_metrics = [c for c in df_clean.columns if "Fusion" in c]
    
    if fusion_metrics:
        # On normalise les données pour le plot si les échelles sont trop différentes,
        # ou on fait plusieurs plots. Ici, on fait un plot par métrique.
        
        for metric in fusion_metrics:
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=df_clean[metric], color="skyblue")
            plt.title(f"Distribution de {metric}")
            plt.ylabel("Valeur")
            
            # Sauvegarde du plot
            plot_name = f"Boxplot_{metric}.png"
            plt.savefig(output_folder / plot_name)
            plt.close()
            print(f"🖼️ Graphique généré : {plot_name}")

    # 6. Analyse par Paramètres (Si on a fait varier alpha, beta...)
    # Si 'alpha' est constant, std sera 0, donc on ignore.
    # Si 'alpha' varie, on groupe par alpha pour voir son impact.
    group_cols = [c for c in ['alpha', 'beta', 'level', 'gamma'] if c in df_clean.columns and df_clean[c].nunique() > 1]
    
    if group_cols:
        print(f"\n🔍 Analyse groupée par : {group_cols}")
        grouped = df_clean.groupby(group_cols)[fusion_metrics].mean()
        print(grouped)
        
        # Export de l'analyse groupée
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace') as writer:
            grouped.to_excel(writer, sheet_name='Grouped_By_Params')

if __name__ == "__main__":
    # Exemple d'utilisation
    # Remplacer par le chemin de la base
    db_file = Path("../results/RASMD_global_val/results.db") 
    output = Path("../results/RASMD_global_val/Rapports")
    
    analyze_database(db_file, output)