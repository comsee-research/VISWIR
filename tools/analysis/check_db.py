from sqlalchemy import create_engine, text
from pathlib import Path

# Adapte le chemin si besoin
# db_path = Path("results/RASMD_global_val/results.db") # ou ton dossier de sortie
db_path = Path(r"C:\Users\Riffard\Documents\__WSL\VISWIR\results\RASMD_global_train\results.db") # ou ton dossier de sortie
engine = create_engine(f"sqlite:///{db_path}")

with engine.connect() as conn:
    # 1. Compter le total
    total = conn.execute(text("SELECT COUNT(*) FROM fusion_results")).scalar()
    
    # 2. Compter les succès (là où error est NULL)
    success = conn.execute(text("SELECT COUNT(*) FROM fusion_results WHERE error IS NULL")).scalar()
    
    # 3. Compter les échecs
    errors = conn.execute(text("SELECT COUNT(*) FROM fusion_results WHERE error IS NOT NULL")).scalar()

    print(f"📊 BILAN DE SANTÉ DE LA BASE DE DONNÉES")
    print(f"=======================================")
    print(f"📂 Base : {db_path}")
    print(f"📝 Total lignes : {total}")
    print(f"✅ Succès       : {success} ({(success/total)*100:.1f}%)")
    print(f"❌ Erreurs      : {errors} ({(errors/total)*100:.1f}%)")

    if errors > 0:
        print("\n🔎 Exemple d'erreurs :")
        # Affiche les 5 premières erreurs pour voir ce qui cloche
        err_samples = conn.execute(text("SELECT visible_img, error FROM fusion_results WHERE error IS NOT NULL LIMIT 5"))
        for img, err in err_samples:
            print(f"   - {Path(img).name} : {err}")