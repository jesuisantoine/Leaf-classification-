import os

# --- CORRECTION DU BUG WINDOWS (Message rouge) ---
# On fixe manuellement le nombre de cœurs pour éviter que joblib ne cherche 'wmic'
# Cela supprime le Warning [WinError 2]
os.environ['LOKY_MAX_CPU_COUNT'] = '4' 

from src.data_loader import LeafDataLoader
from src.models import get_model_configs
from src.model_trainer import ModelTrainer
from src.visualization import plot_confusion_matrix, plot_cv_results
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# Plus d'import de tqdm ici

def main():
    # ---------------------------------------------------------
    # 1. Chargement des données
    # ---------------------------------------------------------
    loader = LeafDataLoader(raw_data_dir='data/raw')
    X, y, X_submission, submission_ids, classes = loader.load_data()

    print(f"Données totales: {X.shape[0]} exemples, {X.shape[1]} features")

    # ---------------------------------------------------------
    # 2. Création du "Hold-out Set" (Test set local)
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train set: {X_train.shape[0]} | Test set (Secret): {X_test.shape[0]}")

    # ---------------------------------------------------------
    # 3. Entraînement (GridSearch + Cross-Validation)
    # ---------------------------------------------------------
    configs = get_model_configs()
    trainer = ModelTrainer(X_train, y_train)
    
    results = []
    best_overall_model = None
    best_overall_score = 0
    best_model_name = ""

    print("\n--- Démarrage de l'optimisation des modèles (Patience...) ---")
    
    # Boucle standard (Sans barre de progression)
    for name, config in configs.items():
        print(f" > Entraînement de {name} en cours...")
        
        # Assure-toi que verbose=0 dans src/model_trainer.py pour ne pas avoir de spam
        result = trainer.train_and_evaluate(name, config)
        
        model = result['best_model']
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        
        result['test_score'] = test_acc
        results.append(result)

        if test_acc > best_overall_score:
            best_overall_score = test_acc
            best_overall_model = model
            best_model_name = name

    # ---------------------------------------------------------
    # 4. Analyse et Résultats
    # ---------------------------------------------------------
    print("\n--- TABLEAU RÉCAPITULATIF FINAL ---")
    df_results = pd.DataFrame(results)
    summary = df_results[['model_name', 'best_score', 'test_score', 'best_params']].sort_values(by='test_score', ascending=False)
    print(summary)

    # Sauvegarde CSV
    os.makedirs('rapport', exist_ok=True)
    summary.to_csv('rapport/resultats_finaux.csv', index=False)

    # ---------------------------------------------------------
    # 5. Bonus : Analyse approfondie du champion
    # ---------------------------------------------------------
    print(f"\n--- Analyse détaillée du champion : {best_model_name} ---")
    y_pred_final = best_overall_model.predict(X_test)
    print(classification_report(y_test, y_pred_final, target_names=classes))

    # ---------------------------------------------------------
    # 6. Génération Kaggle
    # ---------------------------------------------------------
    if hasattr(best_overall_model, "predict_proba"):
        print("Génération du fichier de soumission Kaggle...")
        y_proba = best_overall_model.predict_proba(X_submission)
        submission = pd.DataFrame(y_proba, columns=classes)
        submission.insert(0, 'id', submission_ids)
        submission.to_csv('rapport/submission.csv', index=False)

    # ---------------------------------------------------------
    # 7. Visualisations
    # ---------------------------------------------------------
    print("\nGénération des graphiques...")
    os.makedirs('rapport/figures', exist_ok=True)
    
    plot_cv_results(df_results)
    plot_confusion_matrix(y_test, y_pred_final, classes, title=f"Matrice de Confusion ({best_model_name})")
    
    print("Terminé ! Vérifie le dossier 'rapport/figures'.")

if __name__ == "__main__":
    main()