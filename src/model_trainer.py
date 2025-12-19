from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def train_and_evaluate(self, model_name, config):
        # 1. Pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', config['model'])
        ])

        pipeline_params = {f'clf__{k}': v for k, v in config['params'].items()}

        # 2. Grid Search (Verbose=0 pour ne pas casser tqdm)
        grid = GridSearchCV(
            pipeline, 
            pipeline_params, 
            cv=self.cv, 
            scoring='accuracy', 
            n_jobs=-1,          
            verbose=0  # <--- CORRECTION ICI
        )

        grid.fit(self.X, self.y)

        # 3. Extraction des métriques scientifiques (Moyenne ET Écart-type)
        best_index = grid.best_index_
        std_score = grid.cv_results_['std_test_score'][best_index]

        return {
            'model_name': model_name,
            'best_model': grid.best_estimator_,
            'best_params': grid.best_params_,
            'best_score': grid.best_score_, # Moyenne
            'std_score': std_score          # Écart-type (pour les barres d'erreur)
        }