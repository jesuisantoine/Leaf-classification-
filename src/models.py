from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

def get_model_configs():
    """
    Retourne un dictionnaire contenant les modèles à tester 
    et leurs grilles d'hyper-paramètres respectives.
    """
    models_config = {
        # Modèle 1 : KNN (Basé sur la distance)
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        # Modèle 2 : SVM (Le classique robuste)
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        },
        # Modèle 3 : Random Forest (Ensemble Bagging)
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100], # Augmenter à 200+ pour le rapport final
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        # Modèle 4 : MLP (Réseau de neurones simple)
        'MLP': {
            'model': MLPClassifier(max_iter=1000, random_state=42),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['tanh', 'relu'],
                'alpha': [0.0001, 0.001]
            }
        },
        # Modèle 5 : Gradient Boosting (Ensemble Boosting - Souvent le meilleur)
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5]
            }
        },
         # Modèle 6 : Régression Logistique (Baseline simple)
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        }
    }
    return models_config