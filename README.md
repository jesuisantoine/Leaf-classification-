## Architecture du Projet

```text
IFT712-Projet-Leaf/
├── data/
│   └── raw/                # Données brutes de Kaggle
├── rapport/
│   ├── figures/            # Diagrammes et courbes de performance
│   └── rapport_final.pdf   # Rapport de session (PDF)
├── src/
│   ├── data_loader.py      # Classe LeafDataLoader
│   ├── models.py           # Configuration des modèles (Pattern Factory)
│   ├── model_trainer.py    # Classe ModelTrainer (GridSearchCV + Pipeline)
│   └── visualization.py    # Fonctions pour les graphiques
├── tests/
│   └── test_loader.py      # Tests unitaires du loader
├── .gitignore              # Fichiers exclus de Git
├── README.md               # Documentation générale
├── requirements.txt        # Dépendances (Scikit-Learn, Pandas, etc.)
└── main.py                 # Script principal (Point d'entrée)
