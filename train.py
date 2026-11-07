"""TP1 - train.py

ROLE      : Premier script de production : charge Iris, entraine une foret aleatoire, sauvegarde le modele.
ENTREE    : le jeu Iris de scikit-learn
SORTIE    : models/iris_model.pkl (ignore par Git grace au .gitignore)
A RETENIR : Un script s'execute de haut en bas en une commande (python train.py), contrairement a un notebook.
"""

from pathlib import Path
import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


MODEL_PATH = Path("models") / "iris_model.pkl"


def execute_pipeline() -> float:
    """Run the training pipeline and return the test accuracy."""
    print("[MLOps Pipeline] Starting pipeline execution...")

    raw_data = load_iris(as_frame=True)
    dataframe = raw_data.frame
    features = dataframe.iloc[:, :-1]
    target = dataframe.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(x_train, y_train)
    accuracy = classifier.score(x_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(classifier, model_file)

    print(
        "[MLOps Pipeline] Target model successfully cached! "
        f"Score: {accuracy:.4f}"
    )
    return accuracy


if __name__ == "__main__":
    execute_pipeline()
