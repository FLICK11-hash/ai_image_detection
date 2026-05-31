import argparse, time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from features import extract_features, get_feature_names
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier



def build_dataset(data_dir, limit=None):
    rows, labels = [], []
    for label_name, label in [("REAL", 0), ("FAKE", 1)]:
        files = list((Path(data_dir) / label_name).glob("*"))
        random.seed(50)
        random.shuffle(files)

        if limit:
            files = files[: limit // 2]
        for path in files:
            try:
                rows.append(extract_features(path))
                labels.append(label)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    return pd.DataFrame(rows), pd.Series(labels, name="label")


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REAL", "FAKE"])
    disp.plot()
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"results/{name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "runtime_seconds": time.time() - start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    X, y = build_dataset(args.data_dir, args.limit)
    X.to_csv("results/feature_matrix.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
    "Linear SVM C=10": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LinearSVC(C=10, max_iter=10000))
    ]),

    "RBF SVM C=1 gamma=scale": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1, gamma="scale"))
    ]),

    "RBF SVM C=1 gamma=0.01": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1, gamma=0.01))
    ]),

    "RBF SVM C=1 gamma=0.05": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1, gamma=0.05))
    ]),

    "RBF SVM C=2 gamma=scale": Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=2, gamma="scale"))
    ]),

    "Logistic Regression C=10": Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=10, max_iter=2000))
    ]),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        random_state=42
    ),
}

    results = [evaluate_model(name, model, X_train, X_test, y_train, y_test) for name, model in models.items()]

    pd.DataFrame(results).to_csv("results/model_results.csv", index=False)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
