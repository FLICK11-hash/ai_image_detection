import argparse, time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from features import extract_features, get_feature_names
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random



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
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        "Linear SVM": Pipeline([("scale", StandardScaler()), ("clf", LinearSVC(max_iter=5000))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = [evaluate_model(name, model, X_train, X_test, y_train, y_test) for name, model in models.items()]

    rf = models["Random Forest"]
    importances = rf.feature_importances_

    feature_names = get_feature_names()

    importance_df = pd.DataFrame({
        "feature_index": range(len(importances)),
        "feature_name": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv("results/random_forest_feature_importances.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature_name"].head(20), importance_df["importance"].head(20))
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.ylabel("Importance")
    plt.title("Top 20 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig("results/random_forest_feature_importances.png")
    plt.close()

    pd.DataFrame(results).to_csv("results/model_results.csv", index=False)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
