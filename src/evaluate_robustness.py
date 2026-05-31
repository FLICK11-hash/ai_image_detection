import time
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from skimage import io

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from features import extract_features

def build_dataset(data_dir, limit=7000):
    # Load a balanced subset of REAL and FAKE images for robustness evaluation
    rows, labels, paths = [], [], []

    for label_name, label in [("REAL", 0), ("FAKE", 1)]:
        files = list((Path(data_dir) / label_name).glob("*"))
        random.seed(50)
        random.shuffle(files)
        files = files[: limit // 2]

        for path in files:
            rows.append(extract_features(path))
            labels.append(label)
            paths.append(path)

    return pd.DataFrame(rows), pd.Series(labels), paths


def transform_image(path, mode):
    # Apply image transformations to test how robust the detector is to image modifications
    img = Image.open(path).convert("RGB")

    if mode == "original":
        return img

    if mode == "resize_16":
        return img.resize((16, 16)).resize((64, 64))

    if mode == "resize_32":
        return img.resize((32, 32)).resize((64, 64))

    if mode == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Simulate lossy JPEG compression that may remove useful image artifacts
    if mode == "jpeg_compression":
        temp_path = "results/temp_compressed.jpg"
        img.save(temp_path, "JPEG", quality=35)
        return Image.open(temp_path).convert("RGB")

    return img


def extract_features_from_pil(img):
    # Extract the same statistical features from transformed images used during training
    temp_path = "results/temp_robustness_image.jpg"
    img.save(temp_path)
    return extract_features(temp_path)


def evaluate_on_transformed_images(model, test_paths, y_test, mode):
    # Measure model performance after applying a specific image transformation
    rows = []

    for path in test_paths:
        img = transform_image(path, mode)
        rows.append(extract_features_from_pil(img))

    X_test_transformed = pd.DataFrame(rows)
    preds = model.predict(X_test_transformed)

    return {
        "condition": mode,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }


def main():
    Path("results").mkdir(exist_ok=True)

    X, y, paths = build_dataset("data/train", limit=7000)

    X_train, X_test, y_train, y_test, _, test_paths = train_test_split(
        X,
        y,
        paths,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Recreate the best-performing RBF SVM model discovered during training
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=2, gamma="scale")),
    ])

    start = time.time()
    model.fit(X_train, y_train)

    # Test several realistic image manipulations that may affect detector accuracy
    conditions = [
        "original",
        "resize_16",
        "resize_32",
        "blur",
        "jpeg_compression",
    ]

    results = []

    for condition in conditions:
        print(f"Testing: {condition}")
        results.append(evaluate_on_transformed_images(model, test_paths, y_test, condition))

    df = pd.DataFrame(results)
    df["runtime_seconds"] = time.time() - start

    df.to_csv("results/robustness_results.csv", index=False)
    print(df)


if __name__ == "__main__":
    main()