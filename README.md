# AI Image Detection

This project detects whether an image is real or AI-generated using statistical image analysis and machine learning.

The project builds on my HW5 image forensics work, where I compared image pairs, RGB histograms, blurred pixel differences, and simple rule-based detection on a small real-vs-AI shoe image dataset. For the final project, I expanded this idea into a complete AI image detection pipeline using the CIFAKE dataset.

The goal was to determine whether handcrafted statistical image features and machine learning models could successfully distinguish real images from AI-generated images.

## Project Goal

Can we detect whether an image is real or AI-generated using computational image analysis?

This problem is increasingly important because AI-generated images are becoming more realistic and can be used in misinformation, security threats, fraud, and digital forensics. Rather than relying solely on human judgment, this project extracts measurable image characteristics and trains machine learning models to classify images as REAL or FAKE. Developing a lightweight AI-image detector also provided a practical way to explore image forensics and machine learning techniques.

## Detection Pipeline

1. Collect real and AI-generated images
2. Preprocess images to a consistent size and format
3. Extract statistical image features
4. Train machine learning classifiers
5. Evaluate accuracy, precision, recall, F1 score, and runtime
6. Test robustness to resizing, blurring, and JPEG compression

## Features Extracted

The project extracts several categories of statistical image features.

### RGB Color Histograms

Measure how color values are distributed across the red, green, and blue channels.

### Intensity Statistics

Summarize image brightness using:

* Mean
* Standard deviation
* Minimum
* Maximum
* Percentiles

### Texture and Edge Features

Capture local image structure using:

* Local Binary Patterns (LBP)
* Sobel edge statistics
* Gray-Level Co-occurrence Matrix (GLCM) texture descriptors

### Frequency-Domain Features

Use Fast Fourier Transform (FFT) statistics to measure low-frequency, mid-frequency, and high-frequency image behavior. These features proved to be some of the strongest predictors of AI-generated content.

## Models Tested

The following machine learning models were evaluated:

* Logistic Regression
* Linear Support Vector Machine (SVM)
* RBF Support Vector Machine (SVM)
* Extra Trees
* Histogram Gradient Boosting
* Simple Convolutional Neural Network (CNN)

## Dataset

This project uses the CIFAKE dataset:

**CIFAKE: Real and AI-Generated Synthetic Images**

https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Dataset size:

* 60,000 real images
* 60,000 AI-generated images

The dataset is not included in this repository because it exceeds GitHub's file size limits.

## Downloading the Dataset

You must download the dataset manually before running the project.

Steps:

1. Create a Kaggle account if necessary
2. Download the CIFAKE dataset from the link above
3. Extract the ZIP file
4. Create a folder named `data` in the project root
5. Copy the extracted dataset into the `data` folder

The directory structure should look like:

```text
data/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

If the dataset is not downloaded and placed correctly, the training scripts will not run.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Project

### Train Statistical Feature Models

```bash
python src/train.py --data_dir data/train --limit 10000
```

This trains all statistical-feature-based machine learning models and saves the results to the `results` directory.

### Train the CNN

```bash
python src/train_cnn.py
```

This trains a simple Convolutional Neural Network directly on image pixels.

### Run Robustness Testing

```bash
python src/evaluate_robustness.py
```

This evaluates model performance after:

* Image resizing
* Gaussian blur
* JPEG compression

## Results

### Best Statistical Model

**RBF SVM (C = 2, gamma = scale)**

Performance:

* Accuracy: 91.0%
* Precision: 91.2%
* Recall: 90.7%
* F1 Score: 90.97%

### CNN Results

**Simple CNN**

Performance:

* Accuracy: 90.6%
* Precision: 92.0%
* Recall: 88.9%
* F1 Score: 90.4%

Interestingly, the handcrafted statistical feature approach achieved performance comparable to the CNN while requiring substantially less training time and computational resources. This suggests that carefully engineered statistical image features contain significant information for distinguishing real images from AI-generated images.

### Robustness Testing

Performance under image modifications:

| Condition        | Accuracy |
| ---------------- | -------- |
| Original         | 90.6%    |
| Resize 32×32     | 87.6%    |
| JPEG Compression | 82.0%    |
| Gaussian Blur    | 55.9%    |
| Resize 16×16     | 51.6%    |

These results suggest that the detector relies heavily on texture, edge, and frequency-domain information. When those signals are destroyed through aggressive resizing or blurring, performance drops significantly.

## Output Files

The project saves outputs in the `results/` directory.

Generated CSV files:

- model_results.csv
- cnn_results.csv
- robustness_results.csv
- feature_matrix.csv
- simple_cnn_model.pth
- *.png visualizations

Generated visualizations:

* Confusion matrices for each classifier
* Feature importance plots

Example PNG outputs:

* `linear_svm_confusion_matrix.png`
* `random_forest_feature_importances.png`

These visualizations can be opened directly to inspect model performance, classification errors, and feature importance rankings.

Additional run logs are included in:

* `output.md`

This file contains example console output from training, evaluation, robustness testing, and CNN experiments.

## Repository Structure

```text
src/
├── features.py
├── train.py
├── train_cnn.py
└── evaluate_robustness.py

results/
├── model_results.csv
├── cnn_results.csv
├── robustness_results.csv
├── feature_matrix.csv
└── *.png visualizations

README.md
requirements.txt
output.md
```

## Future Work

Potential improvements include:

* Training larger CNN architectures
* Testing additional AI-image datasets
* Evaluating newer image generation models
* Improving robustness to image manipulation
* Exploring transfer learning with pretrained vision models
* Combining statistical features with deep learning features