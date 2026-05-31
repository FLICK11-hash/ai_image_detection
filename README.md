# AI Image Detection

This project detects whether an image is real or AI-generated using statistical image analysis and machine learning.

The project builds on my HW5 image forensics work, where I compared image pairs, RGB histograms, blurred pixel differences, and simple rule-based detection on a small real-vs-AI shoe image dataset. For the final project, I expanded this into a full detection pipeline using the CIFAKE dataset.

The best-performing model was an RBF Support Vector Machine (SVM), which achieved approximately 91% accuracy on unseen test images. Feature importance analysis showed that texture, edge, and frequency-domain features were the strongest indicators of AI-generated content.

## Project Goal

Can we detect whether an image is real or AI-generated using computational image analysis?

This is important because AI-generated images are becoming increasingly realistic and can be used in misinformation, security threats, and digital forensics. Instead of relying on human judgment alone, this project extracts measurable image characteristics and trains machine learning models to classify images as real or AI-generated.

## Detection Pipeline

* Collect real and AI-generated images
* Preprocess images to a consistent size and format
* Extract statistical image features
* Train machine learning classifiers
* Evaluate accuracy, precision, recall, F1 score, and runtime
* Test robustness to resizing, blurring, and JPEG compression

## Features Extracted

### RGB Color Histograms

Measure how color values are distributed across the red, green, and blue channels.

### Intensity Statistics

Summarize brightness patterns using:

* Mean
* Standard deviation
* Minimum
* Maximum
* Percentiles

### Texture and Edge Features

Capture local image patterns using:

* Local Binary Patterns (LBP)
* Edge statistics
* Texture descriptors

### Frequency-Domain Features

Use Fast Fourier Transform (FFT) statistics to measure frequency patterns that often differ between real and AI-generated images.

## Models Tested

* Logistic Regression
* Linear Support Vector Machine (SVM)
* RBF Support Vector Machine (SVM)
* Random Forest
* Extra Trees
* Histogram Gradient Boosting
* Simple Convolutional Neural Network (CNN)

## Dataset

This project uses the CIFAKE dataset:

**CIFAKE: Real and AI-Generated Synthetic Images**

https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

The dataset contains:

* 60,000 real images
* 60,000 AI-generated images

The dataset is not included in this repository because it exceeds GitHub's file size limits.

## Downloading the Dataset

You must download the CIFAKE dataset manually from Kaggle before running the project.

Steps:

1. Create a Kaggle account if needed
2. Download the dataset from the link above
3. Extract the ZIP file
4. Create a folder named `data` in the project root
5. Copy the extracted dataset into the `data` folder

The final directory structure should look like:

```text
data/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

If the dataset is not downloaded and placed correctly, the training and evaluation scripts will not run.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

### Train Statistical Feature Models

```bash
python src/train.py --data_dir data/train --limit 10000
```

This trains the machine learning models and saves results to the `results/` folder.

### Train the CNN Baseline

```bash
python src/train_cnn.py
```

### Run Robustness Testing

```bash
python src/evaluate_robustness.py
```

This evaluates model performance after:

* Image resizing
* Gaussian blur
* JPEG compression

## Results

Best statistical model:

* RBF SVM
* Approximately 91% accuracy

Best CNN model:

* Simple CNN
* Approximately 91% accuracy

Robustness testing showed:

* Moderate resizing had only a small impact on performance
* JPEG compression reduced performance moderately
* Heavy resizing and blurring significantly reduced accuracy

These results suggest that the detector relies heavily on texture, edge, and frequency-domain information.

## Output Files

The project saves outputs in the `results/` directory:

- `model_results.csv`
- `cnn_results.csv`
- `robustness_results.csv`
- `feature_matrix.csv`
- Confusion matrix visualizations
- Feature importance visualizations

The generated PNG files include:

- `logistic_regression_confusion_matrix.png`
- `linear_svm_confusion_matrix.png`
- `random_forest_confusion_matrix.png`
- `random_forest_feature_importances.png`

These visualizations can be opened directly to inspect model performance and feature importance. They should be easily accessible from when I ran them.

These images provide an easy way to understand model accuracy, common classification errors, and which image features contributed most to the predictions.

## Future Work

Possible improvements include:

* Training larger CNN architectures
* Testing on additional AI-image datasets
* Adding more frequency-domain features
* Evaluating newer generative models
* Improving robustness to image manipulation
