# AI Image Detection

This project detects whether an image is real or AI-generated using statistical image analysis 
and machine learning. It builds on my HW5 image forensics work, where I compared image pairs, 
RGB histograms, blurred pixel differences, and simple rule-based detection on a 
small real-vs-AI shoe image set.
For the final project, I expanded this into a stronger detection pipeline using the CIFAKE dataset,
which contains 60,000 real images and 60,000 AI-generated images. 
Due to GitHub size limitations, the CIFAKE dataset is not included.
Download it from Kaggle and place it in the data/ directory 
before running the training scripts.
For CIFAKE specifically, the images are already standardized, 
usually 32×32 RGB images, since CIFAKE is based on CIFAR-style images.
Preprocessing images to the same size and format is important because the classifier should learn
differences between real and AI-generated images, not differences caused by image dimensions, 
file types, or color formats.

## Project Goal

Can we detect whether an image is real or AI-generated using computational image analysis?

This matters because AI-generated images are increasingly realistic and can be used in
misinformation, security problems, and digital forensics. 
Instead of relying only on human judgment, this project extracts measurable image features 
and trains classifiers to separate real images from AI-generated ones.

## Detection Pipeline

1. Collect real and AI-generated images (X)
2. Preprocess images to the same size and format (X)
3. Extract statistical image features
4. Train machine learning classifiers
5. Evaluate accuracy, precision, recall, and runtime
6. Test robustness to resizing and compression

(an X means I have completed it for my own notes)

## Features Extracted

The project extracts several types of statistical features:

- **RGB color histograms**: measure how color values are distributed across red, green, and blue channels
- **Pixel intensity statistics**: summarize brightness patterns using mean, standard deviation, minimum, maximum, and percentiles
- **Noise and texture features**: capture local variation and edge/detail patterns that may differ between real and generated images
- **FFT frequency features**: measure patterns in the frequency domain, where AI-generated images may show different high-frequency or low-frequency behavior

## Models

The main classifiers are:

- Logistic Regression
- Support Vector Machine
- Random Forest

These models are intentionally simple so the project can focus on whether statistical image features alone contain useful signal.

## Dataset

This project is designed for the CIFAKE dataset from Kaggle:

**CIFAKE: Real and AI-Generated Synthetic Images**  
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Expected folder format:

```text
data/
  real/
    image1.png
    image2.png
  fake/
    image1.png
    image2.png
```

## How to Run

Install requirements:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python src/train.py --data_dir data --limit 5000
```

Test robustness to resizing and JPEG compression:

```bash
python src/evaluate_robustness.py --data_dir data --limit 1000
```

## Outputs

The scripts save results in the `results/` folder:

- `model_results.csv`: accuracy, precision, recall, F1 score, and runtime for each model
- `robustness_results.csv`: performance after resizing and compression
- `feature_matrix.csv`: extracted features and labels

## Future Work

Possible improvements include testing more datasets, comparing more classifiers, adding deep learning features, and studying how resizing or compression affects detection accuracy.
