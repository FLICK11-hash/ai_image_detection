AI Image Detection Using Statistical Image Features and Machine Learning
Introduction
AI-generated images are becoming increasingly realistic, making it harder to distinguish them from real photographs. This has important implications for misinformation, fraud, cybersecurity, and digital forensics. The goal of this project was to determine whether statistical image features and machine learning could accurately classify images as real or AI-generated from the perspective of a graduate student at the University of Chicago.
Background
Previous research has shown that AI-generated images often contain subtle differences in texture, color distributions, and frequency patterns. While state-of-the-art detectors can achieve accuracy above 95%, they often require large deep learning models, significant computational resources, and extensive development effort.
Methodology
This project used the CIFAKE dataset, which contains 60,000 real images and 60,000 AI-generated images.
The following features were extracted from each image:
RGB color histograms
Intensity statistics
Texture features
Edge statistics
FFT frequency features
Several machine learning models were tested, including Logistic Regression, Support Vector Machines (SVMs), Extra Trees, and Histogram Gradient Boosting. A simple CNN was also trained for comparison.
Results
The best-performing model was an RBF SVM with an accuracy of 91.0%, precision of 91.2%, recall of 90.7%, and an F1 score of 90.97%.
The CNN achieved a similar accuracy of 90.6%.
Robustness testing showed that performance remained relatively strong under moderate resizing and JPEG compression but dropped significantly under heavy resizing and image blurring.
Feature importance analysis showed that frequency-domain features, texture features, and edge statistics were the strongest predictors of AI-generated content.
Discussion
The project successfully demonstrated that statistical image features can distinguish real images from AI-generated images with approximately 91% accuracy. One interesting result was that the handcrafted feature approach performed nearly as well as the CNN, suggesting that texture, edge, and frequency-domain information contain substantial signals for AI-image detection.
This project helps the field by showing that strong AI-image detection performance can be achieved without large deep learning models or extensive computational resources. Simpler and more interpretable approaches may be useful for students, researchers, and organizations that need lightweight detection tools.
While the detector does not outperform state-of-the-art systems that often achieve accuracies above 95%, it provides a simpler and more transparent approach while still achieving competitive performance. The results suggest that carefully engineered statistical features remain valuable even as deep learning continues to dominate computer vision research.
With additional time and resources, future improvements could include larger CNN architectures, transfer learning with pretrained models, additional datasets, and more advanced feature engineering. A future version could also include a UI that allows users to upload images and receive predictions directly through an easy-to-use interface.
References
Birdy654. CIFAKE: Real and AI-Generated Synthetic Images. Kaggle, 2024. https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
Wang, Sheng-Yu, et al. CNN-Generated Images Are Surprisingly Easy to Spot... for Now. CVPR, 2020.
Corvi, Riccardo, et al. Detection of GAN-Generated Images Using Frequency Analysis. International Conference on Image Analysis and Processing, 2021.

# Could not find the submission for the write-up so added this file and the url to my paper below:
https://docs.google.com/document/d/1RYiieFqj5qVWznfNbb-ixV9ROit-Dq3YDinz9Ht8mhI/edit?usp=sharing
