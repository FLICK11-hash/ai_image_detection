(.venv) PS C:\Users\conra\ai_image_detection> python src\train.py --data_dir data\train --limit 10000
                      model  accuracy  precision  recall        f1  runtime_seconds
0           Linear SVM C=10    0.8975   0.897103   0.898  0.897551         1.763072
1   RBF SVM C=1 gamma=scale    0.9020   0.902806   0.901  0.901902         1.983967
2    RBF SVM C=1 gamma=0.01    0.9045   0.905717   0.903  0.904357         1.809735
3    RBF SVM C=1 gamma=0.05    0.8840   0.877210   0.893  0.885035         3.119149
4   RBF SVM C=2 gamma=scale    0.9100   0.912475   0.907  0.909729         1.729029
5  Logistic Regression C=10    0.8970   0.896208   0.898  0.897103         0.501109
6               Extra Trees    0.8855   0.876098   0.898  0.886914         1.084642
7    Hist Gradient Boosting    0.8945   0.888670   0.902  0.895285         6.465440
(.venv) PS C:\Users\conra\ai_image_detection> python src\evaluate_robustness.py
Testing: original
Testing: resize_16
Testing: resize_32
Testing: blur
Testing: jpeg_compression
          condition  accuracy  precision    recall        f1  runtime_seconds
0          original  0.905714   0.903409  0.908571  0.905983       239.838899
1         resize_16  0.516429   0.870968  0.038571  0.073871       239.838899
2         resize_32  0.876429   0.875892  0.877143  0.876517       239.838899
3              blur  0.558571   0.836066  0.145714  0.248175       239.838899
4  jpeg_compression  0.820000   0.944444  0.680000  0.790698       239.838899
(.venv) PS C:\Users\conra\ai_image_detection> python src\train_cnn.py
Using device: cpu
Epoch 1/10 | Loss: 84.1900 | Accuracy: 0.8300
Epoch 2/10 | Loss: 61.0365 | Accuracy: 0.8485
Epoch 3/10 | Loss: 51.0307 | Accuracy: 0.8840
Epoch 4/10 | Loss: 45.7105 | Accuracy: 0.8845
Epoch 5/10 | Loss: 39.9203 | Accuracy: 0.8855
Epoch 6/10 | Loss: 35.2215 | Accuracy: 0.9030
Epoch 7/10 | Loss: 31.7314 | Accuracy: 0.9010
Epoch 8/10 | Loss: 27.3567 | Accuracy: 0.9020
Epoch 9/10 | Loss: 23.6490 | Accuracy: 0.9065
Epoch 10/10 | Loss: 20.3975 | Accuracy: 0.9060
   accuracy  precision  recall        f1       model  runtime_seconds
0     0.906    0.92029   0.889  0.904374  Simple CNN       425.199538