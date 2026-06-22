# Classificacao_CNN_Andre
# Equipe
André Luiz da Silva Junior

# Descrição da abordagem utilizada

Este projeto utiliza Redes Neurais Convolucionais com Transfer Learning e Fine-Tuning para classificar imagens de grãos de café em quatro categorias: Dark, Green, Light e Medium.

O modelo base é o DenseNet121 pré-treinado no ImageNet, escolhido por suas conexões densas e equilíbrio entre capacidade e eficiência computacional, com cerca de 7 milhões de parâmetros. Sobre ele, foram adicionadas camadas de BatchNormalization, GlobalAveragePooling2D, duas camadas Dropout (30% e 50%), uma camada densa com 64 neurônios e regularização L2, e uma camada de saída com 4 neurônios e ativação Softmax.

O treinamento ocorreu em duas etapas: feature extraction com o modelo congelado por 20 épocas (taxa de aprendizado 1e-4) e fine-tuning com as últimas 30 camadas descongeladas por 10 épocas (taxa de aprendizado 1e-5), mantendo as camadas de BatchNormalization congeladas para evitar esquecimento catastrófico.

O pré-processamento utilizou a função preprocess_input da DenseNet121, com data augmentation (flips vertical e horizontal) apenas nos dados de treino. O dataset foi dividido em treino (1200 imagens), validação (200) e teste (200).

O modelo alcançou 99,5% de acurácia no teste, com apenas um erro. A perda reduziu de 0,1798 para 0,1531 após o fine-tuning. Todas as métricas por classe (precisão, recall e F1-score) ficaram acima de 0,98.
# Dataset 
https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224

# Repositório
https://github.com/SwampingTheSwamp/Classificacao_CNN_Andre

# Vídeo

# Acurácia(s) obtida(s)
Epoch 1/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 339s 8s/step - accuracy: 0.2908 - loss: 1.5729 - val_accuracy: 0.3100 - val_loss: 1.4044 - learning_rate: 1.0000e-04
Epoch 2/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 262s 7s/step - accuracy: 0.4742 - loss: 1.2845 - val_accuracy: 0.6050 - val_loss: 1.0940 - learning_rate: 1.0000e-04
Epoch 3/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 266s 7s/step - accuracy: 0.6158 - loss: 1.0755 - val_accuracy: 0.7800 - val_loss: 0.9013 - learning_rate: 1.0000e-04
Epoch 4/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 279s 7s/step - accuracy: 0.7200 - loss: 0.9235 - val_accuracy: 0.9050 - val_loss: 0.7654 - learning_rate: 1.0000e-04
Epoch 5/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 267s 7s/step - accuracy: 0.7783 - loss: 0.8118 - val_accuracy: 0.9100 - val_loss: 0.6587 - learning_rate: 1.0000e-04
Epoch 6/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 281s 7s/step - accuracy: 0.8367 - loss: 0.6952 - val_accuracy: 0.9300 - val_loss: 0.5730 - learning_rate: 1.0000e-04
Epoch 7/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 266s 7s/step - accuracy: 0.8475 - loss: 0.6333 - val_accuracy: 0.9500 - val_loss: 0.5024 - learning_rate: 1.0000e-04
Epoch 8/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 333s 7s/step - accuracy: 0.8533 - loss: 0.5690 - val_accuracy: 0.9650 - val_loss: 0.4464 - learning_rate: 1.0000e-04
Epoch 9/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 271s 7s/step - accuracy: 0.8867 - loss: 0.5288 - val_accuracy: 0.9650 - val_loss: 0.3998 - learning_rate: 1.0000e-04
Epoch 10/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 281s 7s/step - accuracy: 0.8950 - loss: 0.4761 - val_accuracy: 0.9750 - val_loss: 0.3603 - learning_rate: 1.0000e-04
Epoch 11/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 270s 7s/step - accuracy: 0.9142 - loss: 0.4474 - val_accuracy: 0.9750 - val_loss: 0.3266 - learning_rate: 1.0000e-04
Epoch 12/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 274s 7s/step - accuracy: 0.9108 - loss: 0.4174 - val_accuracy: 0.9750 - val_loss: 0.2984 - learning_rate: 1.0000e-04
Epoch 13/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 267s 7s/step - accuracy: 0.9150 - loss: 0.4068 - val_accuracy: 0.9750 - val_loss: 0.2757 - learning_rate: 1.0000e-04
Epoch 14/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 265s 7s/step - accuracy: 0.9333 - loss: 0.3717 - val_accuracy: 0.9750 - val_loss: 0.2553 - learning_rate: 1.0000e-04
Epoch 15/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 273s 7s/step - accuracy: 0.9342 - loss: 0.3534 - val_accuracy: 0.9800 - val_loss: 0.2369 - learning_rate: 1.0000e-04
Epoch 16/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 267s 7s/step - accuracy: 0.9417 - loss: 0.3462 - val_accuracy: 0.9850 - val_loss: 0.2226 - learning_rate: 1.0000e-04
Epoch 17/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 268s 7s/step - accuracy: 0.9400 - loss: 0.3194 - val_accuracy: 0.9850 - val_loss: 0.2092 - learning_rate: 1.0000e-04
Epoch 18/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 268s 7s/step - accuracy: 0.9467 - loss: 0.3041 - val_accuracy: 0.9900 - val_loss: 0.1972 - learning_rate: 1.0000e-04
Epoch 19/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 269s 7s/step - accuracy: 0.9367 - loss: 0.3036 - val_accuracy: 0.9950 - val_loss: 0.1887 - learning_rate: 1.0000e-04
Epoch 20/20
38/38 ━━━━━━━━━━━━━━━━━━━━ 266s 7s/step - accuracy: 0.9433 - loss: 0.2935 - val_accuracy: 0.9950 - val_loss: 0.1798 - learning_rate: 1.0000e-04
Restoring model weights from the end of the best epoch: 20.

Fine-tuning: 30 últimas camadas do DenseNet121 foram descongeladas.

Epoch 1/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 300s 7s/step - accuracy: 0.9558 - loss: 0.2776 - val_accuracy: 0.9950 - val_loss: 0.1840 - learning_rate: 1.0000e-05
Epoch 2/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 278s 7s/step - accuracy: 0.9542 - loss: 0.2727 - val_accuracy: 0.9950 - val_loss: 0.1788 - learning_rate: 1.0000e-05
Epoch 3/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 276s 7s/step - accuracy: 0.9525 - loss: 0.2720 - val_accuracy: 0.9950 - val_loss: 0.1740 - learning_rate: 1.0000e-05
Epoch 4/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 274s 7s/step - accuracy: 0.9517 - loss: 0.2861 - val_accuracy: 0.9950 - val_loss: 0.1703 - learning_rate: 1.0000e-05
Epoch 5/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 274s 7s/step - accuracy: 0.9442 - loss: 0.2736 - val_accuracy: 0.9950 - val_loss: 0.1666 - learning_rate: 1.0000e-05
Epoch 6/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 274s 7s/step - accuracy: 0.9542 - loss: 0.2657 - val_accuracy: 0.9950 - val_loss: 0.1638 - learning_rate: 1.0000e-05
Epoch 7/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 270s 7s/step - accuracy: 0.9542 - loss: 0.2668 - val_accuracy: 0.9950 - val_loss: 0.1617 - learning_rate: 1.0000e-05
Epoch 8/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 331s 7s/step - accuracy: 0.9650 - loss: 0.2460 - val_accuracy: 0.9950 - val_loss: 0.1590 - learning_rate: 1.0000e-05
Epoch 9/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 276s 7s/step - accuracy: 0.9642 - loss: 0.2446 - val_accuracy: 0.9950 - val_loss: 0.1563 - learning_rate: 1.0000e-05
Epoch 10/10
38/38 ━━━━━━━━━━━━━━━━━━━━ 270s 7s/step - accuracy: 0.9525 - loss: 0.2635 - val_accuracy: 0.9950 - val_loss: 0.1531 - learning_rate: 1.0000e-05
Restoring model weights from the end of the best epoch: 10.
