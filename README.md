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


### Feature Extraction (20 épocas)

| Época | Acurácia Treino | Loss Treino | Acurácia Validação | Loss Validação |
|-------|-----------------|-------------|--------------------|----------------|
| 1     | 0.2908          | 1.5729      | 0.3100             | 1.4044         |
| 2     | 0.4742          | 1.2845      | 0.6050             | 1.0940         |
| 3     | 0.6158          | 1.0755      | 0.7800             | 0.9013         |
| 4     | 0.7200          | 0.9235      | 0.9050             | 0.7654         |
| 5     | 0.7783          | 0.8118      | 0.9100             | 0.6587         |
| 6     | 0.8367          | 0.6952      | 0.9300             | 0.5730         |
| 7     | 0.8475          | 0.6333      | 0.9500             | 0.5024         |
| 8     | 0.8533          | 0.5690      | 0.9650             | 0.4464         |
| 9     | 0.8867          | 0.5288      | 0.9650             | 0.3998         |
| 10    | 0.8950          | 0.4761      | 0.9750             | 0.3603         |
| 11    | 0.9142          | 0.4474      | 0.9750             | 0.3266         |
| 12    | 0.9108          | 0.4174      | 0.9750             | 0.2984         |
| 13    | 0.9150          | 0.4068      | 0.9750             | 0.2757         |
| 14    | 0.9333          | 0.3717      | 0.9750             | 0.2553         |
| 15    | 0.9342          | 0.3534      | 0.9800             | 0.2369         |
| 16    | 0.9417          | 0.3462      | 0.9850             | 0.2226         |
| 17    | 0.9400          | 0.3194      | 0.9850             | 0.2092         |
| 18    | 0.9467          | 0.3041      | 0.9900             | 0.1972         |
| 19    | 0.9367          | 0.3036      | 0.9950             | 0.1887         |
| 20    | 0.9433          | 0.2935      | 0.9950             | 0.1798         |

**Melhor época:** 20  
**Melhor acurácia de validação:** 99,50%

---

### Fine-Tuning (10 épocas)

Descongeladas as últimas 30 camadas do DenseNet121 com taxa de aprendizado reduzida (1e-5).

| Época | Acurácia Treino | Loss Treino | Acurácia Validação | Loss Validação |
|-------|-----------------|-------------|--------------------|----------------|
| 1     | 0.9558          | 0.2776      | 0.9950             | 0.1840         |
| 2     | 0.9542          | 0.2727      | 0.9950             | 0.1788         |
| 3     | 0.9525          | 0.2720      | 0.9950             | 0.1740         |
| 4     | 0.9517          | 0.2861      | 0.9950             | 0.1703         |
| 5     | 0.9442          | 0.2736      | 0.9950             | 0.1666         |
| 6     | 0.9542          | 0.2657      | 0.9950             | 0.1638         |
| 7     | 0.9542          | 0.2668      | 0.9950             | 0.1617         |
| 8     | 0.9650          | 0.2460      | 0.9950             | 0.1590         |
| 9     | 0.9642          | 0.2446      | 0.9950             | 0.1563         |
| 10    | 0.9525          | 0.2635      | 0.9950             | 0.1531         |

**Melhor época:** 10  
**Melhor acurácia de validação:** 99,50%

---

### Resumo dos Resultados

| Etapa | Acurácia Final | Loss Final |
|-------|----------------|------------|
| Feature Extraction | 99,50% | 0,1798 |
| Fine-Tuning | 99,50% | 0,1531 |

O fine-tuning reduziu a perda (loss) em aproximadamente 15%, indicando maior confiança do modelo nas previsões, mantendo a acurácia em 99,50%.
38/38 ━━━━━━━━━━━━━━━━━━━━ 270s 7s/step - accuracy: 0.9525 - loss: 0.2635 - val_accuracy: 0.9950 - val_loss: 0.1531 - learning_rate: 1.0000e-05
Restoring model weights from the end of the best epoch: 10.
