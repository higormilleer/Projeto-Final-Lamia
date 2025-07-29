# Instruções para Google Colab - Detecção de Pneumonia

## 🚀 Guia Rápido de Inicialização

### Passo 1: Acessar o Google Colab
1. Vá para [Google Colab](https://colab.research.google.com/)
2. Faça login com sua conta Google
3. Clique em "File" → "Upload notebook"
4. Faça upload do arquivo `pneumonia_detection.ipynb`

### Passo 2: Configurar GPU
1. Vá em "Runtime" → "Change runtime type"
2. Selecione "GPU" em "Hardware accelerator"
3. Clique em "Save"

### Passo 3: Configurar Kaggle API
1. Acesse [Kaggle](https://www.kaggle.com/)
2. Faça login e vá em "Account"
3. Clique em "Create New API Token"
4. Baixe o arquivo `kaggle.json`
5. No Colab, execute a célula que solicita upload do arquivo
6. Faça upload do `kaggle.json`

## 📋 Execução Passo a Passo

### 1. Instalação de Dependências
```python
# Execute a primeira célula para instalar as bibliotecas
!pip install tensorflow-gpu==2.10.0
!pip install opencv-python scikit-learn matplotlib seaborn plotly kaggle
```

### 2. Importações e Configuração
```python
# Execute a segunda célula para importar bibliotecas
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# ... (todas as importações)
```

### 3. Download do Dataset
```python
# Execute a célula de download
from google.colab import files
files.upload()  # Upload do kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip chest-xray-pneumonia.zip -d chest_xray_data
```

### 4. Análise Exploratória
```python
# Execute as células de análise
dataset_stats = analyze_dataset()
display_sample_images()
```

### 5. Pré-processamento
```python
# Execute a célula de data generators
# Configurações automáticas de data augmentation
```

### 6. Criação do Modelo
```python
# Execute a célula de criação do modelo
model = create_custom_cnn()
model = compile_model(model)
model.summary()
```

### 7. Treinamento
```python
# Execute a célula de treinamento
# Dura aproximadamente 30-60 minutos com GPU
history = model.fit(...)
```

### 8. Avaliação
```python
# Execute as células de avaliação
plot_training_history(history)
# Análise de métricas e matriz de confusão
```

## ⚠️ Solução de Problemas

### Erro: "GPU not available"
**Solução:**
1. Verifique se selecionou GPU em "Runtime" → "Change runtime type"
2. Reinicie o runtime: "Runtime" → "Restart runtime"
3. Execute novamente as células

### Erro: "Kaggle API not found"
**Solução:**
1. Certifique-se de que fez upload do `kaggle.json`
2. Verifique se o arquivo está no formato correto
3. Execute novamente a célula de configuração

### Erro: "Out of memory"
**Solução:**
1. Reduza o batch size de 32 para 16
2. Reduza o tamanho da imagem de 224 para 128
3. Use apenas parte do dataset para testes

### Erro: "Model training too slow"
**Solução:**
1. Verifique se está usando GPU
2. Reduza o número de épocas
3. Use um modelo mais simples

## 🔧 Configurações Avançadas

### Modificar Hiperparâmetros
```python
# No início do notebook, modifique estas variáveis:
IMG_SIZE = 224      # Tamanho da imagem (128, 224, 256)
BATCH_SIZE = 32     # Tamanho do batch (16, 32, 64)
EPOCHS = 50         # Número de épocas (20, 50, 100)
LEARNING_RATE = 0.001  # Taxa de aprendizado
```

### Usar Transfer Learning
```python
# Substitua a criação do modelo por:
transfer_model = create_transfer_learning_model()
transfer_model = compile_model(transfer_model, learning_rate=0.0001)
```

### Salvar e Carregar Modelo
```python
# Salvar modelo
model.save('meu_modelo_pneumonia.h5')

# Carregar modelo
from tensorflow import keras
model = keras.models.load_model('meu_modelo_pneumonia.h5')
```

## 📊 Monitoramento do Treinamento

### Métricas Importantes
- **Loss**: Deve diminuir consistentemente
- **Accuracy**: Deve aumentar e estabilizar
- **Validation Loss**: Não deve divergir muito do training loss
- **Overfitting**: Se validation loss aumenta enquanto training loss diminui

### Sinais de Problemas
- **Loss não diminui**: Learning rate muito baixo
- **Loss explode**: Learning rate muito alto
- **Overfitting**: Adicione mais dropout ou data augmentation
- **Underfitting**: Aumente complexidade do modelo

## 💾 Salvando Resultados

### Salvar Modelo
```python
# Salvar melhor modelo
model.save('best_pneumonia_model.h5')

# Salvar histórico de treinamento
import json
with open('training_history.json', 'w') as f:
    json.dump(history.history, f)
```

### Download de Arquivos
```python
# Download do modelo treinado
from google.colab import files
files.download('best_pneumonia_model.h5')

# Download dos resultados
files.download('model_results.json')
```

## 🎯 Dicas de Otimização

### Para Melhor Performance
1. **Use GPU**: Sempre selecione GPU no runtime
2. **Batch Size**: Use 32 ou 64 para GPU, 16 para CPU
3. **Data Augmentation**: Mantenha ativo para evitar overfitting
4. **Early Stopping**: Use para parar treinamento automaticamente

### Para Economizar Tempo
1. **Teste com Subset**: Use apenas 1000 imagens para testes
2. **Reduza Épocas**: Use 20 épocas para testes rápidos
3. **Imagem Menor**: Use 128x128 para treinamento mais rápido

### Para Melhor Acurácia
1. **Transfer Learning**: Use VGG16 ou ResNet
2. **Ensemble**: Combine múltiplos modelos
3. **Fine-tuning**: Ajuste learning rate durante treinamento

## 📱 Usando o Modelo Treinado

### Fazer Predições
```python
# Carregar modelo
model = keras.models.load_model('best_pneumonia_model.h5')

# Fazer predição
import cv2
img = cv2.imread('imagem.jpg')
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)[0][0]
print(f"Probabilidade de pneumonia: {prediction:.2%}")
```

### Interface Web Simples
```python
# Criar interface no Colab
from google.colab import files
import matplotlib.pyplot as plt

uploaded = files.upload()
for filename in uploaded.keys():
    img = cv2.imread(filename)
    # Processar e fazer predição
    # Mostrar resultado
```

## 🔍 Debugging

### Verificar GPU
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("TensorFlow version: ", tf.__version__)
```

### Verificar Dataset
```python
import os
print("Arquivos no dataset:")
!find chest_xray_data -name "*.jpeg" | head -10
```

### Verificar Memória
```python
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

## 📞 Suporte

### Recursos Úteis
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

### Comunidade
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Kaggle Discussions](https://www.kaggle.com/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow)

---

**Nota**: Este projeto é para fins educacionais. Para uso clínico, consulte profissionais médicos e valide adequadamente o modelo. 