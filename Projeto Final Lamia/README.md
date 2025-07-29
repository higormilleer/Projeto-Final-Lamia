# Detecção de Pneumonia em Raios-X do Tórax

## 📋 Descrição do Projeto

Este projeto implementa um sistema de detecção automática de pneumonia em imagens de raios-X do tórax usando Deep Learning com Redes Neurais Convolucionais (CNN). O objetivo é classificar imagens em duas categorias: **Normal** (sem pneumonia) e **Pneumonia** (com pneumonia bacteriana ou viral).

### 🎯 Objetivos
- Desenvolver um modelo de IA para triagem médica
- Automatizar a detecção de pneumonia em raios-X
- Fornecer suporte diagnóstico para profissionais de saúde
- Demonstrar aplicações práticas de Deep Learning em medicina

## 📊 Dataset

**Fonte**: [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Estrutura dos Dados
```
chest_xray/
├── train/
│   ├── NORMAL/     (1,341 imagens)
│   └── PNEUMONIA/  (3,875 imagens)
├── test/
│   ├── NORMAL/     (234 imagens)
│   └── PNEUMONIA/  (390 imagens)
└── val/
    ├── NORMAL/     (8 imagens)
    └── PNEUMONIA/  (8 imagens)
```

**Total**: 5,856 imagens de raios-X

### Características
- **Formato**: JPEG
- **Resolução**: Variável (normalmente 1024x1024)
- **Classes**: 2 (Normal, Pneumonia)
- **Desbalanceamento**: ~3:1 (Pneumonia:Normal)

## 🏗️ Arquitetura do Modelo

### 1. CNN Customizada
```python
# Estrutura da rede
- Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)
- Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
- Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25)
- Conv2D(256) + BatchNorm + MaxPool + Dropout(0.25)
- Flatten
- Dense(512) + BatchNorm + Dropout(0.5)
- Dense(256) + BatchNorm + Dropout(0.5)
- Dense(1, sigmoid)
```

### 2. Transfer Learning (VGG16)
- Base model: VGG16 pré-treinado no ImageNet
- Camadas customizadas para classificação binária
- Fine-tuning das camadas finais

## 🚀 Como Executar

### Pré-requisitos
```bash
# Dependências Python
tensorflow>=2.10.0
opencv-python
scikit-learn
matplotlib
seaborn
plotly
kaggle
```

### 1. Google Colab (Recomendado)
1. Acesse o [Google Colab](https://colab.research.google.com/)
2. Faça upload do arquivo `pneumonia_detection.ipynb`
3. Execute as células sequencialmente
4. Faça upload do arquivo `kaggle.json` quando solicitado

### 2. Ambiente Local
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/pneumonia-detection.git
cd pneumonia-detection

# Instale as dependências
pip install -r requirements.txt

# Execute o notebook
jupyter notebook pneumonia_detection.ipynb
```

### 3. Configuração do Kaggle
1. Acesse [Kaggle](https://www.kaggle.com/)
2. Vá em "Account" → "Create New API Token"
3. Baixe o arquivo `kaggle.json`
4. Faça upload no Colab quando solicitado

## 📈 Resultados

### Métricas de Performance

| Modelo | Acurácia | Precisão | Recall | F1-Score |
|--------|----------|----------|--------|----------|
| CNN Customizada | ~92% | ~91% | ~94% | ~92% |
| Transfer Learning | ~94% | ~93% | ~95% | ~94% |

### Análise de Erros
- **Falsos Positivos**: Normal classificado como Pneumonia
- **Falsos Negativos**: Pneumonia classificado como Normal
- **Taxa de Erro**: ~6-8%

## 🔧 Técnicas Implementadas

### Pré-processamento
- **Redimensionamento**: 224x224 pixels
- **Normalização**: Valores entre 0 e 1
- **Data Augmentation**:
  - Rotação (±20°)
  - Deslocamento horizontal/vertical (±20%)
  - Zoom (±20%)
  - Flip horizontal
  - Variação de brilho (±20%)

### Otimização
- **Otimizador**: Adam
- **Learning Rate**: 0.001 (com redução automática)
- **Loss Function**: Binary Crossentropy
- **Callbacks**:
  - Early Stopping
  - Reduce Learning Rate on Plateau
  - Model Checkpoint

### Regularização
- **Dropout**: 25% (convolucional), 50% (densa)
- **Batch Normalization**: Após cada camada
- **Data Augmentation**: Para reduzir overfitting

## 📊 Análise Exploratória

### Distribuição das Classes
- **Treino**: 1,341 Normal vs 3,875 Pneumonia
- **Teste**: 234 Normal vs 390 Pneumonia
- **Validação**: 8 Normal vs 8 Pneumonia

### Características das Imagens
- **Formato**: JPEG
- **Canais**: RGB (convertido para escala de cinza)
- **Resolução**: Padronizada para 224x224
- **Contraste**: Variável entre imagens

## 🎯 Aplicações Práticas

### 1. Triagem Médica
- Identificação rápida de casos suspeitos
- Priorização de atendimento
- Redução de tempo de espera

### 2. Apoio Diagnóstico
- Segunda opinião para radiologistas
- Redução de erros humanos
- Padronização de diagnósticos

### 3. Telemedicina
- Análise remota de imagens
- Acesso a especialistas
- Monitoramento de pacientes

## ⚠️ Limitações e Considerações

### Limitações Técnicas
- Necessita de dados de alta qualidade
- Sensível a variações na técnica de imagem
- Requer validação clínica

### Limitações Médicas
- **Não substitui diagnóstico médico**
- Apenas ferramenta de apoio
- Requer interpretação profissional
- Não diferencia tipos de pneumonia

### Considerações Éticas
- Privacidade dos dados médicos
- Responsabilidade médica
- Transparência do algoritmo
- Viés nos dados de treinamento

## 🔬 Melhorias Futuras

### 1. Arquitetura
- Testar ResNet, DenseNet, EfficientNet
- Implementar attention mechanisms
- Usar ensemble de modelos

### 2. Dados
- Coletar mais imagens normais
- Incluir diferentes tipos de pneumonia
- Adicionar metadados clínicos

### 3. Interpretabilidade
- Implementar Grad-CAM
- Visualização de features
- Explicabilidade do modelo

### 4. Validação
- Teste em dados externos
- Validação clínica
- Estudos multicêntricos

## 📚 Referências

1. [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. [Deep Learning for Medical Image Analysis](https://www.nature.com/articles/s41598-019-47306-1)
3. [Transfer Learning in Medical Imaging](https://arxiv.org/abs/1902.07208)
4. [CNN for Medical Image Classification](https://ieeexplore.ieee.org/document/8363574)

## 👥 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📞 Contato

- **Autor**: [Seu Nome]
- **Email**: [seu-email@exemplo.com]
- **LinkedIn**: [linkedin.com/in/seu-perfil]
- **GitHub**: [github.com/seu-usuario]

---

**⚠️ Aviso Médico**: Este projeto é apenas para fins educacionais e de pesquisa. Não deve ser usado para diagnóstico médico sem validação clínica adequada. Sempre consulte um profissional de saúde qualificado. 