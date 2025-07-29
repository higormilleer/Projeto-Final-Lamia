#!/usr/bin/env python3
"""
Script de Demonstração - Detecção de Pneumonia em Raios-X
Este script permite fazer predições em novas imagens usando o modelo treinado.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import argparse

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Carrega e pré-processa uma imagem para predição
    
    Args:
        image_path (str): Caminho para a imagem
        target_size (tuple): Tamanho desejado (largura, altura)
    
    Returns:
        numpy.ndarray: Imagem pré-processada
    """
    try:
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
        
        # Converter BGR para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        img = cv2.resize(img, target_size)
        
        # Normalizar
        img = img.astype(np.float32) / 255.0
        
        # Adicionar dimensão do batch
        img = np.expand_dims(img, axis=0)
        
        return img
    
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
        return None

def predict_pneumonia(model, image_path, threshold=0.5):
    """
    Faz predição de pneumonia em uma imagem
    
    Args:
        model: Modelo treinado
        image_path (str): Caminho para a imagem
        threshold (float): Limiar para classificação
    
    Returns:
        dict: Resultados da predição
    """
    # Pré-processar imagem
    img = load_and_preprocess_image(image_path)
    if img is None:
        return None
    
    # Fazer predição
    prediction = model.predict(img)[0][0]
    
    # Classificar
    is_pneumonia = prediction > threshold
    confidence = prediction if is_pneumonia else 1 - prediction
    
    # Determinar classe
    if is_pneumonia:
        class_name = "PNEUMONIA"
        probability = prediction
    else:
        class_name = "NORMAL"
        probability = 1 - prediction
    
    return {
        'class': class_name,
        'probability': probability,
        'confidence': confidence,
        'raw_prediction': prediction,
        'image_path': image_path
    }

def visualize_prediction(image_path, result):
    """
    Visualiza a predição com a imagem
    
    Args:
        image_path (str): Caminho para a imagem
        result (dict): Resultado da predição
    """
    # Carregar imagem original
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Criar figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Imagem original
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Imagem Original')
    ax1.axis('off')
    
    # Resultado da predição
    colors = {'NORMAL': 'green', 'PNEUMONIA': 'red'}
    color = colors[result['class']]
    
    ax2.bar(['NORMAL', 'PNEUMONIA'], 
            [1 - result['raw_prediction'], result['raw_prediction']],
            color=['green', 'red'], alpha=0.7)
    ax2.set_title(f'Predição: {result["class"]} ({result["probability"]:.2%})')
    ax2.set_ylabel('Probabilidade')
    ax2.set_ylim(0, 1)
    
    # Adicionar valores nas barras
    for i, v in enumerate([1 - result['raw_prediction'], result['raw_prediction']]):
        ax2.text(i, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def batch_predict(model, image_folder, threshold=0.5):
    """
    Faz predições em lote para todas as imagens em uma pasta
    
    Args:
        model: Modelo treinado
        image_folder (str): Pasta com imagens
        threshold (float): Limiar para classificação
    
    Returns:
        list: Lista de resultados
    """
    results = []
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for filename in os.listdir(image_folder):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            image_path = os.path.join(image_folder, filename)
            result = predict_pneumonia(model, image_path, threshold)
            if result:
                results.append(result)
    
    return results

def print_results(results):
    """
    Imprime resultados de forma organizada
    
    Args:
        results (list): Lista de resultados
    """
    print("\n" + "="*60)
    print("RESULTADOS DAS PREDIÇÕES")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {os.path.basename(result['image_path'])}")
        print(f"   Classe: {result['class']}")
        print(f"   Probabilidade: {result['probability']:.2%}")
        print(f"   Confiança: {result['confidence']:.2%}")
        print(f"   Predição bruta: {result['raw_prediction']:.4f}")
    
    # Estatísticas
    if results:
        normal_count = sum(1 for r in results if r['class'] == 'NORMAL')
        pneumonia_count = sum(1 for r in results if r['class'] == 'PNEUMONIA')
        
        print(f"\n{'='*60}")
        print("ESTATÍSTICAS")
        print(f"{'='*60}")
        print(f"Total de imagens: {len(results)}")
        print(f"Normal: {normal_count} ({normal_count/len(results)*100:.1f}%)")
        print(f"Pneumonia: {pneumonia_count} ({pneumonia_count/len(results)*100:.1f}%)")
        print(f"Confiança média: {np.mean([r['confidence'] for r in results]):.2%}")

def main():
    """
    Função principal
    """
    parser = argparse.ArgumentParser(description='Detecção de Pneumonia em Raios-X')
    parser.add_argument('--model', type=str, default='pneumonia_detection_model.h5',
                       help='Caminho para o modelo treinado')
    parser.add_argument('--image', type=str, help='Caminho para uma imagem específica')
    parser.add_argument('--folder', type=str, help='Pasta com imagens para predição em lote')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Limiar para classificação (padrão: 0.5)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualizar predições (apenas para imagem única)')
    
    args = parser.parse_args()
    
    # Verificar se o modelo existe
    if not os.path.exists(args.model):
        print(f"Erro: Modelo não encontrado em {args.model}")
        print("Certifique-se de que o modelo foi treinado e salvo.")
        return
    
    # Carregar modelo
    print(f"Carregando modelo: {args.model}")
    try:
        model = keras.models.load_model(args.model)
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return
    
    # Fazer predições
    if args.image:
        # Predição única
        print(f"\nAnalisando imagem: {args.image}")
        result = predict_pneumonia(model, args.image, args.threshold)
        
        if result:
            print(f"\nResultado: {result['class']}")
            print(f"Probabilidade: {result['probability']:.2%}")
            print(f"Confiança: {result['confidence']:.2%}")
            
            if args.visualize:
                visualize_prediction(args.image, result)
    
    elif args.folder:
        # Predição em lote
        print(f"\nAnalisando pasta: {args.folder}")
        results = batch_predict(model, args.folder, args.threshold)
        print_results(results)
    
    else:
        print("Erro: Especifique --image ou --folder")
        parser.print_help()

if __name__ == "__main__":
    main() 