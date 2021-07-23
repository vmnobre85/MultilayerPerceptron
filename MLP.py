# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:58:10 2021

@author: Victor Nobre
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from activation_function import sigmoid
from activation_function import sigmoidDerivada

#REDE NEURAL PARA REALIZAR O TREINAMENTO.

#ENCONTRANDO ARQUIVOS DATASETS PARA TREINAMENTO.
datasets = pd.read_csv('datasets/treinamento.csv.csv')

#DETERMINANDO VALORES DE ENTRADAS E SAÍDAS.
Valoresentradas = datasets.iloc[:,:4].values
entradas = Valoresentradas
Valoressaidas = datasets.iloc[:,4:7].values
saidas = np.empty([130, 3], dtype=int)
for i in range(130):
    saidas[i] = Valoressaidas[i]

#DETERMINANDO PESOS.
Pesosiniciais = 2*np.random.random((4,3)) - 1
pesos0 = Pesosiniciais
Pesoscamadaoculta = 2*np.random.random((3,3)) - 1
pesos1 = Pesoscamadaoculta

#DETERMINANDO TAXAS DE APRENDIZAGEM E PRECISÃO.
taxaAprendizagem = 0.1
precisão = 1

#DETERMINANDO QUANTIDADE DE ÉPOCAS E ERRO MÉDIO PARA SIMULAÇÃO.
epocas = int(input('Digite a quantidade de épocas desejada para realizar os treinamentos: '))
mediatotal = 0

#FUNÇÃO DE TREINAMENTO.
for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = mediatotal
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
        
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
        
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * precisão) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * precisão) + (pesosNovo0 * taxaAprendizagem)
    if j == epocas-1:
        print("O PERCENTUAL DE ERRO FOI DE: %" + str(mediaAbsoluta))  
        print("FORAM REALIZADAS {} ÉPOCAS" .format(j+1))
        
#PLOTANDO RESULTADOS.
        x = erroCamadaSaida[:10,0:1]
        y = erroCamadaSaida[:10,1:2]
        z = y - x
        plt.plot(x,'bo', y,'go',z,'r--' )
        plt.show()
 

#REDE NEURAL PARA REALIZAR OS TESTES.

#ENCONTRANDO ARQUIVOS DATASETS PARA TESTES.
datasets = pd.read_csv('datasets/teste.csv.csv')

#DETERMINANDO VALORES DE ENTRADAS E SAÍDAS.
Valoresentradas = datasets.iloc[:,:4].values
entradas = Valoresentradas
Valoressaidas = datasets.iloc[:,4:7].values
saidas = np.empty([18, 3], dtype=int)
for i in range(18):
    saidas[i] = Valoressaidas[i]

#DETERMINANDO PESOS.
Pesosiniciais = 2*np.random.random((4,3)) - 1
pesos0 = Pesosiniciais
Pesoscamadaoculta = 2*np.random.random((3,3)) - 1
pesos1 = Pesoscamadaoculta

#DETERMINANDO TAXAS DE APRENDIZAGEM E PRECISÃO.
taxaAprendizagem = 0.1
precisão = 1

#DETERMINANDO QUANTIDADE DE ÉPOCAS PARA SIMULAÇÃO.
epocas = 10000

#FUNÇÃO DE TREINAMENTO.
for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
        
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
        
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * precisão) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * precisão) + (pesosNovo0 * taxaAprendizagem)

#MATRIZ FINAL DO RESULTADO APÓS TESTES.
print("O RESULTADO FINAL APÓS {} ÉPOCAS É DE: " .format(epocas))
print(deltaSaida)
print('ONDE PARA CADA RESULTADO NEGATIVO SERÁ CONSIDERADO 0 E PARA CADA RESULTADO POSITIVO SERÁ CONSIDERADO 1')
 



