import math as math
import matplotlib.pyplot as plt
import pandas as pd


caminho_arquivo = "C:/Users/Levy/Desktop/sorvetes.xlsx" 
df = pd.read_excel(caminho_arquivo)
df = df.drop_duplicates(subset="Temperature")
df = df.head(24)
""" Tratamento do dataset para ser utilizado: Usando o excel, a temperatura foi ordenada em ordem crescente e foi calculada a média entre as vendas de sorvete, logo após foi adicionada 
uma coluna condicional utilizando a média para classificar as vendas que estavam maior ou igual a média como '1' caso contrário = 0, assim foi possivel ter uma classificação binária para regressão logistica.
Dataset utilizado: https://www.kaggle.com/datasets/sakshisatre/ice-cream-sales-dataset """

def grad_a(a, b, x, y):
    z = a*x + b 
    sigmoid_z = sigmoid(z)
    return (sigmoid_z - y) * x

def grad_b(a, b, x, y):
    z =  a*x + b
    sigmoid_z = sigmoid(z)
    return (sigmoid_z - y) 


def sigmoid(x):
    return 1/(1+math.exp(-x))


def calcular_acuracia(datasetx, datasety, a, b):
    acertos = 0
    for i in range(len(datasetx)):
        x = datasetx[i]
        y_real = datasety[i]
        z = a*x + b
        y_pred = 1 if sigmoid(z) >= 0.7 else 0  
        if y_pred == y_real:
            acertos += 1
    return acertos / len(datasetx)


def gradDS(datasetx, datasety, a, b):
    grad_a_sum = 0
    grad_b_sum = 0
    

    for i in range(len(datasetx)):
        grad_a_sum += grad_a(a, b, datasetx[i], datasety[i])
        grad_b_sum += grad_b(a, b, datasetx[i], datasety[i])


    return grad_a_sum, grad_b_sum

def dist2(a_0, b_0, a_n, b_n):  
    return ((a_n - a_0)**2 + (b_n - b_0)**2)**0.5

def calcular_f1_score(datasetx, datasety, a, b):
    ver = 0  # Verdadeiros Positivos
    falP = 0  # Falsos Positivos
    faln = 0  # Falsos Negativos

    # Calculando os valores de TP, FP, FN
    for i in range(len(datasetx)):
        x = datasetx[i]
        y_real = datasety[i]
        z = a * x + b
        y_pred = 1 if sigmoid(z) >= 0.7 else 0  

        if y_pred == 1 and y_real == 1:
            ver += 1  # Verdadeiro Positivo
        elif y_pred == 1 and y_real == 0:
            falP += 1  # Falso Positivo
        elif y_pred == 0 and y_real == 1:
            faln += 1  # Falso Negativo

    # Calculando precisão e recall
    precisao = ver / (ver + falP) if (ver + falP) > 0 else 0
    recall = ver / (ver + faln) if (ver + faln) > 0 else 0

    # Calculando o F1-Score
    if precisao + recall > 0:
        f1_score = 2 * (precisao * recall) / (precisao + recall)
    else:
        f1_score = 0

    return f1_score

def gradienteDescendente(a_0, b_0, tol, lr):

    datasetx = df["Temperature"].drop_duplicates().tolist() 
    datasety = df["Vendas abaixo/acima da média"].tolist()  
    
    
    a_n, b_n  = a_0, b_0 

    a_n1, b_n1,  = [99999999] * 2
    
    i = 0

    
    while True:
        # Gradientes
        grad_a, grad_b  = gradDS(datasetx, datasety, a_n, b_n)
        
        # Atualizar coeficientes
        a_n1 = a_n - lr * grad_a
        b_n1 = b_n - lr * grad_b
        
        
        i += 1
        
        acuracia = calcular_acuracia(datasetx, datasety, a_n, b_n)
        f1 = calcular_f1_score(datasetx, datasety, a_n, b_n)
        err = dist2(a_n, b_n, a_n1, b_n1)

        if i%10000 == 0:
            print(f"Iteração {i}: Acurácia = {acuracia:.2%}, F1-Score = {f1:.2f}")
            print(f"dist: {err}")
            print (a_n, b_n)

        if err > tol:
            a_n, b_n = a_n1, b_n1 
        else:
            break

    y_vals = [sigmoid(a_n * x + b_n) for x in datasetx]

    plt.figure(figsize=(12, 8))  # Aumentar o tamanho do gráfico
    plt.scatter(datasetx, datasety, color='red', label='Dados Reais')  # Pontos reais
    plt.plot(datasetx, y_vals, color='blue', label='Curva de Decisão (Sigmoid)')  # Curva

    plt.xlim(min(datasetx) - 5, max(datasetx) + 5)  
    plt.ylim(-0.1, 1.1)  

    plt.xlabel('Dataset X')
    plt.ylabel('Dataset Y (Valores Reais e Previsões)')
    plt.title('Curva de Decisão - Regressão Logística')
    plt.legend()

    # Mostrar o gráfico
    plt.tight_layout()
    plt.show()

    return i, a_n, b_n, acuracia, f1

print(gradienteDescendente(0.1,0.1,1e-5,1e-1))