import math as math
import matplotlib.pyplot as plt
import pandas as pd

caminho_arquivo = "C:/Users/Levy/Desktop/sorvetes.xlsx" 
df = pd.read_excel(caminho_arquivo)
df = df.drop_duplicates(subset="Temperature")
df = df.head(24)

datasetx = df["Temperature"].drop_duplicates().tolist() 
datasety = df["Vendas abaixo/acima da média"].tolist()  

def sigmoid(x):
        return 1/(1+math.exp(-x))

def derivada_da_sigmoid(x):
    s_linha = sigmoid(x) * (1 - sigmoid(x))
    return s_linha

def derivadas(datasetx, datasety, a, b, c, d, k, l):
    derivada_e_a = 0
    derivada_e_b = 0
    derivada_e_c = 0
    derivada_e_d = 0
    derivada_e_k = 0
    derivada_e_l = 0
    for i in range(len(datasetx)):
        u = sigmoid(a * datasetx[i] + b)
        w = sigmoid(c * u + d)
        z = sigmoid(k * w + l)
        erro = datasety[i] - z

        # Gradientes
        dz = -2 * erro * derivada_da_sigmoid(k * w + l)  # derivada em relação a z
        dw = dz * k * derivada_da_sigmoid(c * u + d)     # derivada em relação a w
        du = dw * c * derivada_da_sigmoid(a * datasetx[i] + b)  # derivada em relação a u

        # Parâmetros
        derivada_e_a += du * datasetx[i]
        derivada_e_b += du
        derivada_e_c += dw * u
        derivada_e_d += dw
        derivada_e_k += dz * w
        derivada_e_l += dz

    return derivada_e_a, derivada_e_b, derivada_e_c, derivada_e_d, derivada_e_k, derivada_e_l

def distancia(a_posterior, b_posterior, c_posterior, d_posterior, k_posterior, l_posterior, a_anterior, b_anterior, c_anterior, d_anterior, k_anterior, l_anterior):
    return ((a_posterior - a_anterior)**2 + (b_posterior - b_anterior)**2 + (c_posterior - c_anterior)**2 + (d_posterior - d_anterior)**2 + (k_posterior - k_anterior)**2 + (l_posterior - l_anterior)**2)**0.5

def gradiente_descendente(a, b, c, d, k, l, tolerancia, learning_rate):
    a_anterior = a
    b_anterior = b
    c_anterior = c
    d_anterior = d
    k_anterior = k
    l_anterior = l
    a_posterior, b_posterior, c_posterior, d_posterior, k_posterior, l_posterior = [99999999] * 6

    i = 0
    while True:
        derivada = derivadas(datasetx, datasety, a_anterior, b_anterior, c_anterior, d_anterior, k_anterior, l_anterior)
        a_posterior = a_anterior - learning_rate * derivada[0]
        b_posterior = b_anterior - learning_rate * derivada[1]
        c_posterior = c_anterior - learning_rate * derivada[2]
        d_posterior = d_anterior - learning_rate * derivada[3]
        k_posterior = k_anterior - learning_rate * derivada[4]
        l_posterior = l_anterior - learning_rate * derivada[5]
        
        dist = distancia(a_posterior, b_posterior, c_posterior, d_posterior, k_posterior, l_posterior, a_anterior, b_anterior, c_anterior, d_anterior, k_anterior, l_anterior)
        i += 1

        if i % 10000 == 0:
            print(f"iterações: ", i, dist, "coeficientes", a_posterior, b_posterior, c_posterior, d_posterior, k_posterior, l_posterior)

        if dist > tolerancia:
            a_anterior = a_posterior
            b_anterior = b_posterior
            c_anterior = c_posterior
            d_anterior = d_posterior
            k_anterior = k_posterior
            l_anterior = l_posterior
        else:
            break

    return i, a_anterior, b_anterior, c_anterior, d_anterior, k_anterior, l_anterior


iteracoes, a_final, b_final, c_final, d_final, k_final, l_final = gradiente_descendente(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10**(-5), 0.01)
print(f"Número de iterações: {iteracoes}, valor final de A: {a_final}, B: {b_final}, C: {c_final}, D: {d_final}, K: {k_final} e L: {l_final}")

predicoes = []
for x in datasetx:
    u = sigmoid(a_final * x + b_final)
    w = sigmoid(c_final * u + d_final)
    z = sigmoid(k_final * w + l_final)
    predicoes.append(z)

# Plotando os dados reais e a predição final
plt.plot(datasetx, datasety, 'bo', label='Dados reais')
plt.plot(datasetx, predicoes, 'r-', label='Predição (Modelo ajustado)')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ajuste do Modelo usando Gradiente Descendente')
plt.legend()
plt.grid(True)
plt.show()