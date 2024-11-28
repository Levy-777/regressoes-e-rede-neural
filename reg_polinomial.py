from time import time
import matplotlib.pyplot as plt

dataset = [ (i/10, i**2) for i in range(0,8) ]

x = [i[0] for i in dataset]
y = [i[1] for i in dataset]


def distn( p0, pi ):
    return sum( (pi[i] - p0[i]) ** 2 for i in range(len(p0)) ) ** (1/2)


def gdds(datax, datay, coeficientes = [1]):
    
    n = len(coeficientes)
    coeficientes_grad = [0] * n


    for row in range(len(datax)):
        dx = datax[row]
        for cof in range(n):
            erro = 0
            for k in range(n):
                erro += coeficientes[k] * dx ** (n-k-1)
            
            coeficientes_grad[cof] += 2 * ( datay[row] - erro ) * ( - dx ** (n-cof-1) )
                

    return coeficientes_grad


avgt = 0
def regressao(datax, datay, coeficientes = [1,1], tolerancia = 5e-4, learning_rate = 1e-2):

    n = len(datax)
    nc = len(coeficientes)

    coeficientes0 = [1000] * nc
        
    i = 1
    d = distn( coeficientes0, coeficientes )

    while d > tolerancia:
        coeficientes0 = coeficientes + []

        g = gdds(datax, datay, coeficientes0)
        for j in range(nc):
            coeficientes[j] = coeficientes0[j] - learning_rate * g[j] / n


        d0 = d
        d = distn( coeficientes0, coeficientes )
        i += 1

        if i % 1000 == 0:
            t1 = time()
            print(i, coeficientes)
            print()
            dt = t1-t0
            avg =+ dt
            print(f"time: {round(dt, 2)}\tdist:{d}\n\n")
            print(f"avg t: {round(avg/(i/1000), 2)}\tddist:{d0-d}")
            print(" + ".join( [ str(round(coeficientes[i],3)) + "x^{" + str(nc-i-1) + "}" for i in range(nc) ] ))
            print()

        
    return (i, coeficientes)


cbase = [-0.30831219578928937, -0.14036823000007118, 0.004189785667469635, 0.11699870773053547, 0.18944842543942308, 0.21377052419611475, 0.18492328727072424, 0.10364861573629165, -0.018862052464872433, -0.1543914642965764, -0.2497933971872057, -0.2219480279617965, 0.026570014907499988, 0.5036558291105587, 0.7294046416323973, -0.9656742949149422, 0.9995186835514759]


t0 = time()
c = regressao( x, y, cbase, 5e-4)
t1 = time()

print( c )
print()

print( " + ".join( [ str(round(c[1][i],3)) + "x^{" + str(len(c[1])-i-1) + "}" for i in range(len(c[1])) ] ) )
print()

print(f"time: {round(t1-t0, 2)}s")


plt.plot(x, y, 'bo', label='Dados reais')
plt.plot(x, y, 'r-', label='Predição (Modelo ajustado)')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ajuste do Modelo usando Gradiente Descendente')
plt.legend()
plt.grid(True)
plt.show()