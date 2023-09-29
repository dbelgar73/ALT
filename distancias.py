import numpy as np

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    return D[lenX, lenY]

def levenshtein_edicion(x, y, threshold=None):
    # a partir de la versión levenshtein_matriz
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=np.int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1,
                D[i][j - 1] + 1,
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]),
            )
    # COMPLETAR Y REEMPLAZAR ESTA PARTE
    '''
        l : lista de tuplas a devolver.

        Los dos ultimos bucles se recorren en el caso de
        que se choque contra una pared de la matriz
    '''
    l=[]
    while(i > 0 and j > 0):
        m = min(D[i-1][j], D[i][j-1], D[i-1][j-1])
        if m == D[i-1][j-1]:
            l.append((x[i-1],y[j-1]))
            i -= 1
            j -= 1
        elif D[i+1][j] == m:
            l.append(("",y[j-1]))
            i -= 1
        else:
            l.append((x[i-1],""))
            i -= 1
    while(i > 0):
        l.append((x[i-1],""))
        i -= 1
    while(j > 0):
        l.append(("",y[j-1]))
        j-=1
    l.reverse()
    return l

def levenshtein_reduccion(x, y, threshold=None):
    # completar versión con reducción coste espacial
    # Para realizar este algoritmo tomamos dos vectores en vez de toda la matriz
    lenX, lenY = len(x) + 1, len(y) + 1
    row1 = list(range(lenX))
    row2 = [None] * lenX

    for i in range(1, lenY):
        row1, row2 = row2, row1
        row1[0] = i
        for j in range(1, lenX):
            row1[j] = min(
                row1[j - 1] + 1,
                row2[j] + 1,
                row2[j - 1] + (x[j - 1] != y[i - 1])
            )
    return row1[-1]

def levenshtein(x, y, threshold):
    # Considerar el threshold como limite
    lenX, lenY = len(x) + 1, len(y) + 1
    row1 = list(range(lenX))
    row2 = [None] * lenX

    for i in range(1, lenY):
        row1, row2 = row2, row1
        row1[0] = i
        for j in range(1, lenX):
            row1[j] = min(
                row1[j - 1] + 1,
                row2[j] + 1,
                row2[j - 1] + (x[j - 1] != y[i - 1])
            )
        if all(d > threshold for d in row1):
            return threshold + 1
    return min(row1[-1],threshold+1)

def levenshtein_cota_optimista(x, y, threshold):
    #contar nº veces que aparece cada caracter en x (por ejemplo casa-> {c:1 a:2 s:1})
    f1 = lambda l1: dict(map(lambda item: (item, l1.count(item)), l1))
    listaX = f1(x) #caracter con nºde veces que aparece
    #contar nº veces que aparece cada caracter en y (en negativo)
    f2 = lambda l1: dict(map(lambda item: (item, -(l1.count(item))), l1))
    listaY = f2(y)
    #por cada caracter que aparece en las dos cadenas X, Y, calcular la diferencia (p. ejem. {c:1 a:0 s:1 b:-1 d:-1} "a" aparece 2 veces en casa y 2 veces en abad, 2-2=0)
    # Calcular la diferencia en la frecuencia de caracteres entre x e y
    diferencia = {}
    for char in listaX:
        if char in listaY:
            count_x = listaX[char]
            count_y = listaY.get(char, 0)
            diferencia[char] = count_x + count_y       
    for char in listaX:
        if char not in listaY:
            count_x = listaX[char]
            diferencia[char] = count_x       
    for char in listaY:
        if char not in listaX:
            count_y = listaY[char]
            diferencia[char] = count_y    
    # Sumar las frecuencias de caracteres en x e y
    v_pos = sum(count for count in diferencia.values() if count > 0)
    v_neg = sum(count for count in diferencia.values() if count < 0)
    #cota optimista = max(|valores positivos|, |valores negativos|)
    cota_optimista = max(abs(v_pos), abs(v_neg))
    #si cota_optimista > threshold -> devolver threshold+1
    if cota_optimista > threshold:
        return threshold+1
    return levenshtein(x, y, threshold)
    #Para x=casa y=abad -> {'c': 1, 'a': 2, 's': 1}, {'a': -2, 'b': -1, 'd': -1},Salida:: Valor positivo:  2, Valor negativo:  -2, Cota optimista:  2, La distancia de Levenshtein cota optimista entre 'casa' y 'abad' es <= 3

def damerau_restricted_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein restringida con matriz
    lenX, lenY = len(x), len(y)
    # COMPLETAR
    return D[lenX, lenY]

def damerau_restricted_edicion(x, y, threshold=None):
    # partiendo de damerau_restricted_matriz añadir recuperar
    # secuencia de operaciones de edición
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_restricted(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
     return min(0,threshold+1) # COMPLETAR Y REEMPLAZAR ESTA PARTE

def damerau_intermediate_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein intermedia con matriz
    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y, threshold=None):
    # partiendo de matrix_intermediate_damerau añadir recuperar
    # secuencia de operaciones de edición
    # completar versión Damerau-Levenstein intermedia con matriz
    return 0,[] # COMPLETAR Y REEMPLAZAR ESTA PARTE
    
def damerau_intermediate(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    return min(0,threshold+1) # COMPLETAR Y REEMPLAZAR ESTA PARTE

opcionesSpell = {
    'levenshtein_m': levenshtein_matriz,
    'levenshtein_r': levenshtein_reduccion,
    'levenshtein':   levenshtein,
    'levenshtein_o': levenshtein_cota_optimista,
    'damerau_rm':    damerau_restricted_matriz,
    'damerau_r':     damerau_restricted,
    'damerau_im':    damerau_intermediate_matriz,
    'damerau_i':     damerau_intermediate
}

opcionesEdicion = {
    'levenshtein': levenshtein_edicion,
    'damerau_r':   damerau_restricted_edicion,
    'damerau_i':   damerau_intermediate_edicion
}

