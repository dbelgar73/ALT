import numpy as np

def levenshtein_matriz(x, y, threshold=None):
    # esta versión no utiliza threshold, se pone porque se puede
    # invocar con él, en cuyo caso se ignora
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
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
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
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
        Tres bucles con los que se añaden las tuplas a l,
        lista que represneta la secuencia de ecición. Esta
        se ha de voltear para que esté bien ordenada.
    '''
    l=[]
    while(i > 0 and j > 0):
        m = min(D[i-1][j], D[i][j-1], D[i-1][j-1])
        if m == D[i-1][j-1]:
            l.append((x[i-1],y[j-1]))
            i -= 1
            j -= 1
        elif D[i-1][j] == m:
            l.append((x[i-1],""))
            i -= 1
        else:
            l.append(("",y[j-1]))
            j -= 1
    while(i > 0):
        l.append((x[i-1],""))
        i -= 1
    while(j > 0):
        l.append(("",y[j-1]))
        j-=1
    l.reverse()
    return D[lenX][lenY],l

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
    f=lambda l1, l2: dict(map(lambda item: (item, l1.count(item) - l2.count(item)), set(l1+l2)))#se obtienen los caracteres y el nºveces que aparecen, para y se cuenta en negativo y se suman las listas
    l = f(x,y) #se crea una lista con la diferenciprinta de caracteres entre x y como se indica =>Ejemplo x=casa y=abad ->{c:1 a:0 s:1 b:-1 d:-1}
    #print(l)
    pos, neg= 0,0 #se cuentan los valores para calcular la cota
    for v in l.values(): #recorre la lista p.ej. {c:1 a:0 s:1 b:-1 d:-1} y suma los positivos en pos y los negativos en neg
        if v > 0:
            pos+=v 
        else:
            neg+=v
    cotaOptimista = max(abs(neg), pos) #calculo de la cota optimista
    if cotaOptimista > threshold: return threshold+1
    return levenshtein(x, y, threshold)
    #Para x=casa y=abad -> {'c': 1, 'a': 2, 's': 1}, {'a': -2, 'b': -1, 'd': -1},Salida:: Valor positivo:  2, Valor negativo:  -2, Cota optimista:  2 (<thrshld 3), 
    #La distancia de Levenshtein cota optimista entre 'casa' y 'abad' es <= 3

def damerau_restricted_matriz(x, y, threshold=None):
    # completar versión Damerau-Levenstein restringida con matriz
    lenX, lenY = len(x), len(y)
    # COMPLETAR
    '''
        Palabra x: horizontal
        Palabra y: vertical
    '''

    if lenX < 2 or lenY < 2:
        return levenshtein_reduccion(x,y)

    m = np.zeros((lenY+1,lenX+1),int)
    for i in range(1,lenX+1):
        m[0][i]=i
    for j in range(1,lenY+1):
        m[j][0]=j

    
    for j in range(1,lenY+1):
        for i in range(1,lenX+1):
            m[j][i]=min(
                m[j-1][i]+1,
                m[j][i-1]+1,
                m[j-1][i-1]+(x[i-1]!=y[j-1])
            )
            if j > 1 and i > 1:
                if x[i-1]==y[j-2] and x[i-2]==y[j-1]:
                    d = m[j-2][i-2]+1
                    if d <= m[j][i]:
                        m[j][i] = d

    return m[lenY, lenX]

def damerau_restricted_edicion(x, y, threshold=None):
    # partiendo de damerau_restricted_matriz añadir recuperar
    # secuencia de operaciones de edición
    # COMPLETAR Y REEMPLAZAR ESTA PARTE
    lenX, lenY = len(x), len(y)

    if lenX < 2 or lenY < 2:
        return levenshtein_edicion(x,y)

    '''
        Palabra x: horizontal
        Palabra y: vertical
    '''
    m = np.zeros((lenY+1,lenX+1),int)
    for i in range(1,lenX+1):
        m[0][i]=i
    for j in range(1,lenY+1):
        m[j][0]=j
    
    for j in range(1,lenY+1):
        for i in range(1,lenX+1):
            m[j][i]=min(
                m[j-1][i]+1,
                m[j][i-1]+1,
                m[j-1][i-1]+(x[i-1]!=y[j-1])
            )
            if j > 1 and i > 1:
                if x[i-1]==y[j-2] and x[i-2]==y[j-1]:
                    d = m[j-2][i-2]+1
                    if d <= m[j][i]:
                        m[j][i] = d
    #print(m)
    l=[]
    while(i > 0 and j > 0):
        s = min(m[j-1][i],m[j][i-1],m[j-1][i-1])
        if i > 1 and j > 1 and (x[i-2]==y[j-1] and x[i-1]==y[j-2]) and m[j-2][i-2]<=s:
            l.append((x[i-2]+x[i-1],x[i-1]+x[i-2]))
            i -= 2
            j -= 2
        elif s == m[j-1][i-1]:
            l.append((x[i-1],y[j-1]))
            i -= 1
            j -= 1
        elif s == m[j-1][i]:
            l.append(("",y[j-1]))
            j -= 1
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
    
    return m[lenY][lenX],l

def damerau_restricted(x, y, threshold=None):
    # versión con reducción coste espacial y parada por threshold
    # COMPLETAR Y REEMPLAZAR ESTA PARTE
    lenX,lenY=len(x),len(y)
    
    if lenX < 2 or lenY < 2:
        return levenshtein_reduccion(x,y)
    
    row0=list(range(lenX+1))
    row1=[0]*(lenX+1)
    row2 = row1.copy()
    row1[0]=1

    c=0

    for i in range(1,lenX+1):
        row1[i]=min(row1[i-1]+1,row0[i]+1,row0[i-1]+(x[i-1]!=y[0]))
    
    if min(row1)>threshold: c+=1

    for j in range(2,lenY+1):
        row2[0]=j
        for i in range(1,lenX+1):
            row2[i]=min(row2[i-1]+1,row1[i]+1,row1[i-1]+(x[i-1]!=y[j-1]))
            if i>1 and x[i-2]==y[j-1] and x[i-1]==y[j-2]:
                if 1+row0[i-2]<row2[i]:
                    row2[i]=1+row0[i-2]
        row0=row1
        row1=row2.copy()

        if min(row2)>threshold: 
            c+=1
            if c > 2:
                return threshold+1
        else:
            c = 0

    if row2[lenX]>threshold: return threshold+1

    return row2[lenX]

def damerau_intermediate_matriz(x, y, threshold=None):
    # versión Damerau-Levenshtein intermedia con matriz
    lenX, lenY = len(x), len(y)
    # inicialización de la matriz de resultados intermedios a zeros
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1 # primera fila de Levenshtein 
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1 # primera columna de Levenshtein
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1, # borrado
                D[i][j - 1] + 1, # inserción
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]), # sustitución
                # Damerau
                # intercambio (versión restringida)
                D[i-2][j-2]+1 if (i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]) else float('inf'),
                # intercambio (version intermedia)
                D[i-2][j-3]+2 if (i > 1 and j > 2 and x[i - 1] == y[j - 3] and x[i - 2] == y[j - 1]) else float('inf'),
                # intercambio (versión intermedia)
                D[i-3][j-2]+2 if (i > 2 and j > 1 and x[i - 1] == y[j - 2] and x[i - 3] == y[j - 1]) else float('inf')  
            )
    return D[lenX, lenY]

def damerau_intermediate_edicion(x, y, threshold=None):
    # versión Damerau-Levenshtein intermedia con matriz
    # con recuperación de secuencia de operaciones de edición
    lenX, lenY = len(x), len(y)
    D = np.zeros((lenX + 1, lenY + 1), dtype=int)
    for i in range(1, lenX + 1):
        D[i][0] = D[i - 1][0] + 1
    for j in range(1, lenY + 1):
        D[0][j] = D[0][j - 1] + 1
        for i in range(1, lenX + 1):
            D[i][j] = min(
                D[i - 1][j] + 1, # borrado
                D[i][j - 1] + 1, # inserción
                D[i - 1][j - 1] + (x[i - 1] != y[j - 1]), # sustitución
                # Damerau
                # intercambio (versión restringida)
                D[i-2][j-2]+1 if (i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1]) else float('inf'),
                # intercambio (version intermedia)
                D[i-2][j-3]+2 if (i > 1 and j > 2 and x[i - 1] == y[j - 3] and x[i - 2] == y[j - 1]) else float('inf'),
                # intercambio (versión intermedia)
                D[i-3][j-2]+2 if (i > 2 and j > 1 and x[i - 1] == y[j - 2] and x[i - 3] == y[j - 1]) else float('inf')  
            )

    # recuperación del camino
    # se parte desde la última posición de la matriz de resultados intermedios y se recorre en sentido inverso
    i, j = lenX, lenY
    l=[] # lista del camino
    while(i > 0 and j > 0):
        m = min(D[i-1][j], D[i][j-1], D[i-1][j-1]) 
        
        # caso intercambio (version intermedia)
        if i > 1 and j > 2 and x[i - 1] == y[j - 3] and x[i - 2] == y[j - 1] and D[i-2][j-3]+2 <= m:
            l.append((x[i - 2:i], y[j - 3:j]))
            # retrocedo en D
            i -= 2
            j -= 3
        # caso intercambio (version intermedia)
        elif i > 2 and j > 1 and x[i - 1] == y[j - 2] and x[i - 3] == y[j - 1] and D[i-3][j-2]+2 <= m:
            l.append((x[i - 3:i], y[j - 2:j]))
            i -= 3
            j -= 2
        # caso intercambio (version restringida)
        elif i > 1 and j > 1 and x[i - 1] == y[j - 2] and x[i - 2] == y[j - 1] and D[i-2][j-2]+1 <= m:
            l.append((x[i - 2:i], y[j - 2:j]))
            i -= 2
            j -= 2
        # caso sustitución
        elif m == D[i-1][j-1]:
                l.append((x[i-1], y[j-1]))
                i -= 1
                j -= 1
        # caso inserción
        elif m == D[i][j-1]:
            l.append(("", y[j-1]))
            j -= 1
        # caso borrado
        elif m == D[i-1][j]:
            l.append((x[i-1], ""))
            i -= 1
                    
    # caso en el que aún queden simbolos de x (borrados)
    while i > 0:
        l.append((x[i-1], ""))
        i -= 1
    # caso en el que aún queden simbolos de y (inserciones)
    while j > 0:
        l.append(("", y[j-1]))
        j -= 1

    l.reverse() # para dar el camino ordenado en sentido ascendente
    return D[lenX][lenY], l
    
def damerau_intermediate(x, y, threshold=None):
    # versión Damerau-Levenshtein con reducción coste espacial y parada por threshold
    lenX, lenY = len(x), len(y)
    # ventana de los únicos 3 vectores que se necesitan para realizar las comprobaciones
    row1 = list(range(lenX + 1))
    row2 = [0] * (lenX + 1)
    row3 = [0] * (lenX + 1)

    for i in range(1, lenY + 1):
        rowActual = [0] * (lenX + 1)
        rowActual[0] = i
        for j in range(1, lenX + 1):
            rowActual[j] = min(
                rowActual[j - 1] + 1, # borrado
                row1[j] + 1, # inserción
                row1[j - 1] + (x[j - 1] != y[i - 1]), # sustitución
                # Damerau
                # intercambio (versión restringida)
                row2[j - 2] + 1 if (i > 1 and j > 1 and x[j - 2] == y[i - 1] and x[j - 1] == y[i - 2]) else float('inf'),
                # intercambio (versión intermedia)
                row2[j - 3] + 2 if (i > 1 and j > 2 and x[j - 1] == y[i - 2] and x[j - 3] == y[i - 1]) else float('inf'),
                # intercambio (versión intermedia)
                row3[j - 2] + 2 if (i > 2 and j > 1 and x[j - 2] == y[i - 1] and x[j - 1] == y[i - 3]) else float('inf')
            )
        # reasignación de los vectores para la próxima iteración    
        row3 = row2.copy()
        row2 = row1.copy()
        row1 = rowActual.copy()
        
        if threshold is not None and all(d > threshold for d in rowActual):
            return threshold + 1
    return min(rowActual[-1], threshold + 1)

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

