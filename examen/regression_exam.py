import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: David Alberto
# Apellido 1: Vergara
# Apellido 2: Tabares
# Rama: vergara_tabares


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.array([
    [1, 3.2],
    [2, 4.8],
    [3, 7.3],
    [4, 8.7],
    [5, 11.1],
    [6, 12.8],
    [7, 15.2],
    [8, 16.7]
])

x = np.array([1,2,3,4,5,6,7,8])
y = np.array([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7])


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

print("X:", X)

print("Shape X:", X.shape)
print("Shape y:", y.shape)


# 3. OPERACIONES MATRICIALES
# ------------------------------------------------------------

XtX = X.T @ X
Xty = X.T @ y
print(XtX)
print("Shape XtX:", XtX.shape)
print(Xty)
print("Shape Xty:", Xty.shape)
    


# 4. ESTIMACIÓN DE PARÁMETROS
# ------------------------------------------------------------

beta = np.linalg.inv(XtX) @ Xty

beta_0 = beta[0]
beta_1 = beta[1]

print("Beta 0:", beta_0)
print("Beta 1:", beta_1)


# 5. PREDICCIÓN
# ------------------------------------------------------------

x_new = np.array([1, 9])

prediction = x_new @ beta

print("Prediction:", prediction)


# 6. PREDICCIONES DEL DATASET
# ------------------------------------------------------------

y_pred = X @ beta

print("y_pred:", y_pred)


# 7. ERROR
# ------------------------------------------------------------

errors = y - y_pred
error_norm = np.linalg.norm(errors)

print("Error vector:")
print(errors)

print("Error norm:")
print(error_norm)


# 8. PREGUNTAS
# ------------------------------------------------------------

# 1. ¿Qué representa cada fila de X?
# Respuesta: Cada fila de X representa un dato del conjunto,
# donde la primera columna es el término de intercepto y la segunda columna sería
# el valor de la variable independiente correspondiente


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: La primera columna de unos en X hace que el producto matricial del modelo no esté forzado a pasar por el origen,
#  permitiendo así que el modelo tenga un intercepto (beta_0) que se ajuste a los datos
# 

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: La dimension de X.T @ X es (2, 2). Esto porque X tiene dimensiones (8, 2),
#  y al multiplicar la transpuesta de X, que tiene dimensiones (2, 8), por X, obtenemos
#  una matriz cuadrada de tamaño igual al número de columnas de X, que es 2. Es decir, para este caso,  
# Al multiplicar (2×8)(8×2), las dimensiones internas coinciden (8=8) 
# y el resultado siempre serán las dimensiones externas: (2×2) 


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: beta_0 es el intercepto del modelo. Representa el valor esperado de las
# ventas cuando la inversión en publicidad, es decir, el advertising es igual a 0 en este
# caso, beta_0 es aproximado a 1.114, es decir, el modelo estima que, sin inversión en publicidad (0),
# las ventas serían aproximadamente 1.11 unidades


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:  beta_1 seria la pendiente del modelo. Representa cuánto cambia, en promedio,
# el valor de las ventas por cada unidad adicional que aumenta la inversión en
# advertising. En este caso, beta_1 es 1.969, lo que significa que por cada
# unidad adicional invertida en publicidad, el modelo estima un incremento promedio de
# aproximadamente 1.97 unidades en las ventas


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Cuando la norma del error es pequeña significa que las predicciones del modelo
#  están muy cerca de los valores reales observados

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque el producto matricial aplica la combinación lineal β₀ + β₁x a 
# cada fila de X al mismo tiempo, gracias a las reglas de la multiplicación de matrices.
# Entonces esto evita tener que calcular cada predicción por separado, ya que el resultado de X @ beta ya es un vector con las 8
# predicciones en el mismo orden que las observaciones originales.