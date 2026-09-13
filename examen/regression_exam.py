import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Jaiber
# Apellido 1: Obando
# Apellido 2: Lopez
# Rama: obando_lopez


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

#advertising
x =np.array([1, 2, 3, 4, 5,6,7,8])
#sales
y =np.array([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7])


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X =np.column_stack((np.ones(len(x)), x)) 

print("X:")
print(X)

print("Shape X:", X.shape) 
print("Shape y:", y.shape)


# 3. OPERACIONES MATRICIALES
# ------------------------------------------------------------

XtX = X.T @ X
Xty = X.T @ y  


# 4. ESTIMACIÓN DE PARÁMETROS
# ------------------------------------------------------------

beta =np.linalg.inv(X.T @ X) @ X.T @ y

beta_0 = beta[0]
beta_1 = beta[1]

print("Beta 0:", beta_0)
print("Beta 1:", beta_1)


# 5. PREDICCIÓN
# ------------------------------------------------------------

x_new = np.array([1, 9])

prediction =x_new @ beta

print("Prediction:", prediction)


# 6. PREDICCIONES DEL DATASET
# ------------------------------------------------------------

y_pred =X @ beta
print("Predicciones del dataset:")
print(y_pred)


# 7. ERROR
# ------------------------------------------------------------

errors = y - y_pred
error_norm = np.linalg.norm(errors)

print("Error vector:")
print(errors)

print("Error norm:")
print(error_norm)

print("FIN DE EXAMEN")

# 8. PREGUNTAS
# ------------------------------------------------------------

# 1. ¿Qué representa cada fila de X?
# Respuesta: Cada fila representa una observación del problema.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Porque la primera columna de unos, permite estimar el intercepto (beta_0) en el modelo de regresión lineal.



# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: La dimensión de X.T @ X es (n x n), porque en una multiplicación matricial, la dimensión de la matriz resultante es determinada por el número de filas de la primera matriz y el número de columnas de la segunda matriz.   


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: beta_0 representa el intercepto de la línea de regresión, es decir, el valor de y cuando x es igual a cero. En este contexto, indica la predicción de ventas cuando no hay inversión en publicidad.   


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: beta_1 representa la pendiente de la línea de regresión, es decir, el cambio esperado en y por cada unidad de cambio en x. En este contexto, indica cuánto se espera que aumenten las ventas por cada unidad adicional de inversión en publicidad.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: indica que, en conjunto, las predicciones están relativamente cerca de los valores reales


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque la multiplicación matricial permite calcular todas las predicciones de manera eficiente, aplicando el modelo a todas las observaciones al mismo tiempo.