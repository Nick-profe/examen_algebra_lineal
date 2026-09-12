import numpy as np
import pandas as pd
# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre:jeison 
# Apellido 1:navarro
# Apellido 2:murillo
# Rama:navarro_murillo


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = pd.read_csv("../data/sales_data.csv")

x = data["advertising"].values
y = data["sales"].values


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

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

beta = np.linalg.inv(X.T @ X) @ X.T @ y

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

y_pred = y_pred = X @ beta


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
# Respuesta: cada fila pertenece a un registro o cada pedido en el dataset el cual tiene 2 columnas
# una para el valor del intercepto y la otra para el valor de la publicidad


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: lo que ocurre es que dentro del producto matricial tenemos que poder operar 
# Bo porque si no la formula no estaria completa


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: la dimension es 2x2 ya que tenemos a X. T que tiene forma de (2x8)
# y X tiene la forma de(8x2) de esta forma coinciden los internos 


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: beta_0 vendria siendo un numero constante que representa al intercepto del modelo
# o tambien podriamos decir que es el valor estimado de ventas cuando la 
# publicidad "x" es igual a 0


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:# Respuesta: beta_1 es la pendiente del modelo, que representa la tasa de cambio: 
# indica cuánto aumentan o disminuyen  en promedio las ventas por cada unidad 
# adicional de publicidad invertida.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Que la norma del error sea pequeña significa que, en conjunto, las 
# predicciones del modelo (y_pred) están muy cerca de los valores reales observados (y). 
# y nos demuestra que el modelo se ajusta bien a los datos


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:# cada fila de X representa una observación distinta y al multiplicar la matriz X por 
# el vector beta, la operación combina automáticamente cada fila de X con beta, 
# generando una predicción por cada observación en una sola operación. Esto evita 
# tener que calcular cada predicción por separado (una por una con un ciclo), y se 
# conoce como vectorización.