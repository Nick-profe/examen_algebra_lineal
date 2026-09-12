import numpy as np
 
# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================
 
# Nombre: Juan camilo
# Apellido 1: Joya
# Apellido 2: Duarte
# Rama: joya_duarte
 
 
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
    [8, 16.7],
])
 
x = data[:, 0]
y = data[:, 1]
 
 
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
# Respuesta: Cada fila representa una observación (un pedido). La primera
# columna es siempre 1 (para el término independiente beta_0) y la segunda
# columna es el numero de productos de ese pedido (la variable x).
 
 
# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Para poder incluir el intercepto beta_0 en el modelo. Al
# multiplicar esa columna de unos por beta_0, se suma una constante a cada
# prediccion; sin ella, la recta estaria forzada a pasar por el origen.
 
 
# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: Es de 2x2. X tiene forma (n, 2) -n filas (observaciones) y 2
# columnas (unos y x)-, entonces X.T tiene forma (2, n), y el producto
# (2, n) @ (n, 2) da como resultado una matriz (2, 2).
 
 
# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Es el intercepto: el tiempo base de preparacion estimado
# cuando el numero de productos es 0 (el punto donde la recta cruza el eje y).
 
 
# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Es la pendiente: cuantos minutos adicionales se necesitan,
# en promedio, por cada producto extra que se agrega al pedido.
 
 
# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Que las predicciones del modelo estan, en conjunto, muy cerca
# de los valores reales observados; es decir, el ajuste lineal describe
# bien la relacion entre numero de productos y tiempo de preparacion.
 
 
# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque cada fila de X ya contiene [1, x_i] para cada
# observacion i. Al multiplicar X (n, 2) por beta (2,), numpy calcula
# beta_0*1 + beta_1*x_i para cada fila al mismo tiempo, produciendo un
# vector con las n predicciones de una sola operacion matricial.