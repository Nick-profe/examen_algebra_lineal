import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre:
# Apellido 1: Duque
# Apellido 2: Saavedra
# Rama: duque_saavedra


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = 

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7])


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

print("X:")
print(X)

print("Shape X:", x.shape )
print("Shape y:", y.shape)


# 3. OPERACIONES MATRICIALES
# ------------------------------------------------------------

XtX = X.T @ X
Xty = X.T @ y
print(XtX)
print(XtX.shape)

print(Xty)
print(Xty.shape)


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
# Respuesta: Cada fila de X representa la cantidad de anuncios respectivo a las ventas 


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: X contiene una primera columna de unos porque a la hora de hacer el producto matricial se tiene en cuenta el intercepto, si no se tiene en cuenta los unos se borraria el intercepto 


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: La dimension de X.T @ X es de 2x2 ya que al hacer el producto de las dos matrices se tiene en cuenta el tamaño, las dimensiones externas determinan el tamaño de la matriz resultante


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: beta_0 representa el intercepto del modelo, en este caso seria la cantidad estimada de anuncios cuando las ventas son iguales a 0


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: beta_1 representa la pendiente del modelo, es decir indica el valor de las ventas cuando varia la cantidad de anuncios


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que los valores que predijo el modelo estan cerca de los valores observados inicialemnte 


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque el producto matricial permite obtener las predicciones de todas las observaciones al mismo tiempo.