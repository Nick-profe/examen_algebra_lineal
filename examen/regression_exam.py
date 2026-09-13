import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre:
# Apellido 1: Rivas 
# Apellido 2: Jimenez
# Rama: Rivas_Jimenez


# 1. CARGA DE DATOS
# ------------------------------------------------------------

#data =

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7])


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

print("X:")
print(X)

print("Shape X:", )
print("Shape y:", )


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
# Respuesta:
# Es el numero de publicidades.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:
# Para incluir el término de intercepto en el modelo de regresión.

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:
# Debido a que la dimension de X.T es (2, 8) y la dimension de X es (8, 2), el producto de ambas es una matriz de dimension (2, 2).

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:
# Representa el valor de y cuando no hay publicidad, osea x = 0.

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:
# Representa la pendiente de la regresion, osea que tanto cambia las ventas segun el aumento de la publicidad.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:
# significa que tanta distancia hay entre la predicion y el valor real, osea que a menor valor mejor predicción.

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:
