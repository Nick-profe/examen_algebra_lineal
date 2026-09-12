import numpy as np
import pandas as pd


# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Juan Sebastian 
# Apellido 1: Galindez 
# Apellido 2: Franco
# Rama: Galindez_Franco


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = pd.read_csv("/mnt/c/WINDOWS/system32/parcial1/examen_algebra_lineal/data/sales_data.csv")

x = np.array(data["advertising"])
y = np.array(data["sales"])


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)),x))

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

# la primera fila representa una fila de unos que para el modelo de ecuacion lineal 
# son muy ultiles para dejar intacta la ecuacion lineal y la segunda fila representa los datos de publicidad que se tienen en el dataset

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:
# La primera columna de unos en X representa el término independiente (intercepto) del modelo de regresión lineal.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:
# la dimensión de X.T @ X es (2, 2) porque X tiene 2 columnas, y al multiplicar la transpuesta de X (que tiene
# 2 filas) por X (que tiene 2 columnas), se obtiene una matriz de dimensión (2, 2).


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:
# beta_0 representa el intercepto de la línea de regresión 

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:
# beta_1 representa la pendiente de la línea de regresión, es decir, el cambio esperado en las ventas por cada unidad adicional de publicidad.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:
# significa que las predicciones del modelo sean buenas, muy cercanas a la medido


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:
# porque al multiplicar la matriz de diseño X por el vector de parámetros beta, se obtiene un vector de predicciones
# que contiene las predicciones para todos los datos de entrada en el dataset