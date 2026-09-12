import numpy as np
import pandas as pd

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre:Juan Jose
# Apellido 1: Melo
# Apellido 2: Montenegro
# Rama: melo_montenegro


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = pd.read_csv("../data/sales_data.csv")

x = data["advertising"]
y = data["sales"]


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
B, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
beta = B

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
# Respuesta:Cada fila representa una observacion del dataset. Contiene el termino independiente 1 y el valor de advertising.



# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: La columna de unos permite incluir el intercepto beta_0 en el modelo lineal.



# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: La dimensión de X.T @ X es 2 x 2 porque X tiene 2 columnas y X.T tiene dimensión 2 x 8. Al multiplicar una matriz de dimensiones (2 x 8) por una matriz de dimensiones (8 x 2), el resultado es una matriz de dimensiones (2 x 2).


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: beta_0 es el intercepto: las ventas estimadas cuando advertising vale 0.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: beta_1 es la pendiente: el cambio estimado en sales por cada unidad adicional de advertising.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que las predicciones están cerca de los valores reales y el ajuste del modelo es bueno.  


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Cada fila de X representa una observacion y el producto con beta calcula su prediccion; por eso devuelve todas en un solo vector.

