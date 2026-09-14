import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Juan David
# Apellido 1: Velasco
# Apellido 2: Otero
# Rama: velasco_otero

import numpy as np

# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.genfromtxt('../data/sales_data.csv', delimiter=',', skip_header=1)

x = data [:,[0]]
y = data [:,[1]]

# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

print("X:")
print(X)

print("Shape X:", X)
print("Shape y:", y)


# 3. OPERACIONES MATRICIALES
# ------------------------------------------------------------

Xt=X.T

XtX = Xt @ X
Xty = Xt @ y


# 4. ESTIMACIÓN DE PARÁMETROS
# ------------------------------------------------------------

beta = np.linalg.inv(XtX) @ Xty
print(beta)

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

print("Predictions:")
print(y_pred)

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
#Cada fila representa una observación o muestra individual, 
#compuesta por un 1 asociado al intercepto y el valor correspondiente de la variable independiente x.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:
#Para poder incluir el parámetro beta_0 en el cálculo matricial de la ecuación de regresión

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:
#Es de 2X2. Al multiplicar X.T (dimensión 2X8) por X (dimensión 8X2), el resultado es una matriz cuadrada
#cuyo tamaño coincide con la cantidad de parámetros a estimar

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:
#Representa el intercepto con el eje y, es decir, el valor estimado de y cuando la variable x es igual a cero.

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:
#Representa la pendiente de la línea de regresión, es decir, el cambio estimado en y por cada unidad de cambio en x.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:
#Significa que la diferencia total entre los valores reales (y) y las predicciones (y_pred) es muy baja,
#lo que demuestra un buen ajuste del modelo lineal a los datos.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:
#Porque al multiplicar la matriz de diseño X por el vector de parámetros beta, se realiza una combinación lineal
#de todas las observaciones, generando así un vector de predicciones para cada muestra en el dataset.