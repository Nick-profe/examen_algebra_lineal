import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Jimmy 
# Apellido 1: Lopez
# Apellido 2: Yule
# Rama: lopez_yule


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = A = np.array([[1, 3.2], [2, 4.8], [3, 7.3], [4, 8.7], [5, 11.1], [6, 12.8], [7, 15.2], [8, 16.7]])


x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7])


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(x.shape[0]), x))

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

prediction = X @ beta

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
# Respuesta: Es una represatación de cada observación en el conjunto de datos, donde la primera columna es un término constante (1) y la segunda columna es el valor de la variable independiente x.
# Ejemplo [1, 8] 


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Porque la columna de X tiene uno es para el intercepto y que no cambie el valor cuando se realice la multiplicacion de matrices


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: La dimensión de X.T @ X es (2, 2) porque X tiene 8 filas y 2 columnas, entonces al multiplicar la transpuesta de X (2, 8) por X (8, 2) obtenemos una matriz de 2x2.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: beta_0 representa el valor esperado de y cuando es x=0


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: beta_1 se representa cuánto cambia y cuando 0 aumenta una unidad.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Se indica que en el conjunto, las predicciones están relativamente cerca de los valores reales.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque X @ beta es una operación matricial que aplica la transformación lineal definida por beta a toda la matriz X, resultando en un vector con todas las predicciones de todas las observaciones.
