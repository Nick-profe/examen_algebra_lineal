import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Sebastián 
# Apellido 1: Moreno
# Apellido 2: Dorado
# Rama: moreno_dorado


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.loadtxt("data/sales_data.csv", delimiter=",", skiprows=1) 

x = data[:, 0]
y = data[:, 1]


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

print("X:")
print(X)
print("Shape X:", X.shape)

print("y:")
print(y)
print("Shape y:", y.shape)


# 3. OPERACIONES MATRICIALES
# ------------------------------------------------------------

XtX = X.T @ X
print("XtX:")
print(XtX)
print("Shape XtX:", XtX.shape)

Xty = X.T @ y
print("Xty:")
print(Xty)
print("Shape Xty:", Xty.shape)


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
# Respuesta:

# Cada fila representa un "advertising" o publicidad del problema. 
# Por ejemplo, la primera fila representa el primer dato de publicidad
# Que tiene un valor en ventas de 3.2.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:

# Porque se quiere incluir el término independiente (intercepto) en el modelo de regresión lineal
# La primera columna de unos permite que el modelo pueda ajustar un valor constante (beta_0) que 
# Representa el punto donde la línea de regresión cruza el eje y cuando x es igual a cero.

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:

# La dimensión de X.T @ X es (2, 2) porque X tiene 8 filas, 2 columnas, las dimensiones internas (8,8) coinciden
# Y las dimensiones externas (2,2) definen la forma de la matriz resultante.

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:

# Es el intercepto del modelo que representa el valor esperado de y cuando x es igual a cero. 
# En este caso, beta_0 indica la cantidad de ventas esperadas cuando no se realiza ninguna publicidad.

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:

# Es la pendiente del modelo que representa el cambio esperado en y por cada unidad adicional de x.
# En este caso, beta_1 indica cuánto se espera que aumenten las ventas por cada unidad adicional de publicidad.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:

# Una norma del error pequeña indica que las predicciones del modelo están muy cerca de los valores reales observados.

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:

# Porque la multiplicación de la matriz X por el vector beta permite calcular todas las predicciones de manera eficiente
# En una sola operación, aprovechando las propiedades de la multiplicación matricial.
