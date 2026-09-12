import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Alfonso Marino 
# Apellido 1: Alfonso 
# Apellido 2: Lopez
# Rama: alfonso_lopez


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = None

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7])


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

beta = np.linalg.solve(XtX, Xty)

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
# Respuesta: Cada fila representa una observación del problema, es decir, un registro de publicidad invertida y su correspondiente resultado en ventas. La primera columna (el 1) corresponde al término del intercepto, y la segunda al valor de publicidad de esa observación.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Porque permite incorporar el término independiente (β₀) dentro del producto matricial Xβ. Sin esa columna de unos, el modelo estaría obligado a pasar por el origen (β₀ = 0), lo cual limitaría su capacidad de ajustarse a los datos reales.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: Es una matriz 2×2. Esto se debe a que X tiene dimensión 8×2, por lo que su transpuesta Xᵀ tiene dimensión 2×8. Al multiplicar (2×8)(8×2), las dimensiones internas coinciden (8=8) y el resultado toma las dimensiones externas: 2×2.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Es el intercepto del modelo: representa el valor estimado de ventas cuando la inversión en publicidad es 0.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Es la pendiente del modelo: indica cuánto aumenta en promedio las ventas por cada unidad adicional invertida en publicidad.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que, en conjunto, las predicciones del modelo están muy cerca de los valores reales de ventas observados, es decir, el modelo se ajusta bien a los datos.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque el producto matricial aplica la misma combinación lineal (β₀ + β₁x) a cada fila de X al mismo tiempo, evitando tener que calcular cada predicción por separado. Esto se conoce como vectorización.