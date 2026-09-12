import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Leonardo
# Apellido 1: Monsalve
# Apellido 2: Gomez
# Rama: Monsalve_Gomez


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = {
    "advertising": [1, 2, 3, 4, 5, 6, 7, 8],
    "sales": [3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7]
}

x = np.array(data["advertising"])
y = np.array(data["sales"])


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
beta0 = beta[0]
beta1 = beta[1]

print("Beta 0:", beta0)
print("Beta 1:", beta1)


# 5. PREDICCIÓN
# ------------------------------------------------------------

xnew = np.array([1, 9])

prediction = xnew @ beta

print("Prediction:", prediction)


# 6. PREDICCIONES DEL DATASET
# ------------------------------------------------------------

ypred = X @ beta

# 7. ERROR
# ------------------------------------------------------------

errors = y-ypred
error_norm =np.linalg.norm(errors)

print("Error vector:")
print(errors)

print("Error norm:")
print(error_norm)


# 8. PREGUNTAS
# ------------------------------------------------------------

# 1. ¿Qué representa cada fila de X?
# Respuesta:
"""
Cada fila es una observación: un nivel de inversión en publicidad 
junto con el 1 que permite calcular el intercepto.

"""


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:
"""
Para que beta_0 se pueda sumar en el modelo. Sin esa columna,
el modelo no tendría intercepto y siempre pasaría por el origen.

"""
# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:
"""
Es (2, 2). X es (8, 2) y X.T es (2, 8); al multiplicarlas,
el 8 interno se cancela y quedan los dos números externos: 2 y 2.

"""
# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta:

"""
Es el intercepto. Representa las ventas estimadas cuando la inversión en publicidad es 0.

"""


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta:

"""
Es la pendiente. Indica cuánto aumentan las ventas por cada unidad adicional que se invierte en publicidad.

"""

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta:

"""
Que las predicciones del modelo están muy cerca de los valores reales, 
es decir, el modelo ajusta bien los datos.

"""

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: