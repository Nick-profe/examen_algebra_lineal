import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Andres Pinilla
# Apellido 1: Pinilla 
# Apellido 2: Victoria
# Rama: pinilla_victoria


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.genfromtxt("../data/sales_data.csv", delimiter=",", skip_header=1)

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
# Respuesta: Cada fila corresponde a un registro del dataset, es decir, un
# valor específico de lo invertido en publicidad. El primer número de cada
# fila siempre es 1, y ese es el que permite que el intercepto entre en
# la cuenta cuando se hace la multiplicación de matrices.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Es un truco matemático para que beta_0 quede incluido dentro
# del producto matricial. Sin esa columna, no habría manera de sumar el
# intercepto, porque toda la operación quedaría multiplicando solo por x.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: Da como resultado una matriz 2x2. Esto pasa porque X.T tiene
# forma 2xn y X tiene forma nx2, y al multiplicar matrices las columnas de
# la primera tienen que coincidir con las filas de la segunda (ahí se
# cancela la n), dejando solo el 2x2 como resultado final.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Sería el punto de partida del modelo, básicamente cuánto se
# esperaría vender si no se invirtiera nada en publicidad.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Indica el impacto de la publicidad, o sea, por cada unidad
# extra que se invierte, cuánto sube en promedio la venta esperada.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Quiere decir que el modelo ajusta bien, que la diferencia
# entre lo que predijo y lo que realmente pasó es mínima en general.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque con una sola operación matricial se aplica la misma
# fórmula a todas las filas al mismo tiempo, en lugar de tener que hacer
# un cálculo por separado para cada observación. Eso es justamente lo que
# se conoce como vectorización.