import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Sebastian 
# Apellido 1: Giraldo
# Apellido 2: Acosta
# Rama: giraldo_acosta


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = np.loadtxt("data/sales_data.csv", delimiter=",", skiprows=1)

x = data[:, 0]   # columna "advertising"
y = data[:, 1]   # columna "sales"


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

y_pred =  X @ beta


# 7. ERROR
# ------------------------------------------------------------

errors =  y - y_pred
error_norm = np.linalg.norm(errors)

print("Error vector:")
print(errors)

print("Error norm:")
print(error_norm)


# 8. PREGUNTAS
# ------------------------------------------------------------

# 1. ¿Qué representa cada fila de X?
# Respuesta: Cada fila es un dato de nuestra tabla: 
# un 1 fijo (que usamos como "truco" matemático) y el número de inversión en publicidad 
# de ese mes o registro.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta:Es un truco para que la fórmula funcione bien. 
# Ese 1 hace que Beta_0 (el punto de partida del modelo) se sume solo, sin depender 
# de ningún dato, en cada predicción.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: : Es una matriz de 2x2. Es 2x2 porque X tiene solo 2 columnas 
# (el 1 y la publicidad), y cuando multiplicas una matriz por su transpuesta de esta forma, 
# el resultado siempre queda del tamaño "número de columnas x número de columnas".


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: : Es el punto de partida: 
# cuántas ventas tendríamos si no invirtiéramos nada en publicidad.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: : Es cuánto suben las ventas por cada peso (o unidad) extra 
# que se invierte en publicidad.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Significa que el modelo está prediciendo casi lo mismo que pasó en la realidad,
#  es decir, que se equivoca poco.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta:Porque en lugar de calcular la predicción dato por dato con una
#  fórmula repetida muchas veces, esta multiplicación hace todos los cálculos
#  de una sola vez para toda la tabla.