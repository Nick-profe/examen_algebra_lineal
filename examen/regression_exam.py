import numpy as np
from pathlib import Path

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre:
# Apellido 1: Ruiz
# Apellido 2: Calero
# Rama: ruiz_calero


# 1. CARGA DE DATOS
# ------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"

data = np.genfromtxt(DATA_PATH, delimiter=",", skip_header=1)

x = data[:, 0]
y = data[:, 1]


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

# Ecuación normal: beta = (X^T X)^(-1) X^T y
# Se resuelve el sistema en lugar de invertir explícitamente,
# por ser numéricamente más estable.
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
# Respuesta: Cada fila es una observación del dataset. Contiene un 1 en la
# primera posición (asociado al intercepto) y el valor de advertising de
# esa observación en la segunda.


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Para que el intercepto beta_0 quede incluido dentro del
# producto matricial X @ beta. Sin esa columna el modelo sería
# y = beta_1 * x, una recta forzada a pasar por el origen. La columna de
# unos convierte el término constante en un coeficiente más del vector beta.


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: Es 2x2. X tiene dimensión 8x2, por lo que X.T es 2x8; al
# multiplicar (2x8) @ (8x2) las dimensiones internas se cancelan y queda
# 2x2. En general, si X es n x p, entonces X.T @ X es p x p: siempre
# cuadrada, con tamano igual al numero de parametros del modelo, lo que
# permite resolver el sistema de la ecuacion normal.


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Es el valor estimado de sales cuando advertising es cero, es
# decir el nivel base de ventas sin inversión publicitaria. Corresponde al
# punto de corte de la recta con el eje vertical.


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Es la pendiente de la recta: el cambio estimado en sales por
# cada unidad adicional de advertising. Al ser positivo, indica que más
# inversión publicitaria se asocia con mayores ventas.


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: La norma del error mide la distancia euclidiana entre el
# vector de valores reales y el de valores predichos. Que sea pequeña
# significa que los residuos son pequeños y que la recta estimada se
# ajusta bien a los datos observados. El método de mínimos cuadrados elige
# justamente el beta que minimiza esta norma.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque el producto matricial aplica el mismo vector beta a
# cada fila de X mediante un producto punto. La fila i produce
# beta_0 * 1 + beta_1 * x_i, que es exactamente la predicción de la
# observación i. Así, una sola operación matricial reemplaza el bucle
# sobre las n observaciones y devuelve el vector completo de predicciones.
