from pathlib import Path

import numpy as np
import pandas as pd

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: Juan Camilo 
# Apellido 1: Henao 
# Apellido 2: Espinosa
# Rama: Henao_Espinosa


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data_path = Path(__file__).resolve().parent.parent / "data" / "sales_data.csv"
data = pd.read_csv(data_path)

x = data["advertising"].to_numpy()
y = data["sales"].to_numpy()


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
# Respuesta: Una observación: el término independiente y la inversión en publicidad.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Para representar el intercepto beta_0 en el producto X @ beta.

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: Es 2 x 2, porque X tiene dos columnas: unos y advertising.

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Las ventas estimadas cuando la inversión en publicidad es cero.

# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: El cambio estimado en ventas por cada unidad adicional de publicidad.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Que las predicciones están cerca de los valores reales del dataset.

# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque multiplica cada fila de X por los parámetros beta y produce un valor por observación.

