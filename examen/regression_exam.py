import numpy as np
import pandas as pd

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre:Edwin Andres  
# Apellido 1:Guerrero 
# Apellido 2:Diaz
# Rama:guerrero_diaz


# 1. CARGA DE DATOS
# ------------------------------------------------------------

data = pd.read_csv("data/sales_data.csv")

x = np.array(data.advertising)
y = np.array(data.sales)


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)),x))

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

print("beta:", beta) 

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
# Respuesta: Cada fila de X representa una observación. En ella hay 2 valores:
# el primero siempre será un 1, el cual hace referencia al intercepto, y el otro
# es una cantidad de publicidad.

# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Porque en la fórmula ventas = beta[0] + beta[1] * publicidad,
# el uno multiplica a beta[0], y multiplicar por 1 no cambia su valor.
# Si el uno desaparece, beta[0] no aportaría nada a la fórmula: con
# publicidad = 0, el modelo predeciría 0 ventas, y la recta tendría
# que pasar por el punto (0, 0). Por ende, el uno es necesario para
# incluir beta[0] (el intercepto) en el modelo.

# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta:La dimension es (2,2) debido a las dimensiones de afuera
#determinan el resultado y la multiplicacion es (2,8)(8,2)


# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: Teoricamente beta[0] significa el valor esperado de y(ventas) cuando
# x(publicidad) es 0, en una oracion es reprensenta la cantidad de ventas cuando la
# la publicidad es 0


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: Este representa cuanto cambia y, cuando aumenta x, en este caso ccuanto
# cambia las ventas cuando aumenta la publicidad


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: Cada error es la diferencia entre el valor real y el valor de la predicion,
# que la norma sea pequeña significa que las predicciones están conn un valor muy cercana
# de los valores reales, es decir, que el modelo preedice bien los datos.


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: Porque cada fila de X es una observación, y al multiplicar X @ beta
# cada fila se multiplica con beta: 1 · beta_0 + publicidad · beta_1, que es
# la prediccion de esa observación. Como X tiene 8 filas, se obtienen 8
# resultados en una sola operación. A esto se le llama vetorizaxion.