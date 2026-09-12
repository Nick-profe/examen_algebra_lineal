import numpy as np

# EXAMEN 1
# Álgebra lineal aplicada a regresión lineal
# ============================================================

# Nombre: juana gabriela 
# Apellido 1: lopez
# Apellido 2: trejos
# Rama: lopez_trejos


# 1. CARGA DE DATOS
# ------------------------------------------------------------

#data = load_data(sales_data.csv)

x = np.array([1, 2, 3, 4, 5, 6, 7, 8]) #advertising/publicidad
y = np.array ([3.2, 4.8, 7.3, 8.7, 11.1, 12.8, 15.2, 16.7]) #sales/ventas


# 2. MATRIZ DE DISEÑO
# ------------------------------------------------------------

X = np.column_stack((np.ones(len(x)), x))

print("X:")
print(X)

print("Shape X:", )
print("Shape y:", )


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

x_new = np.array([1, 9]) #nueva predicción cuando x (advertising)= 9

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
# Respuesta:Corresponde a las observaciones, en este caso a la cantidad de advertisment/publicidad


# 2. ¿Por qué X contiene una primera columna de unos?
# Respuesta: Para lograr incluir el intercepto en las operaciones que se van a realizar


# 3. ¿Cuál es la dimensión de X.T @ X y por qué?
# Respuesta: X.T (2,8) @  X(8,2) = (2,2)  --> (8X2)(2X8) --> La dimensión es 2,2 porque tomamos 
# las filas de transpuesta y las columnas de la matriz original, en este caso 2 filas y 2 columnas

# 4. ¿Qué representa beta_0 dentro de este problema?
# Respuesta: en este caso, beta_0  es el intercepto del modelo y hace referencia al valor estimado (ventas = y) cuando
# la cantidad de publicidad (x) es igual a cero, basicamante, el valor de ventas cuando no
# se tiene publicidad


# 5. ¿Qué representa beta_1 dentro de este problema?
# Respuesta: en este caso beta_1 es la pendiente del modelo y hace referencia al valor de venta, 
# el cual varía o se ve afectado por la cantidad de publicidad que se genere


# 6. ¿Qué significa que la norma del error sea pequeña?
# Respuesta: la norma del error hace referencia a qué tan cercano está el valor
# que predice el modelo (y_pred) frente al valor real (y). Cuando este valor es pequeño
# o  cercano a cero, significa que la predicción realizada es buena


# 7. ¿Por qué X @ beta permite obtener todas las
#    predicciones simultáneamente?
# Respuesta: porque es una alternativa (se le conoce como vectorización) que nos permite generar todas
# las predicciones en lugar de generar predición por predicción