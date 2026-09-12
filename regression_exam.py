import numpy as np

# EXAMEN 1: REGRESIÓN LINEAL
# Estudiante: Wilson Navia Valencia
# Rama: navia_valencia

# 1. CARGA DE DATOS
# x: número de productos en el pedido
# y: tiempo de preparación en minutos
x = np.array([2, 4, 6, 8, 10], dtype=float)
y = np.array([5.0, 7.1, 9.2, 10.8, 13.1], dtype=float)

print("Datos cargados correctamente.")
print("x:", x)
print("y:", y)
# ============================================================

# 1. CARGA Y DEFINICIÓN DE DATOS
# x: Productos por pedido | y: Tiempo de preparación (minutos)
x = np.array([2, 4, 6, 8, 10], dtype=float)
y = np.array([5.0, 7.1, 9.2, 10.8, 13.1], dtype=float)

# 2. CONSTRUCCIÓN DE LA MATRIZ DE DISEÑO X
# Agregamos una columna de unos a la izquierda para el intercepto (beta_0)
X = np.column_stack((np.ones(len(x)), x))

# 3. CÁLCULO DE PRODUCTOS MATRICIALES
XTX = X.T @ X
XTy = X.T @ y

# 4. ESTIMACIÓN DE PARÁMETROS (Ecuación Normal: beta = (X^T X)^-1 X^T y)
beta_hat = np.linalg.inv(XTX) @ XTy
beta_0 = beta_hat[0]
beta_1 = beta_hat[1]

# 5. PREDICCIONES Y CÁLCULO DE ERRORES
y_hat = X @ beta_hat
errores = y - y_hat
norma_error = np.linalg.norm(errores)

# 6. PREDICCIÓN ESPECÍFICA PARA UN PEDIDO CON 12 PRODUCTOS
x_nuevo = np.array([1.0, 12.0])
prediccion_12 = x_nuevo @ beta_hat

# IMPRESIÓN DE RESULTADOS EN CONSOLA
print("=== RESULTADOS DEL EXAMEN ===")
print("Matriz X (Diseño):\n", X)
print("\nMatriz X^T X:\n", XTX)
print("\nVector X^T y:\n", XTy)
print(f"\nParámetros Estimados:")
print(f"  - Beta_0 (Intercepto): {beta_0:.4f}")
print(f"  - Beta_1 (Pendiente) : {beta_1:.4f}")
print("\nPredicciones (y_hat):\n", np.round(y_hat, 4))
print("\nVector de Errores (e):\n", np.round(errores, 4))
print(f"\nNorma Euclidiana del Error (||e||): {norma_error:.4f}")
print(f"\nTiempo estimado para 12 productos: {prediccion_12:.2f} minutos")

# ============================================================
# RESPUESTAS A LAS PREGUNTAS TEÓRICAS DEL EXAMEN
# ============================================================
# 1. ¿Qué representa cada fila de X?
# Resp: Cada fila representa una observación (un pedido) con el término de intercepto y el número de productos.

# 2. ¿Por qué X contiene una primera columna de unos?
# Resp: Para incluir el término constante (beta_0) dentro de la multiplicación matricial.

# 3. ¿Cuál es la dimensión de X^T X y por qué?
# Resp: Es de dimensión 2x2, resultando de multiplicar X^T (2x5) por X (5x2), correspondiente al número de parámetros.

# 4. ¿Qué representa beta_0 en este problema?
# Resp: El tiempo base o fijo necesario para procesar un pedido sin importar el número de productos.

# 5. ¿Qué representa beta_1 en este problema?
# Resp: El tiempo adicional estimado que toma procesar cada producto individual agregado.

# 6. ¿Qué significa que la norma del error sea pequeña?
# Resp: Indica que los valores calculados por el modelo son muy cercanos a los tiempos reales observados.

# 7. ¿Por qué X @ beta permite obtener todas las predicciones simultáneamente?
# Resp: Porque vectoriza el cálculo realizando el producto punto entre cada fila de X y el vector beta en una sola operación.