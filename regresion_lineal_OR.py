import numpy as np

def sigmoid(Theta0, Theta1, Theta2, X):
    z = Theta0 * 1 + Theta1 * X[0] + Theta2 * X[1]
    sig = 1 / (1 + np.exp(-z))
    return sig

def predict(Theta0, Theta1, Theta2, X):
    sig = sigmoid(Theta0, Theta1, Theta2, X)
    return 1 if sig > 0.5 else 0

def maximum_likelihood(Theta0, Theta1, Theta2, X, y):
    n = len(y)
    error_total = 0

    for i in range(n):
        sig = sigmoid(Theta0, Theta1, Theta2, X[i])
        error_total += y[i] * np.log(sig) + (1 - y[i]) * np.log(1 - sig)

    error_promedio = -error_total / n

    return error_promedio

def gradiente_descendiente(X, y):
    dimensions = 3  # Ahora tenemos 3 parámetros (theta0, theta1 y theta2)
    t = np.array([0, 1])
    f_range = np.tile(t, (dimensions, 1))

    max_iter = 5000
    num_agents = 1

    agents = np.zeros((num_agents, dimensions))

    for i in range(dimensions):
        dim_f_range = f_range[i, 1] - f_range[i, 0]
        agents[:, i] = np.random.rand(num_agents) * dim_f_range + f_range[i, 0]

    best_position = np.zeros(dimensions)
    best_fitness = np.inf
    fitness = np.empty(num_agents)

    for i in range(num_agents):
        theta0, theta1, theta2 = agents[i]  # Desempacar los valores en theta0, theta1 y theta2
        fitness[i] = maximum_likelihood(theta0, theta1, theta2, X, y)
        if fitness[i] < best_fitness:
            best_position = agents[i]
            best_fitness = fitness[i]

    # Bucle de optimización
    alpha = 0.005  # Tasa de aprendizaje
    delta = 0.001

    for iteration in range(max_iter):
        # Cálculo del gradiente para theta0, theta1 y theta2
        gradient_theta0 = (maximum_likelihood(best_position[0] + delta, best_position[1], best_position[2], X, y) - maximum_likelihood(best_position[0], best_position[1], best_position[2], X, y)) / delta
        gradient_theta1 = (maximum_likelihood(best_position[0], best_position[1] + delta, best_position[2], X, y) - maximum_likelihood(best_position[0], best_position[1], best_position[2], X, y)) / delta
        gradient_theta2 = (maximum_likelihood(best_position[0], best_position[1], best_position[2] + delta, X, y) - maximum_likelihood(best_position[0], best_position[1], best_position[2], X, y)) / delta

        # Actualización de theta0, theta1 y theta2
        best_position[0] -= alpha * gradient_theta0
        best_position[1] -= alpha * gradient_theta1
        best_position[2] -= alpha * gradient_theta2

        # Cálculo de la nueva aptitud
        best_fitness = maximum_likelihood(best_position[0], best_position[1], best_position[2], X, y)

    print("Mejor solución: Theta0 =", best_position[0], ", Theta1 =", best_position[1], ", Theta2 =", best_position[2])
    print("Mejor valor de aptitud:", best_fitness)

    return best_position

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])


best_position = gradiente_descendiente(X, y)

n = len(X)
for i in range(n):
    decision = predict(best_position[0], best_position[1], best_position[2], X[i])
    print("Instancia:", X[i], "Predicción:", decision)
