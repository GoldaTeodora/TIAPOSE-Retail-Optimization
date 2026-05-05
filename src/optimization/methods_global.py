import numpy as np
from core.profit_calculator import evaluate_solution_global

class OptimizationMethodsGlobal:

    def __init__(self, stores, forecasts, objective='O2', seed=None):
        self.rng = np.random.default_rng(seed)
        self.stores = stores
        self.forecasts = forecasts
        self.objective = objective

        self.n_stores = len(stores)
        self.days = 7

        # dimensão total (21 por loja)
        self.dim = self.n_stores * 21

        # bounds
        self.J_bounds = (0, 20)
        self.X_bounds = (0, 20)
        self.PR_bounds = (0.0, 0.3)

        self.PR_values = np.arange(0, 0.31, 0.05)

        self.low = np.array(
            [self.J_bounds[0]]*(7*self.n_stores) +
            [self.X_bounds[0]]*(7*self.n_stores) +
            [self.PR_bounds[0]]*(7*self.n_stores)
        )

        self.high = np.array(
            [self.J_bounds[1]]*(7*self.n_stores) +
            [self.X_bounds[1]]*(7*self.n_stores) +
            [self.PR_bounds[1]]*(7*self.n_stores)
        )

    def _split_solution(self, solution):
        solution = np.clip(solution, self.low, self.high)

        n = self.n_stores * 7

        J = np.clip(solution[0:n], self.J_bounds[0], self.J_bounds[1])
        X = np.clip(solution[n:2*n], self.X_bounds[0], self.X_bounds[1])
        PR = solution[2*n:3*n]

        # discretizar PR
        PR = self.PR_values[np.abs(self.PR_values[:, None] - PR).argmin(axis=0)]

        return J, X, PR
    

    def _evaluate(self, solution):
        J, X, PR = self._split_solution(solution)

        J_dict = {}
        X_dict = {}
        PR_dict = {}

        idx = 0

        for store in self.stores:
            J_dict[store] = J[idx:idx+7]
            X_dict[store] = X[idx:idx+7]
            PR_dict[store] = PR[idx:idx+7]
            idx += 7

        result = evaluate_solution_global(
            J_dict,
            X_dict,
            PR_dict,
            self.forecasts,
            self.objective
        )

        value = result['best_value']

        if not np.isfinite(value):
               return -1e10

        return value

            

    def random_search(self, n_iter=100):
        best_solution = None
        best_value = -np.inf
        history = []

        for _ in range(n_iter):
            solution = self.rng.uniform(self.low, self.high)

            # 🔥 reduzir escala inicial
            n = self.n_stores * 7
            solution[0:n] *= 0.3        # J
            solution[n:2*n] *= 0.3      # X
            value = self._evaluate(solution)

            if value > best_value:
                best_value = value
                best_solution = solution

            history.append(best_value)

        return {
           'solution': best_solution,
           'best_value': best_value,
           'history': history
        }
    

    def hill_climbing(self, max_iter=300):
        # tentar encontrar solução inicial válida
        for _ in range(100):
            solution = self.rng.uniform(self.low, self.high)

            n = self.n_stores * 7
            solution[0:n] *= 0.3
            solution[n:2*n] *= 0.3
            value = self._evaluate(solution)
            # aceitar solução inicial razoável
            if value > -10000:
                break

        best_value = self._evaluate(solution)
        best_solution = solution.copy() 

        history = [best_value]

        for _ in range(max_iter):

            neighbor = solution + self.rng.normal(0, 0.2, self.dim)
            neighbor = np.clip(neighbor, self.low, self.high)

            value = self._evaluate(neighbor)

            if value > best_value:
                best_value = value
                best_solution = neighbor
                solution = neighbor

            history.append(best_value)

        return {
            'solution': best_solution,
            'best_value': best_value,
            'history': history
        }
    
    def genetic_algorithm(self, population_size=30, generations=50):

        population = self.rng.uniform(self.low, self.high, size=(population_size, self.dim))

        # 🔥 inicialização mais segura
        n = self.n_stores * 7
        population[:, 0:n] *= 0.3
        population[:, n:2*n] *= 0.3
 
        history = []

        for _ in range(generations):

            fitness = np.array([self._evaluate(ind) for ind in population])

            best_idx = np.argmax(fitness)
            best_value = fitness[best_idx]
            history.append(best_value)

            # 🔁 seleção (torneio)
            def select():
                idx = self.rng.choice(population_size, 3, replace=False)
                return population[idx[np.argmax(fitness[idx])]].copy()

            new_population = [population[best_idx].copy()]  # elitismo

            while len(new_population) < population_size:

               p1 = select()
               p2 = select()

               alpha = self.rng.random(self.dim)
               child = alpha * p1 + (1 - alpha) * p2

               # mutação
               if self.rng.random() < 0.2:
                   mutation = self.rng.normal(0, 0.2, self.dim)
                   child += mutation

               child = np.clip(child, self.low, self.high)

               new_population.append(child)

        population = np.array(new_population[:population_size])

        best_idx = np.argmax([self._evaluate(ind) for ind in population])
        best_solution = population[best_idx]

        return {
           'solution': best_solution,
           'best_value': self._evaluate(best_solution),
           'history': history
        }
    


    def particle_swarm(self, n_particles=30, max_iter=50):

        particles = self.rng.uniform(self.low, self.high, size=(n_particles, self.dim))
        velocities = self.rng.uniform(-1, 1, size=(n_particles, self.dim))

        # 🔥 inicialização mais controlada
        n = self.n_stores * 7
        particles[:, 0:n] *= 0.3
        particles[:, n:2*n] *= 0.3

        personal_best = particles.copy()
        personal_best_values = np.array([self._evaluate(p) for p in particles])

        best_idx = np.argmax(personal_best_values)
        global_best = particles[best_idx].copy()
        global_best_value = personal_best_values[best_idx]

        history = [global_best_value]

        # parâmetros PSO
        w = 0.7
        c1 = 1.5
        c2 = 1.5

        for _ in range(max_iter):

            for i in range(n_particles):

                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (personal_best[i] - particles[i])
                    + c2 * r2 * (global_best - particles[i])
                )

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.low, self.high)

                value = self._evaluate(particles[i])

                if value > personal_best_values[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_values[i] = value

                    if value > global_best_value:
                        global_best = particles[i].copy()
                        global_best_value = value

            history.append(global_best_value)

        return {
            'solution': global_best,
            'best_value': global_best_value,
            'history': history
        }
    

    def simulated_annealing(self, max_iter=300, temp_init=100):

        solution = self.rng.uniform(self.low, self.high)

        n = self.n_stores * 7
        solution[0:n] *= 0.3
        solution[n:2*n] *= 0.3

        best_solution = solution.copy()
        best_value = self._evaluate(solution)
        current_value = best_value

        history = [best_value]

        for i in range(max_iter):

            temp = temp_init * (0.95 ** i)

            neighbor = solution + self.rng.normal(0, 0.2, self.dim)
            neighbor = np.clip(neighbor, self.low, self.high)

            value = self._evaluate(neighbor)

            delta = value - current_value

            if delta > 0 or self.rng.random() < np.exp(delta / (temp + 1e-10)):
                solution = neighbor
                current_value = value

            if current_value > best_value:
                best_solution = solution.copy()
                best_value = current_value

            history.append(best_value)

        return {
            'solution': best_solution,
            'best_value': best_value,
            'history': history
        }
    

    def differential_evolution(self, pop_size=30, generations=50):

        population = self.rng.uniform(self.low, self.high, size=(pop_size, self.dim))

        n = self.n_stores * 7
        population[:, 0:n] *= 0.3
        population[:, n:2*n] *= 0.3

        history = []

        for _ in range(generations):

            new_population = []

            for i in range(pop_size):

                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[self.rng.choice(idxs, 3, replace=False)]

                mutant = a + 0.8 * (b - c)
                mutant = np.clip(mutant, self.low, self.high)

                cross = self.rng.random(self.dim) < 0.7
                trial = np.where(cross, mutant, population[i])
                trial = np.clip(trial, self.low, self.high)

                if self._evaluate(trial) > self._evaluate(population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            best_value = max(self._evaluate(ind) for ind in population)
            history.append(best_value)

        best_idx = np.argmax([self._evaluate(ind) for ind in population])

        return {
            'solution': population[best_idx],
            'best_value': self._evaluate(population[best_idx]),
            'history': history
        }
    
        
    

    def optimize(self, method):

        if method == 'random':
            result = self.random_search()

        elif method == 'hill_climbing':
            result = self.hill_climbing()

        elif method == 'simulated_annealing':
            result = self.simulated_annealing()

        elif method == 'genetic':
            result = self.genetic_algorithm()

        elif method == 'pso':
            result = self.particle_swarm()

        elif method == 'de':
            result = self.differential_evolution()

        else:
            raise ValueError(f"Método desconhecido: {method}")

        return {
            'method': method,
            'solution': result['solution'],
            'best_value': result['best_value'],
            'history': result['history']
        }