
import numpy as np
from core.profit_calculator import calculate_daily_profit, calculate_weekly_profit, evaluate_solution_global
from core.config import OPTIMIZATION

class OptimizationMethodsGlobal:

    def __init__(self, stores, forecasts, objective='O2', seed=None):
        self.rng = np.random.default_rng(seed)
        self.stores = stores
        self.forecasts = forecasts
        self.objective = objective
        self.units_constraint = OPTIMIZATION.get('units_constraint_O2', 10000)

        self.n_stores = len(stores)
        self.days = 7

        # dimensão total (21 por loja)
        self.dim = self.n_stores * 21

        # bounds baseados nas previsões

        all_forecasts = []

        for store in forecasts:
            all_forecasts.extend(forecasts[store])

        max_clients = int(np.max(all_forecasts))

        # pior caso: todos clientes atendidos por juniores
        max_hr = int(np.ceil(max_clients / 6))

        self.J_bounds = (0, int(np.ceil(max_clients / 6)))
        self.X_bounds = (0, int(np.ceil(max_clients / 7)))
        self.PR_bounds = (0.0, 0.3)
        

        self.PR_values = np.arange(0, 0.31, 0.05)

                # estimativa de máximos para normalização O3

        from core.profit_calculator import calculate_daily_profit

        max_total_profit = 0
        max_total_hr = 0

        for store in forecasts:

            forecast = forecasts[store]

            for d in range(7):

                customers = forecast[d]

                max_x = int(np.ceil(customers / 7))

                is_weekend = (d == 0) or (d == 6)

                daily_profit, _, daily_hr = calculate_daily_profit(
                    num_customers=customers,
                    J=0,
                    X=max_x,
                    PR=0.3,
                    is_weekend=is_weekend,
                    store_name=store
                )

                max_total_profit += daily_profit
                max_total_hr += daily_hr

        self.max_profit = max(max_total_profit, 1)
        self.max_hr = max(max_total_hr, 1)

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

    def _generate_feasible_solution(self):

        all_J = []
        all_X = []
        all_PR = []

        for store in self.stores:

            forecast = self.forecasts[store]

            for customers in forecast:

                max_x = int(np.ceil(customers / 7))

                if max_x <= 0:
                    x = 0
                else:
                    x = self.rng.integers(
                        0,
                        max_x + 1
                    )

                remaining = max(
                    0,
                    customers - x * 7
                )

                max_j = int(np.ceil(remaining / 6))

                if max_j <= 0:
                    j = 0
                else:
                    j = self.rng.integers(
                        0,
                        max_j + 1
                    )

                pr = self.rng.choice(self.PR_values)

                all_J.append(j)
                all_X.append(x)
                all_PR.append(pr)

        solution = np.array(
            all_J + all_X + all_PR,
            dtype=float
        )

        return solution


    def _fix_solution(self, solution):

        solution = solution.copy()

        n = self.n_stores * 7

        # =========================
        # J
        # =========================
        solution[0:n] = np.round(
            solution[0:n]
        ).astype(int)

        # =========================
        # X
        # =========================
        solution[n:2*n] = np.round(
            solution[n:2*n]
        ).astype(int)

        # =========================
        # PR
        # =========================
        pr = solution[2*n:3*n]

        pr = self.PR_values[
            np.abs(
                self.PR_values[:, None] - pr
            ).argmin(axis=0)
        ]

        solution[2*n:3*n] = pr

        return solution
    
    
    def _compute_total_units(self, J, X, PR):
        """Calcula o total de unidades para um vetor J, X, PR já dividido."""
        total_units = 0
        idx = 0
        for store in self.stores:
            for d in range(7):
                _, units, _ = calculate_daily_profit(
                    num_customers=int(self.forecasts[store][d]),
                    J=int(round(J[idx + d])),
                    X=int(round(X[idx + d])),
                    PR=float(PR[idx + d]),
                    is_weekend=(d == 0) or (d == 6),
                    store_name=store
                )
                total_units += units
            idx += 7
        return total_units

    def _repair_solution(self, solution):
        """
        Repara solução inviável garantindo que o constraint global de unidades
        seja respeitado (máx. 10000 unidades para os objetivos O2/O3).

        Estratégia:
        1. Clipping e fix inicial
        2. Redução proporcional rápida de trabalhadores se necessário
        3. Ajuste iterativo fino até satisfazer o constraint
        """
        solution = np.clip(solution.copy(), self.low, self.high)
        solution = self._fix_solution(solution)

        if self.objective not in ['O2', 'O3', 'O3_WEIGHTED', 'O3_NS']:
            return solution

        limit = int(self.units_constraint)
        J, X, PR = self._split_solution(solution)
        total_units = self._compute_total_units(J, X, PR)

        if total_units <= limit:
            return np.concatenate([J, X, PR])

        # --- Passo 1: redução proporcional rápida ---
        # Calcular quantos clientes podemos atender no total
        total_customers = sum(
            int(self.forecasts[store][d])
            for store in self.stores for d in range(7)
        )
        # Estimar unidades por cliente (média ponderada)
        if total_customers > 0:
            avg_units_per_customer = total_units / total_customers
        else:
            avg_units_per_customer = 17.0

        target_customers = limit / max(avg_units_per_customer, 1.0)
        scale = min(target_customers / max(total_customers, 1), 1.0)

        idx = 0
        for store in self.stores:
            for d in range(7):
                target = int(self.forecasts[store][d] * scale)
                max_x = int(np.ceil(target / 7))
                X[idx + d] = min(X[idx + d], float(max_x))
                remaining = max(0, target - max_x * 7)
                max_j = int(np.ceil(remaining / 6))
                J[idx + d] = min(J[idx + d], float(max_j))
            idx += 7

        # --- Passo 2: ajuste iterativo fino ---
        for _ in range(500):
            total_units = self._compute_total_units(J, X, PR)
            if total_units <= limit:
                break

            # Coletar métricas por dia
            daily_metrics = []
            idx = 0
            for store in self.stores:
                for d in range(7):
                    _, units, _ = calculate_daily_profit(
                        num_customers=int(self.forecasts[store][d]),
                        J=int(round(J[idx + d])),
                        X=int(round(X[idx + d])),
                        PR=float(PR[idx + d]),
                        is_weekend=(d == 0) or (d == 6),
                        store_name=store
                    )
                    daily_metrics.append((units, idx + d))
                idx += 7

            daily_metrics.sort(reverse=True, key=lambda x: x[0])

            # Reduzir os 3 dias com mais unidades
            for units_val, day_idx in daily_metrics[:3]:
                if X[day_idx] > 0:
                    X[day_idx] = max(0.0, X[day_idx] - 1)
                elif J[day_idx] > 0:
                    J[day_idx] = max(0.0, J[day_idx] - 1)

        return np.concatenate([J, X, PR])

    def _split_solution(self, solution):

        solution = self._fix_solution(solution)
        solution = np.clip(solution, self.low, self.high)

        n = self.n_stores * 7

        # estrutura correta:
        # [J global][X global][PR global]

        J = solution[0:n]
        X = solution[n:2*n]
        PR = solution[2*n:3*n]

        J = np.array(J)
        X = np.array(X)
        PR = np.array(PR)

        # discretizar PR
        PR = self.PR_values[
            np.abs(self.PR_values[:, None] - PR).argmin(axis=0)
        ]

        # limitar por forecast real
        idx = 0

        for store in self.stores:

            forecast = self.forecasts[store]

            for d in range(7):

                max_x = int(np.ceil(forecast[d] / 7))

                X[idx + d] = np.clip(
                    X[idx + d],
                    0,
                    max_x
                )

                remaining = max(
                    0,
                    forecast[d] - X[idx + d] * 7
                )

                max_j = int(np.ceil(remaining / 6))

                J[idx + d] = np.clip(
                    J[idx + d],
                    0,
                    max_j
                )   

            idx += self.days

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
            self.objective,
            self.max_profit,
            self.max_hr
        )

        if self.objective == 'O3_WEIGHTED':

            profit = result.get('profit', -1e10)
            hr = result.get('hr', 1e10)

            normalized_profit = profit / self.max_profit
            normalized_hr = hr / self.max_hr

            value = (
                0.7 * normalized_profit
                - 0.3 * normalized_hr
            )

        else:

            value = result.get('best_value', -1e10)

        if not np.isfinite(value):
            return -1e10

        return value
    

    def _evaluate_multiobjective(self, solution):
        

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
            'O3_NS',
            self.max_profit,
            self.max_hr
        )

        profit = result.get('profit', result.get('best_value', -1e10))
        hr = result.get('hr', 1e10)

        # infeasible
        if not np.isfinite(profit):
            return (-1e10, 1e10)

        return (
            profit,
            hr
        )
    

    def dominates(self, a, b):

        profit_a, hr_a = a
        profit_b, hr_b = b

        better_or_equal = (
            profit_a >= profit_b and
            hr_a <= hr_b
        )

        strictly_better = (
            profit_a > profit_b or
            hr_a < hr_b
        )

        return better_or_equal and strictly_better
    
    def fast_non_dominated_sort(self, objectives):

        population_size = len(objectives)

        S = [[] for _ in range(population_size)]
        n = [0] * population_size
        rank = [0] * population_size

        fronts = [[]]

        for p in range(population_size):

            for q in range(population_size):

                if self.dominates(objectives[p], objectives[q]):

                    S[p].append(q)

                elif self.dominates(objectives[q], objectives[p]):

                    n[p] += 1

            if n[p] == 0:

                rank[p] = 0
                fronts[0].append(p)

        i = 0

        while fronts[i]:

            next_front = []

            for p in fronts[i]:

                for q in S[p]:

                    n[q] -= 1

                    if n[q] == 0:

                        rank[q] = i + 1
                        next_front.append(q)

            i += 1
            fronts.append(next_front)

        fronts.pop()

        return fronts
    
    def crowding_distance(self, front, objectives):

        distance = np.zeros(len(front))

        if len(front) == 0:
            return distance

        num_objectives = 2

        for m in range(num_objectives):

            values = np.array([
                objectives[i][m]
                for i in front
            ])

            sorted_idx = np.argsort(values)

            distance[sorted_idx[0]] = np.inf
            distance[sorted_idx[-1]] = np.inf

            vmin = values[sorted_idx[0]]
            vmax = values[sorted_idx[-1]]

            if vmax - vmin == 0:
                continue

            for k in range(1, len(front)-1):

                prev_value = values[sorted_idx[k-1]]
                next_value = values[sorted_idx[k+1]]

                distance[sorted_idx[k]] += (
                    next_value - prev_value
                ) / (vmax - vmin)

        return distance

            

    def random_search(self, n_iter=300):

        best_solution = None
        best_value = -np.inf
        history = []

        for _ in range(n_iter):

            solution = self._repair_solution(
                self._generate_feasible_solution()
            )

            value = self._evaluate(solution)

            if value > best_value:
                best_value = value
                best_solution = solution

            history.append(best_value)

        return {
            'solution': best_solution.copy(),
            'best_value': best_value,
            'history': history
        }
    

    def hill_climbing(self, max_iter=300):
        # tentar encontrar solução inicial válida
        for _ in range(100):
            solution = self._repair_solution(
                self._generate_feasible_solution()
            )

            value = self._evaluate(solution)
            # aceitar solução inicial razoável
            if value > -10000:
                break

        best_value = self._evaluate(solution)
        best_solution = solution.copy() 

        history = [best_value]

        for _ in range(max_iter):

            neighbor = solution + self.rng.normal(0, 1.0, self.dim)
            neighbor = np.clip(neighbor, self.low, self.high)
            neighbor = self._repair_solution(neighbor)          

            value = self._evaluate(neighbor)

            if value > best_value:
                best_value = value
                best_solution = neighbor.copy()
                solution = neighbor.copy()

            history.append(best_value)

        return {
            'solution': best_solution.copy(),
            'best_value': best_value,
            'history': history
        }
    
    def genetic_algorithm(self, population_size=30, generations=50):

        population = np.array([
            self._repair_solution(
                self._generate_feasible_solution()
            )
            for _ in range(population_size)
        ])
 
        history = []

        for _ in range(generations):

            fitness = np.array([self._evaluate(ind) for ind in population])

            best_idx = np.argmax(fitness)
            best_value = fitness[best_idx]
            history.append(best_value)

            #  seleção (torneio)
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
                   mutation = self.rng.normal(0, 1.0, self.dim)
                   child += mutation

               child = np.clip(child, self.low, self.high)
               child = self._repair_solution(child)

               new_population.append(child)

            population = np.array(new_population[:population_size])

        best_idx = np.argmax([self._evaluate(ind) for ind in population])
        best_solution = population[best_idx]

        return {
           'solution': best_solution.copy(),
           'best_value': self._evaluate(best_solution),
           'history': history
        }
    

    def nsga2(self, population_size=60, generations=100):

        population = np.array([
            self._repair_solution(
                self._generate_feasible_solution()
         )
            for _ in range(population_size)
        ])

        history = []

        for generation in range(generations):

            objectives = [
                self._evaluate_multiobjective(ind)
                for ind in population
            ]

            fronts = self.fast_non_dominated_sort(objectives)

            new_population = []

            for front in fronts:

                if len(new_population) + len(front) > population_size:

                    distances = self.crowding_distance(
                        front,
                        objectives
                    )

                    sorted_front = [
                        front[i]
                        for i in np.argsort(-distances)
                    ]

                    remaining = (
                        population_size - len(new_population)
                    )

                    new_population.extend(
                        population[idx]
                        for idx in sorted_front[:remaining]
                    )

                    break

                else:

                    new_population.extend(
                        population[idx]
                        for idx in front
                    )

            offspring = []

            while len(offspring) < population_size:

                p1 = new_population[
                    self.rng.integers(0, len(new_population))
                ]

                p2 = new_population[
                    self.rng.integers(0, len(new_population))
                ]

                alpha = self.rng.random(self.dim)

                child = alpha * p1 + (1 - alpha) * p2

                if self.rng.random() < 0.2:

                    child += self.rng.normal(
                        0,
                        1.0,
                        self.dim
                    )

                

                child = np.clip(
                    child,
                    self.low,
                    self.high
                )
                child = self._repair_solution(child)

                offspring.append(child)

            population = np.array(
                new_population + offspring
            )[:population_size]

            best_profit = max(
                obj[0]
                for obj in objectives
            )

            history.append(best_profit)

        objectives = [
            self._evaluate_multiobjective(ind)
            for ind in population
        ]

        fronts = self.fast_non_dominated_sort(objectives)

        pareto_front = fronts[0]

        valid_front = [
            idx for idx in pareto_front
            if objectives[idx][0] > -1e9
        ]

        if len(valid_front) == 0:
            best_idx = np.argmax([
                obj[0] for obj in objectives
            ])
        else:
            best_idx = max(
                valid_front,
                key=lambda idx: objectives[idx][0]
            )

        pareto_solutions = [
            {
                'solution': self._fix_solution(population[idx].copy()),
                'profit': objectives[idx][0],
                'hr': objectives[idx][1]
            }
            for idx in valid_front
        ]

        return {
            'solution': population[best_idx].copy(),
            'pareto_front': pareto_front,
            'pareto_solutions': pareto_solutions,
            'objectives': objectives,
            'history': history,
            'best_value': objectives[best_idx][0]
        }
    


    def particle_swarm(self, n_particles=30, max_iter=50):

        particles = np.array([
            self._repair_solution(
                self._generate_feasible_solution()
            )
            for _ in range(n_particles)
        ])
        velocities = self.rng.uniform(-1, 1, size=(n_particles, self.dim))

        
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
                particles[i] = self._repair_solution(particles[i])
                

                value = self._evaluate(particles[i])

                if value > personal_best_values[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_values[i] = value

                    if value > global_best_value:
                        global_best = self._repair_solution(
                            self._fix_solution(
                                particles[i].copy()
                        )
            )
                        global_best_value = value

            history.append(global_best_value)

        return {
            'solution': self._repair_solution(
                self._fix_solution(
                    global_best.copy()
                )
            ),
            'best_value': global_best_value,
            'history': history
        }       
    

    def simulated_annealing(self, max_iter=300, temp_init=100):

        solution = self._repair_solution(
            self._generate_feasible_solution()
        )

        best_solution = solution.copy()
        best_value = self._evaluate(solution)
        current_value = best_value

        history = [best_value]

        for i in range(max_iter):

            temp = temp_init * (0.95 ** i)

            neighbor = solution + self.rng.normal(0, 1.0, self.dim)
            neighbor = np.clip(neighbor, self.low, self.high)
            neighbor = self._repair_solution(neighbor)
                        

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
            'solution': best_solution.copy(),
            'best_value': best_value,
            'history': history
        }
    

    def differential_evolution(self, pop_size=30, generations=50):

        population = np.array([
            self._repair_solution(
                self._generate_feasible_solution()
            )
            for _ in range(pop_size)
        ])

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
                trial = self._repair_solution(trial)

                if self._evaluate(trial) > self._evaluate(population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            best_value = max(self._evaluate(ind) for ind in population)
            history.append(best_value)

        best_idx = np.argmax([self._evaluate(ind) for ind in population])

        return {
            'solution': population[best_idx].copy(),
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

        elif method == 'nsga2':
            result = self.nsga2()

        else:
            raise ValueError(f"Método desconhecido: {method}")

        output = {
            'method': method,
            'solution': result['solution'],
            'best_value': result['best_value'],
            'history': result['history']
        }

        if 'pareto_solutions' in result:
            output['pareto_solutions'] = result['pareto_solutions']

        if 'pareto_front' in result:
            output['pareto_front'] = result['pareto_front']

        if 'objectives' in result:
            output['objectives'] = result['objectives']

        return output