"6 métodos de otimização: Random Search, Hill Climbing, Simulated Annealing, Genetic Algorithm, PSO, Differential Evolution."

import sys
from pathlib import Path
from unittest import result

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.profit_calculator import calculate_daily_profit

import numpy as np
from core.profit_calculator import calculate_daily_profit, evaluate_solution


class OptimizationMethods:
    """Classe que implementa 6 métodos de otimização metaheurística."""
    
    def __init__(self, store_name, customers_forecast, objective='O1', method='random', seed=None):
        self.rng = np.random.default_rng(seed)
        self.store_name = store_name
        self.customers_forecast = customers_forecast
        self.objective = objective
        self.method = method
        self.PR_values = np.arange(0, 0.31, 0.05)
        
        # Espaço de solução baseado na previsão

        max_clients = int(np.max(customers_forecast))

        # capacidade mínima = 6 clientes por funcionário
        max_hr = max(
            int(np.ceil(max_clients / 6)),
            int(np.ceil(max_clients / 7))
)

        self.J_bounds = (0, max_hr)
        self.X_bounds = (0, max_hr)
        self.PR_bounds = (0.0, 0.3)

        # estimativa de máximos para normalização O3

        max_daily_profit = 0
        max_daily_hr = 0

        for d in range(7):

            customers = customers_forecast[d]

            # máximo RH possível
            max_x = int(np.ceil(customers / 7))

            is_weekend = (d == 0) or (d == 6)

            # promoção máxima
            pr = 0.0

            # cálculo otimista
            

            daily_profit, _, daily_hr = calculate_daily_profit(
                num_customers=customers,
                J=0,
                X=max_x,
                PR=pr,
                is_weekend=is_weekend,
                store_name=store_name
            )

            max_daily_profit += daily_profit
            max_daily_hr += daily_hr

        self.max_profit = max(max_daily_profit, 1)
        self.max_hr = max(max_daily_hr, 1)
        
        # Solução = [J0, J1, ..., J6, X0, X1, ..., X6, PR0, PR1, ..., PR6] (21 dims)
        self.dim = 21

    def _generate_feasible_solution(self):

        J = []
        X = []

        for customers in self.customers_forecast:

            max_x = int(np.ceil(customers / 7))

            x = self.rng.integers(0, max_x + 1)

            remaining = max(0, customers - x * 7)

            max_j = int(np.ceil(remaining / 6))

            j = self.rng.integers(0, max_j + 1)

            J.append(j)
            X.append(x)

        PR = self.rng.choice(
            self.PR_values,
            size=7
       )

        solution = np.array(
            J + X + list(PR),
            dtype=float
        )

        return solution


    def _fix_solution(self, solution):

        solution = solution.copy()

        # J inteiros
        solution[0:7] = np.round(solution[0:7]).astype(int)

        # X inteiros
        solution[7:14] = np.round(solution[7:14]).astype(int)

        # PR discreto
        pr = solution[14:21]

        pr = np.array([
            self.PR_values[
                np.abs(self.PR_values - p).argmin()
            ]
            for p in pr
        ])

        solution[14:21] = pr

        return solution 
    
    
    def _evaluate(self, solution):
        """Avalia uma solução completa.
        
        Args:
            solution: Array de 21 valores
        
        Returns:
            Float: valor (quanto maior, melhor)
        """
        solution = self._fix_solution(solution)

        J = np.clip(solution[0:7], self.J_bounds[0], self.J_bounds[1])
        X = np.clip(solution[7:14], self.X_bounds[0], self.X_bounds[1])
        PR = np.clip(
            solution[14:21],
            self.PR_bounds[0],
            self.PR_bounds[1]
        )

        PR = np.array([
            self.PR_values[
                np.abs(self.PR_values - p).argmin()
            ]
            for p in PR
        ])
        
        result = evaluate_solution(
            J,
    X,
    PR,
    self.customers_forecast,
    self.store_name,
    self.objective,
    self.max_profit,
    self.max_hr
)

        # corrigir tipo (dict → float)
        if isinstance(result, dict):
            value = result.get('best_value', result.get('value', -1e10))
        else:
            value = result

        value = float(value)

        # segurança
        if not np.isfinite(value):
            return value

        return value
    

   

    def hill_climbing(self, max_iter=300):
        """Hill Climbing com vizinhança de 6 pontos.
        
        Returns:
            Dict com 'solution', 'value', 'history'
        """
        # Inicializar solução aleatória
        solution = self._generate_feasible_solution( )         
        
        
        best_value = self._evaluate(solution)
        history = [best_value]
        
        for iteration in range(max_iter):
            improved = False
            
            # Testar 6 vizinhos (perturbação em cada dimensão)
            neighbors = []
            indices = self.rng.choice(self.dim, size=6, replace=False)

            step = 0.5
            
            for i in indices:
               plus = solution.copy()
               minus = solution.copy()

               plus[i] += step
               minus[i] -= step

               neighbors.append(plus)
               neighbors.append(minus)
            
            # Clip aos limites
            for neighbor in neighbors:
                neighbor[0:7] = np.clip(neighbor[0:7], self.J_bounds[0], self.J_bounds[1])
                neighbor[7:14] = np.clip(neighbor[7:14], self.X_bounds[0], self.X_bounds[1])
                neighbor[14:21] = np.clip(neighbor[14:21], self.PR_bounds[0], self.PR_bounds[1])
            
            best_neighbor = None

            for neighbor in neighbors:
                neighbor_value = self._evaluate(neighbor)
                if neighbor_value > best_value:
                     best_neighbor = neighbor
                     best_value = neighbor_value

            if best_neighbor is not None:
                solution = best_neighbor                
                history.append(best_value)
            else:
                history.append(best_value)
            
            
        return {
            'solution': self._fix_solution(solution),
            'value': best_value,
            'history': history
        }
    
    def simulated_annealing(self, max_iter=300, temp_init=100):
        """Simulated Annealing com cooling schedule.
        
        Returns:
            Dict com 'solution', 'value', 'history'
        """
        solution = self._generate_feasible_solution()
            
        
        
        best_solution = solution.copy()
        best_value = self._evaluate(solution)
        current_value = best_value
        
        history = [best_value]
        
        for iteration in range(max_iter):
            # Cooling schedule
            temp = temp_init * (0.95 ** iteration)
            
            # Gerar vizinho por perturbação aleatória
            scale = temp / temp_init
            neighbor = solution + self.rng.normal(0, 0.3 * scale, self.dim)
            neighbor[0:7] = np.clip(neighbor[0:7], self.J_bounds[0], self.J_bounds[1])
            neighbor[7:14] = np.clip(neighbor[7:14], self.X_bounds[0], self.X_bounds[1])
            neighbor[14:21] = np.clip(neighbor[14:21], self.PR_bounds[0], self.PR_bounds[1])
            
            neighbor_value = self._evaluate(neighbor)
            
            # Critério de aceitação
            delta = neighbor_value - current_value
            if delta > 0 or self.rng.random() < np.exp(delta / (temp + 1e-10)):
                solution = neighbor
                current_value = neighbor_value
            
            # Atualizar melhor encontrado
            if current_value > best_value:
                best_solution = solution.copy()
                best_value = current_value
            
            history.append(best_value)
        
        return {
            'solution': self._fix_solution(best_solution),
            'value': best_value,
            'history': history
        }
    
    def genetic_algorithm(self, population_size=30, generations=50):
        """Genetic Algorithm com crossover e mutação.
        
        Returns:
            Dict com 'solution', 'value', 'history'
        """
        population = np.array([
            self._generate_feasible_solution()
            for _ in range(population_size)
        ])
        
        history = []
        
        for generation in range(generations):
            # Avaliar população
            fitness = np.array([self._evaluate(ind) for ind in population])
            
            best_idx = np.argmax(fitness)
            best_value = fitness[best_idx]
            history.append(best_value)
            
            # Selection: torneio
            def tournament_select():
                idx = self.rng.choice(population_size, 3, replace=False)
                return population[idx[np.argmax(fitness[idx])]].copy()
            
            # Criar nova população
            new_population = [population[best_idx].copy()]  # Elitismo
            
            while len(new_population) < population_size:
                # Crossover
                parent1 = tournament_select()
                parent2 = tournament_select()
                
                alpha = self.rng.random(self.dim)
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Mutação
                if self.rng.random() < 0.2:
                    mutation_idx = self.rng.integers(0, self.dim)
                    if mutation_idx < 7:
                        child[mutation_idx] += self.rng.normal(0, 1)
                    elif mutation_idx < 14:
                        child[mutation_idx] += self.rng.normal(0, 1)
                    else:
                        child[mutation_idx] += self.rng.normal(0, 0.05)
                
                # Clip
                child[0:7] = np.clip(child[0:7], self.J_bounds[0], self.J_bounds[1])
                child[7:14] = np.clip(child[7:14], self.X_bounds[0], self.X_bounds[1])
                child[14:21] = np.clip(child[14:21], self.PR_bounds[0], self.PR_bounds[1])
                
                new_population.append(child)
            
            population = np.array(new_population[:population_size])
        
        # Retornar melhor solução encontrada
        best_idx = np.argmax(np.array([self._evaluate(ind) for ind in population]))
        best_solution = population[best_idx]
        
        return {
            'solution': self._fix_solution(best_solution),
            'value': self._evaluate(best_solution),
            'history': history
        }
    
    
    def random_search(self, n_iter=300):
        best_solution = None
        best_value = -np.inf
        history = []

        for _ in range(n_iter):
            solution = self._generate_feasible_solution()
                
            

            value = self._evaluate(solution)

            if value > best_value:
                best_value = value
                best_solution = solution

            history.append(best_value)

        return {
            'solution': self._fix_solution(best_solution),
            'value': best_value,
            'history': history
        }
    
    def differential_evolution(self, pop_size=30, generations=50):

        population = np.array([
            self._generate_feasible_solution()
            for _ in range(pop_size)
        ])

        history = []

        for _ in range(generations):
            new_population = []

            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[self.rng.choice(idxs, 3, replace=False)]

                mutant = a + 0.8 * (b - c)
                mutant[0:7] = np.clip(mutant[0:7], self.J_bounds[0], self.J_bounds[1])
                mutant[7:14] = np.clip(mutant[7:14], self.X_bounds[0], self.X_bounds[1])
                mutant[14:21] = np.clip(mutant[14:21], self.PR_bounds[0], self.PR_bounds[1])

                cross_points = self.rng.random(self.dim) < 0.7
                trial = np.where(cross_points, mutant, population[i])

                trial[0:7] = np.clip(trial[0:7], self.J_bounds[0], self.J_bounds[1])
                trial[7:14] = np.clip(trial[7:14], self.X_bounds[0], self.X_bounds[1])
                trial[14:21] = np.clip(trial[14:21], self.PR_bounds[0], self.PR_bounds[1])

                if self._evaluate(trial) > self._evaluate(population[i]):
                    new_population.append(trial)
                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            best_value = max([self._evaluate(ind) for ind in population])
            history.append(best_value)

        best_idx = np.argmax([self._evaluate(ind) for ind in population])

        return {
            'solution': self._fix_solution(population[best_idx]),
            'value': self._evaluate(population[best_idx]),
            'history': history
       }
    
    def particle_swarm(self, n_particles=30, max_iter=50):
        """Particle Swarm Optimization.
        
        Returns:
            Dict com 'solution', 'value', 'history'
        """
        particles = np.array([
            self._generate_feasible_solution()
            for _ in range(n_particles)
        ])
        
        velocities = self.rng.uniform(-1, 1, size=(n_particles, self.dim))
        
        
        # Best pessoal e global
        particles_best = particles.copy()
        particles_best_values = np.array([self._evaluate(p) for p in particles])
        
        global_best_idx = np.argmax(particles_best_values)
        global_best = particles[global_best_idx].copy()
        global_best_value = particles_best_values[global_best_idx]
        
        history = [global_best_value]
        
        # PSO iterations
        w = 0.7              # Inércia
        c1 = 1.5             # Coeficiente cognitivo
        c2 = 1.5             # Coeficiente social
        
        for iteration in range(max_iter):
            for i in range(n_particles):
                # Atualizar velocidade
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                
                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (particles_best[i] - particles[i]) +
                    c2 * r2 * (global_best - particles[i])
                )
                velocities[i] = np.clip(velocities[i], -2, 2)
                # Atualizar posição
                particles[i] = particles[i] + velocities[i]
                
                
                # Clip
                particles[i][0:7] = np.clip(particles[i][0:7], self.J_bounds[0], self.J_bounds[1])
                particles[i][7:14] = np.clip(particles[i][7:14], self.X_bounds[0], self.X_bounds[1])
                particles[i][14:21] = np.clip(particles[i][14:21], self.PR_bounds[0], self.PR_bounds[1])
                
                # Avaliar
                value = self._evaluate(particles[i])
                if value > particles_best_values[i]:
                    particles_best[i] = particles[i].copy()
                    particles_best_values[i] = value
                
                # Atualizar global
                if value > global_best_value:
                    global_best = particles[i].copy()
                    global_best_value = value
            
            history.append(global_best_value)
        
        return {
            'solution': self._fix_solution(global_best),
            'value': global_best_value,
            'history': history
        }
    

    def optimize(self):
        print(f"\n  Otimizando {self.objective} para {self.store_name} usando {self.method}...")

        if self.method == 'random':
           result = self.random_search()

        elif self.method == 'hill_climbing':
           result = self.hill_climbing()

        elif self.method == 'simulated_annealing':
           result = self.simulated_annealing()

        elif self.method == 'genetic':
           result = self.genetic_algorithm()

        elif self.method == 'pso':
           result = self.particle_swarm()

        elif self.method == 'de':
           result = self.differential_evolution()  # só se tiveres este método

        else:
           raise ValueError(f"Método desconhecido: {self.method}")

        return {
             'method': self.method,
             'solution': result['solution'],
             'best_value': result['value'],
             'history': result.get('history', [])
        }