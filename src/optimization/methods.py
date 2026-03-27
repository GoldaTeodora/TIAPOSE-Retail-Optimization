"""
4 métodos de otimização: Hill Climbing, Simulated Annealing, Genetic Algorithm, PSO.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.profit_calculator import evaluate_solution


class OptimizationMethods:
    """Classe que implementa 4 métodos de otimização metaheurística."""
    
    def __init__(self, store_name, customers_forecast, objective='O1'):
        """Inicializa otimizador.
        
        Args:
            store_name: Nome da loja
            customers_forecast: Array de 7 previsões de clientes
            objective: 'O1', 'O2', ou 'O3'
        """
        self.store_name = store_name
        self.customers_forecast = customers_forecast
        self.objective = objective
        
        # Espaço de solução
        self.J_bounds = (0, 20)
        self.X_bounds = (0, 20)
        self.PR_bounds = (0.0, 0.3)
        
        # Solução = [J0, J1, ..., J6, X0, X1, ..., X6, PR0, PR1, ..., PR6] (21 dims)
        self.dim = 21
    
    def _evaluate(self, solution):
        """Avalia uma solução completa.
        
        Args:
            solution: Array de 21 valores
        
        Returns:
            Float: valor (quanto maior, melhor)
        """
        J = np.clip(solution[0:7], self.J_bounds[0], self.J_bounds[1])
        X = np.clip(solution[7:14], self.X_bounds[0], self.X_bounds[1])
        PR = np.clip(solution[14:21], self.PR_bounds[0], self.PR_bounds[1])
        
        return evaluate_solution(J, X, PR, self.customers_forecast, 
                                self.store_name, self.objective)
    
    def hill_climbing(self, max_iter=100):
        """Hill Climbing com vizinhança de 6 pontos.
        
        Returns:
            Dict com 'solution', 'value', 'history'
        """
        # Inicializar solução aleatória
        solution = np.random.uniform(
            [self.J_bounds[0]]*7 + [self.X_bounds[0]]*7 + [self.PR_bounds[0]]*7,
            [self.J_bounds[1]]*7 + [self.X_bounds[1]]*7 + [self.PR_bounds[1]]*7
        )
        
        best_value = self._evaluate(solution)
        history = [best_value]
        
        for iteration in range(max_iter):
            improved = False
            
            # Testar 6 vizinhos (perturbação em cada dimensão)
            neighbors = [solution.copy() for _ in range(6)]
            step = 0.5  # Tamanho do passo
            
            for i in range(3):
                neighbors[2*i][i] += step
                neighbors[2*i+1][i] -= step
            
            # Clip aos limites
            for neighbor in neighbors:
                neighbor[0:7] = np.clip(neighbor[0:7], self.J_bounds[0], self.J_bounds[1])
                neighbor[7:14] = np.clip(neighbor[7:14], self.X_bounds[0], self.X_bounds[1])
                neighbor[14:21] = np.clip(neighbor[14:21], self.PR_bounds[0], self.PR_bounds[1])
            
            # Avaliar vizinhos
            for neighbor in neighbors:
                neighbor_value = self._evaluate(neighbor)
                if neighbor_value > best_value:
                    solution = neighbor
                    best_value = neighbor_value
                    improved = True
                    history.append(best_value)
                    break
            
            if not improved:
                history.append(best_value)
        
        return {
            'solution': solution,
            'value': best_value,
            'history': history
        }
    
    def simulated_annealing(self, max_iter=200, temp_init=100):
        """Simulated Annealing com cooling schedule.
        
        Returns:
            Dict com 'solution', 'value', 'history'
        """
        solution = np.random.uniform(
            [self.J_bounds[0]]*7 + [self.X_bounds[0]]*7 + [self.PR_bounds[0]]*7,
            [self.J_bounds[1]]*7 + [self.X_bounds[1]]*7 + [self.PR_bounds[1]]*7
        )
        
        best_solution = solution.copy()
        best_value = self._evaluate(solution)
        current_value = best_value
        
        history = [best_value]
        
        for iteration in range(max_iter):
            # Cooling schedule
            temp = temp_init * (1 - iteration / max_iter)
            
            # Gerar vizinho por perturbação aleatória
            neighbor = solution + np.random.normal(0, 0.3, self.dim)
            neighbor[0:7] = np.clip(neighbor[0:7], self.J_bounds[0], self.J_bounds[1])
            neighbor[7:14] = np.clip(neighbor[7:14], self.X_bounds[0], self.X_bounds[1])
            neighbor[14:21] = np.clip(neighbor[14:21], self.PR_bounds[0], self.PR_bounds[1])
            
            neighbor_value = self._evaluate(neighbor)
            
            # Critério de aceitação
            delta = neighbor_value - current_value
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                solution = neighbor
                current_value = neighbor_value
            
            # Atualizar melhor encontrado
            if current_value > best_value:
                best_solution = solution.copy()
                best_value = current_value
            
            history.append(best_value)
        
        return {
            'solution': best_solution,
            'value': best_value,
            'history': history
        }
    
    def genetic_algorithm(self, population_size=30, generations=50):
        """Genetic Algorithm com crossover e mutação.
        
        Returns:
            Dict com 'solution', 'value', 'history'
        """
        # Inicializar população
        population = np.random.uniform(
            [self.J_bounds[0]]*7 + [self.X_bounds[0]]*7 + [self.PR_bounds[0]]*7,
            [self.J_bounds[1]]*7 + [self.X_bounds[1]]*7 + [self.PR_bounds[1]]*7,
            size=(population_size, self.dim)
        )
        
        history = []
        
        for generation in range(generations):
            # Avaliar população
            fitness = np.array([self._evaluate(ind) for ind in population])
            
            best_idx = np.argmax(fitness)
            best_value = fitness[best_idx]
            history.append(best_value)
            
            # Selection: torneio
            def tournament_select():
                idx = np.random.choice(population_size, 3, replace=False)
                return population[idx[np.argmax(fitness[idx])]].copy()
            
            # Criar nova população
            new_population = [population[best_idx].copy()]  # Elitismo
            
            while len(new_population) < population_size:
                # Crossover
                parent1 = tournament_select()
                parent2 = tournament_select()
                
                alpha = np.random.random(self.dim)
                child = alpha * parent1 + (1 - alpha) * parent2
                
                # Mutação
                if np.random.random() < 0.1:
                    mutation_idx = np.random.randint(0, self.dim)
                    if mutation_idx < 7:
                        child[mutation_idx] += np.random.normal(0, 1)
                    elif mutation_idx < 14:
                        child[mutation_idx] += np.random.normal(0, 1)
                    else:
                        child[mutation_idx] += np.random.normal(0, 0.05)
                
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
            'solution': best_solution,
            'value': self._evaluate(best_solution),
            'history': history
        }
    
    def particle_swarm(self, n_particles=20, max_iter=50):
        """Particle Swarm Optimization.
        
        Returns:
            Dict com 'solution', 'value', 'history'
        """
        # Inicializar partículas
        particles = np.random.uniform(
            [self.J_bounds[0]]*7 + [self.X_bounds[0]]*7 + [self.PR_bounds[0]]*7,
            [self.J_bounds[1]]*7 + [self.X_bounds[1]]*7 + [self.PR_bounds[1]]*7,
            size=(n_particles, self.dim)
        )
        
        velocities = np.random.uniform(-1, 1, size=(n_particles, self.dim))
        
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
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                velocities[i] = (
                    w * velocities[i] +
                    c1 * r1 * (particles_best[i] - particles[i]) +
                    c2 * r2 * (global_best - particles[i])
                )
                
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
            'solution': global_best,
            'value': global_best_value,
            'history': history
        }
    
    def optimize(self):
        """Executa todos os 4 métodos e retorna o melhor.
        
        Returns:
            Dict com 'method', 'solution', 'value', 'all_results'
        """
        print(f"\n  Otimizando {self.objective} para {self.store_name}...")
        
        results = {
            'hill_climbing': self.hill_climbing(),
            'simulated_annealing': self.simulated_annealing(),
            'genetic_algorithm': self.genetic_algorithm(),
            'particle_swarm': self.particle_swarm()
        }
        
        # Encontrar melhor
        best_method = max(results.keys(), key=lambda m: results[m]['value'])
        best = results[best_method]
        
        print(f"    Melhor: {best_method} (valor={best['value']:.2f})")
        
        return {
            'method': best_method,
            'solution': best['solution'],
            'value': best['value'],
            'all_results': results
        }
