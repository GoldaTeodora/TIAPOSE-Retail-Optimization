import numpy as np

methods = ['random', 'hill_climbing', 'simulated_annealing', 'genetic', 'pso']
seeds = [0, 1, 2, 3, 4]

results = []

for method in methods:
    for seed in seeds:
        optimizer = OptimizationMethods(
            store_name='store_1',
            customers_forecast=forecast,
            method=method,
            seed=seed
        )

        import time

        start = time.time()
        result = optimizer.optimize()
        elapsed = time.time() - start

        results.append({
            'method': method,
            'seed': seed,
            'value': result['value'],
            'history': result['history'],
            'time': elapsed
        })

import pandas as pd

df = pd.DataFrame(results)

summary = df.groupby('method').agg({
    'value': ['mean', 'std', 'max'],
    'time': 'mean'
}).reset_index()

summary.columns = ['method', 'mean', 'std', 'max', 'avg_time']

summary['cv'] = summary['std'] / (summary['mean'] + 1e-10)
summary['efficiency'] = summary['mean'] / (summary['avg_time'] + 1e-10)

summary['rank'] = summary['mean'].rank(ascending=False).astype(int)
summary = summary.sort_values('rank')

print(summary)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

colors = plt.cm.tab10.colors

for i, method in enumerate(summary['method']):
    subset = df[df['method'] == method]

    max_len = max(len(h) for h in subset['history'])

    histories = []
    for h in subset['history']:
        padded = h + [h[-1]] * (max_len - len(h))
        histories.append(padded)

    avg_history = np.mean(histories, axis=0)
    plt.plot(avg_history, label=method, color=colors[i])

plt.legend()
plt.xlabel("Iterações")
plt.ylabel("Lucro")
plt.title("Convergência média dos algoritmos de otimização")
plt.grid(alpha=0.3)

plt.savefig("optimization_convergence.png", dpi=300, bbox_inches='tight')
plt.show()

df.to_csv("optimization_results_by_run.csv", index=False)

summary.to_csv("optimization_summary_by_method.csv", index=False)

