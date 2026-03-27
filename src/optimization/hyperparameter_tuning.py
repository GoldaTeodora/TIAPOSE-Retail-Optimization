"""
HYPERPARAMETER TUNING COM GRIDSEARCH
====================================

Este módulo implementa TIER 1: GridSearch para otimizar hiperparâmetros do XGBoost.

Técnica: Busca em grelha (GridSearch) com validação cruzada
- Testa múltiplas combinações de hiperparâmetros
- Usa k-fold cross-validation (CV=5) para ser robusto
- Selecciona melhor combinação baseado em score do fold de validação

Hiperparâmetros testados:
1. max_depth: Profundidade das árvores (controla complexidade)
2. learning_rate: Taxa de aprendizado (controla velocidade)
3. n_estimators: Número de árvores (controla capacidade)

Impacto esperado: +10-15% R² vs configuração padrão

Tempo esperado: ~20-30 minutos para grid completo (24 combinações × 5 CV folds)

Autor: AI Assistant | Data: Março 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')


class XGBoostTuner:
    """
    Tuner automático para XGBoost usando GridSearch.
    
    Fluxo:
    1. Recebe dados de treino
    2. Define grid de hiperparâmetros
    3. Executa GridSearchCV (testa todas as combinações)
    4. Retorna melhor modelo e parâmetros
    5. Valida em dados de teste
    """
    
    def __init__(self, verbose=True):
        """
        Args:
            verbose: Se True, imprime progresso detalhado
        """
        self.verbose = verbose
        self.best_model = None
        self.best_params = None
        self.grid_results = None
        self.cv_scores = None
        self.best_score = None
    
    def _log(self, msg):
        """Helper para printar apenas se verbose=True"""
        if self.verbose:
            print(f"  [Tuner] {msg}")
    
    # ==================== DEFINIÇÃO DE GRIDS ====================
    
    def get_param_grid(self, grid_size='medium'):
        """
        Define grid de hiperparâmetros a testar.
        
        Três opções:
        
        1. 'quick' (5-10 min): Grid pequeno, apenas testes base
           - max_depth: [5, 6, 7, 8]
           - learning_rate: [0.05, 0.1]
           - n_estimators: [300]
           
        2. 'medium' (20-30 min): Grid completo recomendado
           - max_depth: [5, 6, 7, 8, 9]
           - learning_rate: [0.05, 0.1, 0.15]
           - n_estimators: [300, 500]
           
        3. 'extensive' (60-90 min): Grid muito grande, mais completo
           - max_depth: [4, 5, 6, 7, 8, 9, 10]
           - learning_rate: [0.01, 0.05, 0.1, 0.15, 0.2]
           - n_estimators: [200, 300, 500, 800]
        
        Args:
            grid_size: 'quick', 'medium', ou 'extensive'
        
        Returns:
            Dict com grid de parâmetros
        """
        
        if grid_size == 'quick':
            param_grid = {
                'max_depth': [5, 6, 7, 8],
                'learning_rate': [0.05, 0.1],
                'n_estimators': [300]
            }
            n_combinations = 4 * 2 * 1
            
        elif grid_size == 'medium':
            param_grid = {
                'max_depth': [5, 6, 7, 8, 9],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [300, 500]
            }
            n_combinations = 5 * 3 * 2
            
        elif grid_size == 'extensive':
            param_grid = {
                'max_depth': [4, 5, 6, 7, 8, 9, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'n_estimators': [200, 300, 500, 800]
            }
            n_combinations = 7 * 5 * 4
        
        else:
            raise ValueError(f"grid_size deve ser 'quick', 'medium' ou 'extensive', recebido: {grid_size}")
        
        self._log(f"Grid '{grid_size}': {n_combinations} combinações a testar")
        
        return param_grid
    
    # ==================== GRIDSEARCH COM CV ====================
    
    def tune_xgboost(self, X_train, y_train, grid_size='medium', cv_folds=5):
        """
        Executa GridSearch com validação cruzada.
        
        Processo:
        1. Define grid de parâmetros
        2. Cria GridSearchCV com cross-validation
        3. Testa cada combinação em k folds
        4. Retorna melhor combinação
        
        Cross-validation (CV) explicado:
        - CV=5 significa dividir dados em 5 partes
        - Treino/validação shuffle 5 vezes
        - Score final é média das 5 iterações
        - Evita overfitting a dados específicos
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            grid_size: 'quick', 'medium', ou 'extensive'
            cv_folds: Número de CV folds (5 = padrão, 10 = mais rigoroso)
        
        Returns:
            best_model: Modelo XGBoost treinado com melhores parâmetros
            best_params: Dict com melhores parâmetros encontrados
            results_df: DataFrame com resultados de todas as combinações
        """
        
        start_time = time.time()
        
        # ---- OBTER GRID ----
        param_grid = self.get_param_grid(grid_size)
        
        # ---- BASE MODEL (parâmetros fixos) ----
        # Estes parâmetros não são tuned, mas mantemos invariantes:
        # - subsample e colsample_bytree = 0.8 (regularização)
        # - objective = 'reg:squarederror' (regressão)
        # - random_state = 42 (reproducibilidade)
        base_model = XGBRegressor(
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            verbosity=0,
            n_jobs=-1  # Usar todos os CPUs
        )
        
        # ---- GRIDSEARCH ----
        self._log(f"Iniciando GridSearchCV com {cv_folds} folds...")
        
        # Scoring: R² é melhor que MSE para séries temporais
        # Também rastreamos MAE
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='r2',  # Optimizar R²
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
            n_jobs=-1,  # Paralelize múltiplos CPUs
            verbose=1 if self.verbose else 0
        )
        
        # Executar search
        grid_search.fit(X_train, y_train)
        
        # ---- RESULTADOS ----
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        # Converter resultados para DataFrame (para análise)
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        elapsed_time = time.time() - start_time
        
        self._log(f"GridSearch completado em {elapsed_time:.1f}s")
        self._log(f"Melhor R² no CV: {self.best_score:.4f}")
        self._log(f"Melhor parâmetros: {self.best_params}")
        
        self.grid_results = results_df
        
        return self.best_model, self.best_params, results_df
    
    # ==================== VALIDAÇÃO E COMPARAÇÃO ====================
    
    def evaluate_on_test_set(self, X_test, y_test, baseline_model=None):
        """
        Avalia melhor modelo em conjunto de teste.
        
        Args:
            X_test: Features de teste
            y_test: Target de teste
            baseline_model: Modelo baseline (ex: original config) para comparação
        
        Returns:
            Dict com métricas de teste
        """
        
        if self.best_model is None:
            raise RuntimeError("Tuning não foi executado. Chamar tune_xgboost() primeiro.")
        
        # ---- PREDIÇÕES ----
        y_pred_tuned = self.best_model.predict(X_test)
        y_pred_tuned = np.maximum(y_pred_tuned, 0)  # Sem valores negativos
        
        # ---- MÉTRICAS ----
        metrics = {
            'r2_tuned': r2_score(y_test, y_pred_tuned),
            'mae_tuned': mean_absolute_error(y_test, y_pred_tuned),
            'rmse_tuned': np.sqrt(mean_squared_error(y_test, y_pred_tuned)),
            'mape_tuned': np.mean(np.abs((y_test - y_pred_tuned) / y_test[y_test != 0])) if np.any(y_test != 0) else np.inf
        }
        
        # ---- COMPARAÇÃO COM BASELINE ----
        if baseline_model is not None:
            y_pred_baseline = baseline_model.predict(X_test)
            y_pred_baseline = np.maximum(y_pred_baseline, 0)
            
            r2_baseline = r2_score(y_test, y_pred_baseline)
            mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
            
            # Ganho percentual
            metrics['r2_improvement'] = ((metrics['r2_tuned'] - r2_baseline) / abs(r2_baseline) * 100) if r2_baseline != 0 else np.inf
            metrics['mae_improvement'] = ((mae_baseline - metrics['mae_tuned']) / mae_baseline * 100)
            metrics['r2_baseline'] = r2_baseline
            metrics['mae_baseline'] = mae_baseline
        
        self._log(f"Métricas de teste - R²: {metrics['r2_tuned']:.4f}, MAE: {metrics['mae_tuned']:.2f}")
        
        return metrics
    
    # ==================== ANÁLISE DE RESULTADOS ====================
    
    def print_tuning_summary(self):
        """
        Imprime sumário legível dos resultados de tuning.
        """
        
        if self.grid_results is None or self.best_params is None:
            raise RuntimeError("Tuning não foi executado ainda.")
        
        print("\n" + "="*70)
        print("SUMÁRIO DE TUNING (GridSearch)")
        print("="*70)
        
        # ---- TOP 5 COMBINAÇÕES ----
        print("\nTop 5 Combinações (por CV R²):")
        print("-" * 70)
        
        # Filtrar colunas relevantes
        result_cols = [col for col in self.grid_results.columns 
                      if 'param' in col or 'mean_test' in col]
        top_results = self.grid_results.nlargest(5, 'mean_test_score')[result_cols]
        
        for idx, row in top_results.iterrows():
            params = {col.replace('param_', ''): row[col] 
                     for col in top_results.columns if 'param' in col}
            score = row['mean_test_score']
            
            print(f"\n  {idx+1}. R² = {score:.4f}")
            print(f"     Parâmetros: {params}")
        
        print(f"\n{'='*70}")
        print(f"MELHOR MODELO ENCONTRADO:")
        print(f"  R² (CV): {self.best_score:.4f}")
        print(f"  Parâmetros: {self.best_params}")
        print(f"{'='*70}\n")
    
    def get_feature_importance(self, feature_names=None):
        """
        Extrai importância de features do melhor modelo.
        
        Args:
            feature_names: Nomes das features (para label)
        
        Returns:
            Dict com importâncias ordenadas
        """
        
        if self.best_model is None:
            raise RuntimeError("Tuning não foi executado ainda.")
        
        importance = self.best_model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'F{i}' for i in range(len(importance))]
        
        importance_dict = {
            name: score for name, score in zip(feature_names, importance)
        }
        
        # Ordenar
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return importance_dict


if __name__ == '__main__':
    # TESTE: Criar dados dummy e fazer tuning
    print("Testando XGBoostTuner...")
    
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = 50 + 10*np.sin(X[:, 0]) + 3*X[:, 1] + np.random.normal(0, 3, n_samples)
    
    # Split
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Tuner
    tuner = XGBoostTuner(verbose=True)
    best_model, best_params, results = tuner.tune_xgboost(
        X_train, y_train, grid_size='quick', cv_folds=3
    )
    
    # Avaliar
    tuner.print_tuning_summary()
    
    # Baseline para comparação
    baseline = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42
    )
    baseline.fit(X_train, y_train)
    
    metrics = tuner.evaluate_on_test_set(X_test, y_test, baseline_model=baseline)
    print(f"\nMétricas de Teste:")
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.4f}")
