"""
TIAPOSE Retail Optimization - Entry Point Principal

Executa:
  - Treino: Treina todos os modelos de forecasting
        - Fase 1: Forecasting rápido (Seasonal Naive, ARIMA, ETS, XGBoost, ARIMAX)
  - Fase 2: Optimization (4 métodos × 3 objetivos)
  - Fase 3: DSS interativo para gerar planos

Uso:
  python main.py                  # Executa treino + fases 1-3
    python main.py --phase 1        # Apenas forecasting rápido
  python main.py --phase 2        # Apenas optimization (mock)
  python main.py --phase 3        # Apenas DSS demo
    python main.py --forecast-only  # Treino + comparação rápida de forecasting
  python main.py --dss            # DSS interativo
"""

import sys
import argparse
from datetime import datetime

# Imports
import sys
from pathlib import Path

# Ensure src is in path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from core.config import STORES
from training.train_all import train_all_models
from forecasting.rolling_window import evaluate_all_stores
from optimization.dss import DSS, DSSLoja


def print_header(title):
    """Imprime cabeçalho formatado."""
    print(f"\n{'='*70}")
    print(title.center(70))
    print(f"{'='*70}")


def phase_training():
    """Fase preliminar: treinar modelos (configuração padrão)."""
    print_header("FASE PRELIMINAR: TREINAMENTO DE MODELOS")
    train_all_models(verbose=True)
    print("\n[OK] Treinamento completo!")


def phase_1_forecasting():
    """Fase 1: Forecasting rápido sem Rolling Window."""
    print_header("FASE 1: FORECASTING RÁPIDO (SEM ROLLING WINDOW)")
    results = evaluate_all_stores()
    print("\n[OK] Fase 1 completa: Forecasting rápido avaliado")
    return results


def phase_2_optimization():
    """Fase 2: Optimization (demo com um store)."""
    print_header("FASE 2: OPTIMIZATION (DEMO)")
    
    from optimization.methods import OptimizationMethods
    import numpy as np
    
    store = STORES[3]  # Richmond
    forecast = np.array([100, 105, 110, 115, 120, 125, 130])
    
    print(f"\nDemo: Otimizando {store.upper()} com previsão de clientes")
    print(f"  Previsão: {forecast}")
    
    for objective in ['O1']:  # O1 é o mais rápido
        print(f"\n  Testando objetivo {objective}...")
        opt = OptimizationMethods(store, forecast, objective)
        result = opt.optimize()
        print(f"    Valor encontrado: {result['value']:.2f}")
    
    print("\n[OK] Fase 2 completa: Optimization testado")


def phase_3_dss_demo():
    """Fase 3: DSS demo (um plano de exemplo)."""
    print_header("FASE 3: DECISION SUPPORT SYSTEM (DEMO)")
    
    store = STORES[3]  # Richmond
    print(f"\nDemo: Gerando plano para {store.upper()}")
    
    dss = DSSLoja(store)
    result = dss.generate_plan('O1')
    
    print(result['summary'])
    print("\n" + result['plan_df'].to_string(index=False))
    
    print("\n[OK] Fase 3 completa: DSS testado")


def run_all_phases():
    """Executa todas as fases."""
    print_header("PROJETO TIAPOSE - RETAIL OPTIMIZATION")
    print("\nDado/hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Modo: Todas as fases")
    
    try:
        phase_training()
        phase_1_forecasting()
        phase_2_optimization()
        phase_3_dss_demo()
        
        print_header("RESUMO FINAL")
        print("\n[OK] Todas as fases executadas com sucesso!")
        print("\nPróximos passos:")
        print("  - Usar DSS interativo: python main.py --dss")
        print("  - Ver resultados em: ../reports/")
        print("  - Ver modelos em: ../models/")
        
    except Exception as e:
        print(f"\n[FAIL] Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


def run_dss_interactive():
    """Executa DSS em modo interativo."""
    print_header("SYSTEM DE SUPORTE À DECISÃO - MODO INTERATIVO")
    dss = DSS()
    dss.run_interactive()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TIAPOSE Retail Optimization System"
    )
    parser.add_argument(
        '--phase',
        choices=['1', '2', '3', 'all'],
        default='all',
        help='Qual fase executar (default: all)'
    )
    parser.add_argument(
        '--dss',
        action='store_true',
        help='Executar DSS em modo interativo'
    )
    parser.add_argument(
        '--forecast-only',
        action='store_true',
        help='Executar somente treino + comparação de forecasting'
    )
    
    args = parser.parse_args()
    
    if args.dss:
        run_dss_interactive()
    elif args.forecast_only:
        phase_training()
        phase_1_forecasting()
    elif args.phase == '1':
        print_header("FASE 1: FORECASTING RÁPIDO")
        phase_training()
        phase_1_forecasting()
    elif args.phase == '2':
        print_header("FASE 2: OPTIMIZATION")
        phase_2_optimization()
    elif args.phase == '3':
        print_header("FASE 3: DSS DEMO")
        phase_3_dss_demo()
    else:  # 'all'
        run_all_phases()


if __name__ == '__main__':
    main()
