#!/usr/bin/env python3
"""Extract and analyze cross-category generalization results"""

import pandas as pd
from pathlib import Path
import sys

def main():
    base_dir = Path('results/cross_category_generalization')
    
    # Define experiment mappings - order matters for table
    experiments = {
        # Two-category training (train on 2, test on 1)
        'train_NF_test_E': {'train': 'Fastbacks + Notchbacks', 'test': 'Estatebacks'},
        'train_EF_test_N': {'train': 'Estatebacks + Fastbacks', 'test': 'Notchbacks'},
        'train_EN_test_F': {'train': 'Estatebacks + Notchbacks', 'test': 'Fastbacks'},
        # Single-category training (train on 1, test on 2)
        'train_E_test_NF': {'train': 'Estatebacks', 'test': 'Fastbacks + Notchbacks'},
        'train_N_test_EF': {'train': 'Notchbacks', 'test': 'Estatebacks + Fastbacks'},
        'train_F_test_EN': {'train': 'Fastbacks', 'test': 'Estatebacks + Notchbacks'},
    }
    
    results = {}
    
    print("\n" + "="*80)
    print("CROSS-CATEGORY GENERALIZATION RESULTS")
    print("="*80)
    
    for model in ['transolver', 'transolver_plus']:
        results[model] = {}
        print(f"\n{model.upper()}:")
        print("-"*80)
        
        for exp_name, exp_info in experiments.items():
            csv_file = base_dir / model / exp_name / 'comprehensive_benchmark_summary.csv'
            
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                # Divide by 10000 to get actual number of car designs (not total points)
                train_samples = int(df['train_num_samples'].values[0]) // 10000
                test_samples = int(df['test_num_samples'].values[0]) // 10000
                
                results[model][exp_name] = {
                    'train_cat': exp_info['train'],
                    'test_cat': exp_info['test'],
                    'rel_l2': float(df['test_rel_l2'].values[0]),
                    'r2': float(df['test_r2'].values[0]),
                    'mae': float(df['test_mae'].values[0]),
                    'rmse': float(df['test_rmse'].values[0]),
                    'train_size': train_samples,
                    'test_size': test_samples,
                }
                r = results[model][exp_name]
                ratio = r['train_size'] / r['test_size']
                print(f"  ✅ {exp_name}: Rel L2={r['rel_l2']:.4f}, R²={r['r2']:.4f}, "
                      f"Train={r['train_size']}, Test={r['test_size']}, Ratio={ratio:.2f}")
            else:
                results[model][exp_name] = None
                print(f"  ❌ {exp_name}: NOT COMPLETED")
    
    # Generate LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE ROWS")
    print("="*80 + "\n")
    
    for exp_name, exp_info in experiments.items():
        # Get results from both models
        t_res = results['transolver'].get(exp_name)
        tp_res = results['transolver_plus'].get(exp_name)
        
        # Use first available model for train/test sizes
        ref = t_res or tp_res
        
        if ref:
            train_size = f"{ref['train_size']:,}".replace(",", "{,}")
            test_size = f"{ref['test_size']:,}".replace(",", "{,}")
            ratio = ref['train_size'] / ref['test_size']
            
            # Format metrics - use "xxx" for missing values
            t_rel_l2 = f"{t_res['rel_l2']:.4f}" if t_res else "xxx"
            tp_rel_l2 = f"{tp_res['rel_l2']:.4f}" if tp_res else "xxx"
            t_r2 = f"{t_res['r2']:.4f}" if t_res else "xxx"
            tp_r2 = f"{tp_res['r2']:.4f}" if tp_res else "xxx"
        else:
            train_size = "xxx"
            test_size = "xxx"
            ratio = 0
            t_rel_l2 = tp_rel_l2 = t_r2 = tp_r2 = "xxx"
        
        # Build row
        train_cat = ref['train_cat'] if ref else exp_info['train']
        test_cat = ref['test_cat'] if ref else exp_info['test']
        
        row = f"{train_cat:<35} & {test_cat:<35} & {t_rel_l2:>6} & {tp_rel_l2:>6} & {t_r2:>6} & {tp_r2:>6} & {train_size:>7} & {test_size:>7} & {ratio:.2f} \\\\"
        
        print(row)
        
        # Add hline after two-category group
        if exp_name == 'train_EN_test_F':
            print("\\hline")
            print("\\\\[-2ex]")

if __name__ == '__main__':
    main()