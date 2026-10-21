# benchmark/cross_category_evaluator.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Cross-category generalization evaluation for automotive aerodynamics.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

class CrossCategoryEvaluator:
    """Evaluate cross-category generalization performance."""
    
    def __init__(self, base_results_dir: str = 'results'):
        self.base_results_dir = base_results_dir
        self.categories = ['sedan', 'suv', 'wagon', 'hatchback']
        
    def run_cross_category_benchmark(
        self,
        model_name: str,
        dataset_path: str,
        category_splits_dir: str,
        config_path: str
    ):
        """Run LOCO evaluation for a model."""
        
        results = {}
        
        for test_category in self.categories:
            print(f"\n{'='*60}")
            print(f"Testing generalization to: {test_category.upper()}")
            print(f"Training on: {[c for c in self.categories if c != test_category]}")
            print('='*60)
            
            # Run benchmark with category-specific split
            split_dir = os.path.join(category_splits_dir, f"loco_{test_category}")
            
            cmd = f"""
            CUDA_VISIBLE_DEVICES=0,3 python scripts/run_single_model.py {model_name} \
                --dataset-path {dataset_path} \
                --subset-dir {split_dir} \
                --save-dir {self.base_results_dir}/loco_{test_category}
            """
            
            os.system(cmd)
            
            # Load results
            results_file = f"{self.base_results_dir}/loco_{test_category}/{model_name}/*_results_*.json"
            with open(results_file, 'r') as f:
                results[test_category] = json.load(f)
        
        # Analyze cross-category performance
        self.analyze_results(model_name, results)
        
        return results
    
    def analyze_results(self, model_name: str, results: Dict[str, Any]):
        """Analyze and visualize cross-category results."""
        
        # Create comparison DataFrame
        metrics_data = []
        
        for test_cat, result in results.items():
            train_cats = [c for c in self.categories if c != test_cat]
            
            metrics_data.append({
                'Test Category': test_cat,
                'Train Categories': ', '.join(train_cats),
                'Test MSE': result['accuracy_metrics']['mse'],
                'Test MAE': result['accuracy_metrics']['mae'],
                'Test R²': result['accuracy_metrics']['r2'],
                'Rel L2': result['accuracy_metrics'].get('rel_l2', 0),
                'Rel L1': result['accuracy_metrics'].get('rel_l1', 0),
                'Inference Time (ms)': result['inference_metrics']['mean_inference_time_ms']
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Save summary
        summary_path = f"{self.base_results_dir}/cross_category_summary_{model_name}.csv"
        df.to_csv(summary_path, index=False)
        
        # Visualizations
        self.create_visualizations(model_name, df)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"CROSS-CATEGORY GENERALIZATION SUMMARY - {model_name}")
        print('='*60)
        print(df.to_string(index=False))
        
        # Compute generalization gap
        avg_mse = df['Test MSE'].mean()
        std_mse = df['Test MSE'].std()
        worst_case = df.loc[df['Test MSE'].idxmax()]
        best_case = df.loc[df['Test MSE'].idxmin()]
        
        print(f"\nGeneralization Statistics:")
        print(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
        print(f"Best generalization: {best_case['Test Category']} (MSE: {best_case['Test MSE']:.6f})")
        print(f"Worst generalization: {worst_case['Test Category']} (MSE: {worst_case['Test MSE']:.6f})")
        print(f"Generalization gap: {worst_case['Test MSE'] - best_case['Test MSE']:.6f}")
        
    def create_visualizations(self, model_name: str, df: pd.DataFrame):
        """Create cross-category performance visualizations."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. MSE comparison
        ax = axes[0, 0]
        df.plot(x='Test Category', y='Test MSE', kind='bar', ax=ax)
        ax.set_title('MSE by Test Category')
        ax.set_ylabel('MSE')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. R² comparison
        ax = axes[0, 1]
        df.plot(x='Test Category', y='Test R²', kind='bar', ax=ax, color='green')
        ax.set_title('R² by Test Category')
        ax.set_ylabel('R²')
        ax.set_ylim([0.9, 1.0])  # Adjust based on your data
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Relative errors
        ax = axes[1, 0]
        df[['Test Category', 'Rel L1', 'Rel L2']].set_index('Test Category').plot(kind='bar', ax=ax)
        ax.set_title('Relative Errors by Test Category')
        ax.set_ylabel('Relative Error')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(['Rel L1', 'Rel L2'])
        
        # 4. Heatmap of all metrics
        ax = axes[1, 1]
        metrics_matrix = df[['Test MSE', 'Test MAE', 'Test R²', 'Rel L2']].T
        metrics_matrix.columns = df['Test Category']
        
        # Normalize for better visualization
        normalized = (metrics_matrix - metrics_matrix.mean(axis=1).values.reshape(-1, 1)) / metrics_matrix.std(axis=1).values.reshape(-1, 1)
        
        sns.heatmap(normalized, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax)
        ax.set_title('Normalized Metrics Heatmap')
        
        plt.tight_layout()
        plt.savefig(f"{self.base_results_dir}/cross_category_analysis_{model_name}.png", dpi=300)
        plt.close()
        
        # Create generalization matrix
        self.create_generalization_matrix(model_name, df)
    
    def create_generalization_matrix(self, model_name: str, df: pd.DataFrame):
        """Create a matrix showing train->test generalization."""
        
        # Create 4x4 matrix (train categories vs test category)
        matrix = np.zeros((4, 4))
        
        for i, test_cat in enumerate(self.categories):
            row = df[df['Test Category'] == test_cat]
            if not row.empty:
                # Diagonal = 0 (not tested)
                matrix[i, i] = np.nan
                # Off-diagonal = generalization performance
                for j, train_cat in enumerate(self.categories):
                    if i != j:
                        matrix[j, i] = row['Test MSE'].values[0]
        
        # Visualize
        plt.figure(figsize=(8, 6))
        mask = np.isnan(matrix)
        sns.heatmap(matrix, 
                    xticklabels=self.categories,
                    yticklabels=[f"Train without {c}" for c in self.categories],
                    annot=True, 
                    fmt='.4f',
                    cmap='RdYlGn_r',
                    mask=mask,
                    cbar_kws={'label': 'Test MSE'})
        
        plt.title(f'Cross-Category Generalization Matrix - {model_name}')
        plt.xlabel('Test Category')
        plt.ylabel('Training Set')
        plt.tight_layout()
        plt.savefig(f"{self.base_results_dir}/generalization_matrix_{model_name}.png", dpi=300)
        plt.close()