#!/usr/bin/env python3
"""
Generate comprehensive statistics and publication-quality figures for DrivAerNet++ dataset.
This script analyzes the pressure field data and creates visualizations for the paper.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyvista as pv
from scipy import stats
from tqdm import tqdm
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,  # TrueType fonts for editing
    'ps.fonttype': 42
})


class DrivAerNetAnalyzer:
    """Analyze DrivAerNet++ dataset and generate publication figures."""
    
    def __init__(self, dataset_path: str, subset_dir: str, output_dir: str = 'publication_figures'):
        self.dataset_path = Path(dataset_path)
        self.subset_dir = Path(subset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load subset information
        self.train_ids = self._load_ids('train_design_ids.txt')
        self.val_ids = self._load_ids('val_design_ids.txt')
        self.test_ids = self._load_ids('test_design_ids.txt')
        
        # Storage for statistics
        self.stats = {
            'train': {'pressures': [], 'designs': [], 'points_per_mesh': []},
            'val': {'pressures': [], 'designs': [], 'points_per_mesh': []},
            'test': {'pressures': [], 'designs': [], 'points_per_mesh': []}
        }
        
        # Design categories parsing
        self.design_categories = {
            'body_type': {'F': 'Fastback', 'N': 'Notchback', 'E': 'Estateback'},
            'underbody': {'S': 'Smooth', 'D': 'Detailed'},
            'wheels': {'WW': 'With Wheels', 'OW': 'Without Wheels'},
            'wheelhouses': {'S': 'Open', 'C': 'Covered'},
            'mirrors': {'WM': 'With Mirrors', 'OM': 'Without Mirrors'}
        }
    
    def _load_ids(self, filename: str) -> List[str]:
        """Load design IDs from file."""
        with open(self.subset_dir / filename, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _parse_design_name(self, filename: str) -> Dict[str, str]:
        """Parse design characteristics from filename."""
        # Example: F_S_WWS_WM_459.vtk
        parts = filename.replace('.vtk', '').split('_')
        if len(parts) >= 5:
            return {
                'body_type': self.design_categories['body_type'].get(parts[0], 'Unknown'),
                'underbody': self.design_categories['underbody'].get(parts[1], 'Unknown'),
                'wheels': 'With Wheels' if 'WW' in parts[2] else 'Without Wheels',
                'mirrors': 'With Mirrors' if 'WM' in parts[3] else 'Without Mirrors',
                'id': parts[-1]
            }
        return {}
    
    def collect_statistics(self, sample_fraction: float = 0.1):
        """Collect pressure statistics from the dataset."""
        print("Collecting dataset statistics...")
        
        all_files = list(self.dataset_path.glob('*.vtk'))
        
        # Process a sample of files for efficiency
        sample_size = int(len(all_files) * sample_fraction)
        sampled_files = np.random.choice(all_files, size=min(sample_size, len(all_files)), replace=False)
        
        for vtk_file in tqdm(sampled_files, desc="Processing VTK files"):
            filename = vtk_file.name
            
            # Determine which split this belongs to
            split = None
            if any(id_ in filename for id_ in self.train_ids):
                split = 'train'
            elif any(id_ in filename for id_ in self.val_ids):
                split = 'val'
            elif any(id_ in filename for id_ in self.test_ids):
                split = 'test'
            
            if split:
                try:
                    mesh = pv.read(vtk_file)
                    pressures = mesh.point_data['p']
                    
                    # Sample pressures for memory efficiency
                    if len(pressures) > 10000:
                        sampled_pressures = np.random.choice(pressures, size=10000, replace=False)
                    else:
                        sampled_pressures = pressures
                    
                    self.stats[split]['pressures'].extend(sampled_pressures)
                    self.stats[split]['designs'].append(self._parse_design_name(filename))
                    self.stats[split]['points_per_mesh'].append(len(pressures))
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Convert to numpy arrays
        for split in ['train', 'val', 'test']:
            self.stats[split]['pressures'] = np.array(self.stats[split]['pressures'])
    
    def generate_pressure_distribution_figure(self):
        """Generate comprehensive pressure distribution analysis figure."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DrivAerNet++ Pressure Distribution Analysis', fontsize=18, y=1.02)
        
        # 1a. Overall pressure distribution with KDE
        all_pressures = np.concatenate([self.stats[split]['pressures'] for split in ['train', 'val', 'test']])
        
        axes[0, 0].hist(all_pressures, bins=100, density=True, alpha=0.7, 
                       color='#1f77b4', edgecolor='black', linewidth=0.5)
        
        # Add KDE
        kde = stats.gaussian_kde(all_pressures)
        x_range = np.linspace(all_pressures.min(), all_pressures.max(), 1000)
        axes[0, 0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add statistics
        mean_p = np.mean(all_pressures)
        std_p = np.std(all_pressures)
        median_p = np.median(all_pressures)
        skew_p = stats.skew(all_pressures)
        
        stats_text = f'μ = {mean_p:.1f} Pa\nσ = {std_p:.1f} Pa\nMedian = {median_p:.1f} Pa\nSkew = {skew_p:.2f}'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[0, 0].set_xlabel('Pressure (Pa)')
        axes[0, 0].set_ylabel('Probability Density')
        axes[0, 0].set_title('Global Pressure Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 1b. Split-wise comparison
        split_data = [self.stats[split]['pressures'] for split in ['train', 'val', 'test']]
        split_labels = ['Train', 'Validation', 'Test']
        
        bp = axes[0, 1].boxplot(split_data, labels=split_labels, patch_artist=True,
                               notch=True, showmeans=True)
        
        # Color the boxes
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0, 1].set_ylabel('Pressure (Pa)')
        axes[0, 1].set_title('Pressure Distribution by Dataset Split')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 1c. Log-scale pressure magnitude distribution
        abs_pressures = np.abs(all_pressures)
        abs_pressures = abs_pressures[abs_pressures > 0]  # Remove zeros for log scale
        
        axes[1, 0].hist(abs_pressures, bins=np.logspace(0, np.log10(abs_pressures.max()), 50),
                       alpha=0.7, color='#2ca02c', edgecolor='black')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_xlabel('|Pressure| (Pa)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Pressure Magnitude Distribution (Log Scale)')
        axes[1, 0].grid(True, alpha=0.3, which='both')
        
        # 1d. Percentile plot
        percentiles = np.arange(0, 101, 1)
        percentile_values = np.percentile(all_pressures, percentiles)
        
        axes[1, 1].plot(percentiles, percentile_values, 'b-', linewidth=2)
        axes[1, 1].fill_between(percentiles, percentile_values, alpha=0.3)
        axes[1, 1].set_xlabel('Percentile')
        axes[1, 1].set_ylabel('Pressure (Pa)')
        axes[1, 1].set_title('Pressure Percentile Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Mark key percentiles
        key_percentiles = [5, 25, 50, 75, 95]
        for p in key_percentiles:
            val = np.percentile(all_pressures, p)
            axes[1, 1].axhline(y=val, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].text(p, val, f'P{p}: {val:.0f}', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pressure_distribution_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'pressure_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_design_diversity_figure(self):
        """Generate figure showing design category distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('DrivAerNet++ Design Diversity', fontsize=18, y=1.02)
        
        # Count design categories
        all_designs = []
        for split in ['train', 'val', 'test']:
            all_designs.extend(self.stats[split]['designs'])
        
        # 2a. Body type distribution
        body_types = [d.get('body_type', 'Unknown') for d in all_designs]
        body_counts = pd.Series(body_types).value_counts()
        
        wedges, texts, autotexts = axes[0, 0].pie(body_counts.values, labels=body_counts.index,
                                                   autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Body Type Distribution')
        
        # 2b. Feature combinations
        feature_combinations = []
        for d in all_designs:
            combo = f"{d.get('wheels', 'Unknown')}+{d.get('mirrors', 'Unknown')}"
            feature_combinations.append(combo)
        
        combo_counts = pd.Series(feature_combinations).value_counts()
        axes[0, 1].bar(range(len(combo_counts)), combo_counts.values)
        axes[0, 1].set_xticks(range(len(combo_counts)))
        axes[0, 1].set_xticklabels(combo_counts.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Feature Combination Distribution')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 2c. Points per mesh distribution
        all_points = []
        for split in ['train', 'val', 'test']:
            all_points.extend(self.stats[split]['points_per_mesh'])
        
        axes[1, 0].hist(all_points, bins=30, alpha=0.7, color='#9467bd', edgecolor='black')
        axes[1, 0].set_xlabel('Number of Points')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Mesh Resolution Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_points = np.mean(all_points)
        axes[1, 0].axvline(x=mean_points, color='r', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_points:.0f}')
        axes[1, 0].legend()
        
        # 2d. Split-wise design distribution
        split_counts = {
            'Train': len(self.train_ids),
            'Validation': len(self.val_ids),
            'Test': len(self.test_ids)
        }
        
        bars = axes[1, 1].bar(split_counts.keys(), split_counts.values(), 
                             color=['#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
        axes[1, 1].set_ylabel('Number of Designs')
        axes[1, 1].set_title('Dataset Split Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'design_diversity_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'design_diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_statistical_summary_table(self):
        """Generate LaTeX table with dataset statistics."""
        summary_data = []
        
        for split in ['train', 'val', 'test']:
            pressures = self.stats[split]['pressures']
            if len(pressures) > 0:
                summary_data.append({
                    'Split': split.capitalize(),
                    'Designs': len(self.stats[split]['designs']),
                    'Total Points': f"{len(pressures)/1e6:.1f}M",
                    'Pressure Range': f"[{pressures.min():.0f}, {pressures.max():.0f}]",
                    'Mean ± Std': f"{pressures.mean():.1f} ± {pressures.std():.1f}",
                    'Median': f"{np.median(pressures):.1f}",
                    'Skewness': f"{stats.skew(pressures):.2f}"
                })
        
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        df.to_csv(self.output_dir / 'dataset_statistics.csv', index=False)
        
        # Generate LaTeX table
        latex_table = df.to_latex(index=False, escape=False)
        
        # Enhance LaTeX formatting
        latex_table = latex_table.replace('\\toprule', '\\toprule\n\\midrule')
        latex_table = latex_table.replace('\\bottomrule', '\\midrule\n\\bottomrule')
        
        with open(self.output_dir / 'dataset_statistics.tex', 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{DrivAerNet++ Dataset Statistics}\n")
            f.write("\\label{tab:dataset_stats}\n")
            f.write(latex_table)
            f.write("\\end{table}\n")
        
        return df
    
    def generate_model_performance_comparison(self, results_csv_path: str = None):
        """Generate model performance comparison figures if results are available."""
        if results_csv_path is None or not os.path.exists(results_csv_path):
            print("Model results not found, skipping performance comparison")
            return
        
        df = pd.read_csv(results_csv_path)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison on DrivAerNet++', fontsize=18, y=1.02)
        
        # Sort by R² score
        df = df.sort_values('test_r2', ascending=False)
        
        # 3a. R² scores
        axes[0, 0].bar(df['model_name'], df['test_r2'], color='#1f77b4', alpha=0.7)
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model Accuracy (R² Score)')
        axes[0, 0].set_xticklabels(df['model_name'], rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim(0, 1)
        
        # 3b. Relative L2 Error
        axes[0, 1].bar(df['model_name'], df['test_rel_l2'], color='#ff7f0e', alpha=0.7)
        axes[0, 1].set_ylabel('Relative L2 Error')
        axes[0, 1].set_title('Relative L2 Error (Lower is Better)')
        axes[0, 1].set_xticklabels(df['model_name'], rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3c. Inference time vs accuracy
        axes[0, 2].scatter(df['mean_inference_time_ms'], df['test_r2'], s=100, alpha=0.7)
        for idx, row in df.iterrows():
            axes[0, 2].annotate(row['model_name'], 
                              (row['mean_inference_time_ms'], row['test_r2']),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 2].set_xlabel('Inference Time (ms)')
        axes[0, 2].set_ylabel('R² Score')
        axes[0, 2].set_title('Speed vs Accuracy Trade-off')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 3d. MAE comparison
        axes[1, 0].bar(df['model_name'], df['test_mae'], color='#2ca02c', alpha=0.7)
        axes[1, 0].set_ylabel('MAE (Pa)')
        axes[1, 0].set_title('Mean Absolute Error')
        axes[1, 0].set_xticklabels(df['model_name'], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 3e. Parameters vs Performance
        axes[1, 1].scatter(df['total_parameters']/1e6, df['test_r2'], s=100, alpha=0.7)
        for idx, row in df.iterrows():
            axes[1, 1].annotate(row['model_name'], 
                              (row['total_parameters']/1e6, row['test_r2']),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[1, 1].set_xlabel('Parameters (Millions)')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('Model Size vs Performance')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xscale('log')
        
        # 3f. Efficiency score (R²/parameters)
        efficiency = df['test_r2'] / (df['total_parameters'] / 1e6)
        axes[1, 2].bar(df['model_name'], efficiency, color='#9467bd', alpha=0.7)
        axes[1, 2].set_ylabel('Efficiency (R²/M parameters)')
        axes[1, 2].set_title('Model Efficiency')
        axes[1, 2].set_xticklabels(df['model_name'], rotation=45, ha='right')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_error_distribution_analysis(self):
        """Generate error distribution analysis if sample predictions are available."""
        # This would require actual model predictions
        # For now, we'll create a template
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Pressure-Dependent Error Analysis', fontsize=16)
        
        # Placeholder for when actual predictions are available
        axes[0].text(0.5, 0.5, 'Requires Model Predictions\n(Placeholder)', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_xlabel('Ground Truth Pressure (Pa)')
        axes[0].set_ylabel('Prediction Error (Pa)')
        axes[0].set_title('Error vs Pressure Magnitude')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].text(0.5, 0.5, 'Requires Model Predictions\n(Placeholder)', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_xlabel('Spatial Region')
        axes[1].set_ylabel('Average Error (Pa)')
        axes[1].set_title('Spatial Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'error_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_statistics(self):
        """Save comprehensive statistics to JSON."""
        comprehensive_stats = {}
        
        for split in ['train', 'val', 'test']:
            pressures = self.stats[split]['pressures']
            if len(pressures) > 0:
                comprehensive_stats[split] = {
                    'num_designs': len(self.stats[split]['designs']),
                    'total_points': int(len(pressures)),
                    'pressure_statistics': {
                        'min': float(pressures.min()),
                        'max': float(pressures.max()),
                        'mean': float(pressures.mean()),
                        'std': float(pressures.std()),
                        'median': float(np.median(pressures)),
                        'skewness': float(stats.skew(pressures)),
                        'kurtosis': float(stats.kurtosis(pressures)),
                        'percentiles': {
                            '1%': float(np.percentile(pressures, 1)),
                            '5%': float(np.percentile(pressures, 5)),
                            '10%': float(np.percentile(pressures, 10)),
                            '25%': float(np.percentile(pressures, 25)),
                            '50%': float(np.percentile(pressures, 50)),
                            '75%': float(np.percentile(pressures, 75)),
                            '90%': float(np.percentile(pressures, 90)),
                            '95%': float(np.percentile(pressures, 95)),
                            '99%': float(np.percentile(pressures, 99))
                        }
                    },
                    'mesh_statistics': {
                        'points_per_mesh': {
                            'min': int(np.min(self.stats[split]['points_per_mesh'])),
                            'max': int(np.max(self.stats[split]['points_per_mesh'])),
                            'mean': float(np.mean(self.stats[split]['points_per_mesh'])),
                            'std': float(np.std(self.stats[split]['points_per_mesh']))
                        }
                    }
                }
        
        with open(self.output_dir / 'drivaernet_comprehensive_statistics.json', 'w') as f:
            json.dump(comprehensive_stats, f, indent=2)
        
        return comprehensive_stats


def main():
    parser = argparse.ArgumentParser(description='Generate DrivAerNet++ statistics and figures')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to DrivAerNet++ VTK files')
    parser.add_argument('--subset-dir', type=str, required=True,
                       help='Path to train/val/test split files')
    parser.add_argument('--output-dir', type=str, default='publication_figures',
                       help='Output directory for figures')
    parser.add_argument('--sample-fraction', type=float, default=0.1,
                       help='Fraction of dataset to sample for statistics (0.1 = 10%)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = DrivAerNetAnalyzer(args.dataset_path, args.subset_dir, args.output_dir)
    
    # Collect statistics
    analyzer.collect_statistics(sample_fraction=args.sample_fraction)
    
    # Generate all figures
    print("Generating pressure distribution analysis...")
    analyzer.generate_pressure_distribution_figure()
    
    print("Generating design diversity analysis...")
    analyzer.generate_design_diversity_figure()
    
    print("Generating statistical summary table...")
    summary_df = analyzer.generate_statistical_summary_table()
    print("\nDataset Summary:")
    print(summary_df)
    
    print("Saving comprehensive statistics...")
    stats = analyzer.save_comprehensive_statistics()
    
    print(f"\nAll figures and statistics saved to: {analyzer.output_dir}")
    
    # Print key statistics
    print("\nKey Statistics:")
    for split in ['train', 'val', 'test']:
        if split in stats:
            print(f"\n{split.upper()}:")
            print(f"  Designs: {stats[split]['num_designs']}")
            print(f"  Pressure range: [{stats[split]['pressure_statistics']['min']:.1f}, "
                  f"{stats[split]['pressure_statistics']['max']:.1f}] Pa")
            print(f"  Mean ± Std: {stats[split]['pressure_statistics']['mean']:.1f} ± "
                  f"{stats[split]['pressure_statistics']['std']:.1f} Pa")


if __name__ == "__main__":
    main()