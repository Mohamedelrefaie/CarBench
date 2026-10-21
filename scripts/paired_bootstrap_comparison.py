#!/usr/bin/env python3
"""
Paired Bootstrap Hypothesis Testing for Model Comparison
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu

Performs rigorous statistical comparison between models using:
1. Paired bootstrap resampling (preserves correlation structure)
2. Hypothesis testing for metric differences
3. Holm-Bonferroni correction for multiple comparisons
4. Effect size calculation (Cohen's d)

This ensures statistically sound claims about model performance differences.
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging


class PairedBootstrapTester:
    """Performs paired bootstrap hypothesis tests between models."""
    
    def __init__(self, bootstrap_dir: Path, alpha: float = 0.05, n_bootstrap: int = 2000):
        """
        Initialize the paired bootstrap tester.
        
        Args:
            bootstrap_dir: Directory containing bootstrap results
            alpha: Significance level (default: 0.05 for 95% confidence)
            n_bootstrap: Number of bootstrap iterations (should match saved results)
        """
        self.bootstrap_dir = Path(bootstrap_dir)
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.logger = logging.getLogger(__name__)
        
        # Load all model bootstrap results
        self.bootstrap_data = self._load_all_bootstrap_data()
        
    def _load_all_bootstrap_data(self) -> Dict[str, Dict]:
        """Load bootstrap data for all models."""
        bootstrap_data = {}
        
        # Find all model directories with bootstrap results
        # Try both possible directory structures
        search_patterns = [
            self.bootstrap_dir / '*/metrics/bootstrap',
            self.bootstrap_dir / '*/bootstrap'
        ]
        
        for pattern in search_patterns:
            for model_dir in pattern.parent.parent.glob(str(pattern).replace(str(pattern.parent.parent) + '/', '')):
                if not model_dir.exists():
                    continue
                    
                # Check for bootstrap_ci.json (contains bootstrap_indices)
                if (model_dir / 'bootstrap_ci.json').exists():
                    model_name = model_dir.parent.parent.name if 'metrics' in str(model_dir) else model_dir.parent.name
                    
                    # Load bootstrap data with indices
                    with open(model_dir / 'bootstrap_ci.json', 'r') as f:
                        data = json.load(f)
                    
                    # Also need to find checkpoint CSV for per-sample data
                    checkpoint_csv = None
                    possible_paths = [
                        # Standard paths
                        model_dir.parent / f"{model_name}_test_eval_checkpoint.csv",
                        model_dir / f"{model_name}_test_eval_checkpoint.csv",
                        model_dir.parent.parent / f"{model_name}_test_eval_checkpoint.csv",
                        # AB-UPT special case
                        self.bootstrap_dir / "surface-abupt-drivaernet" / "abupt_test_eval_checkpoint.csv",
                        # Generic checkpoint.csv
                        model_dir.parent / "checkpoint.csv",
                    ]
                    
                    for path in possible_paths:
                        if path.exists():
                            checkpoint_csv = path
                            break
                    
                    if checkpoint_csv:
                        checkpoint_df = pd.read_csv(checkpoint_csv)
                        data['checkpoint_data'] = checkpoint_df
                        data['format'] = 'json_with_indices'
                        bootstrap_data[model_name] = data
                        self.logger.info(f"Loaded bootstrap data for {model_name} (from bootstrap_ci.json)")
                    else:
                        self.logger.warning(f"Found bootstrap_ci.json for {model_name} but no checkpoint CSV")
                
                # Check for bootstrap_ci.csv (old format - need to load checkpoint CSV)
                elif (model_dir / 'bootstrap_ci.csv').exists():
                    model_name = model_dir.parent.parent.name if 'metrics' in str(model_dir) else model_dir.parent.name
                    
                    # Load bootstrap CI (just for metadata)
                    ci_df = pd.read_csv(model_dir / 'bootstrap_ci.csv')
                    
                    # Find checkpoint CSV with per-sample metrics
                    checkpoint_csv = None
                    possible_paths = [
                        model_dir.parent.parent / f"{model_name}_test_eval_checkpoint.csv",
                        model_dir.parent.parent / "checkpoint.csv",
                        model_dir / f"{model_name}_test_eval_checkpoint.csv",
                        self.bootstrap_dir / f"{model_name}_test_eval_checkpoint.csv",
                        self.bootstrap_dir / "abupt_test_eval_checkpoint.csv",  # AB-UPT special case
                    ]
                    
                    for path in possible_paths:
                        if path.exists():
                            checkpoint_csv = path
                            break
                    
                    if checkpoint_csv:
                        # Load checkpoint data
                        checkpoint_df = pd.read_csv(checkpoint_csv)
                        
                        # Create synthetic bootstrap data by resampling
                        # (This maintains the paired structure for comparison)
                        np.random.seed(42)  # For reproducibility
                        n_samples = len(checkpoint_df)
                        
                        # Generate bootstrap indices
                        bootstrap_indices = [
                            np.random.choice(n_samples, size=n_samples, replace=True).tolist()
                            for _ in range(self.n_bootstrap)
                        ]
                        
                        bootstrap_data[model_name] = {
                            'bootstrap_indices': bootstrap_indices,
                            'checkpoint_data': checkpoint_df,
                            'ci_data': ci_df,
                            'format': 'legacy'
                        }
                        self.logger.info(f"Loaded bootstrap data for {model_name} (legacy format, generated indices)")
                    else:
                        self.logger.warning(f"Could not find checkpoint CSV for {model_name}")
        
        if not bootstrap_data:
            raise ValueError(f"No bootstrap data found in {self.bootstrap_dir}")
        
        return bootstrap_data
    
    def _get_bootstrap_samples(self, model_name: str, metric_name: str) -> np.ndarray:
        """
        Get bootstrap samples for a specific model and metric.
        
        Handles both new format (with saved indices) and legacy format (generating samples on-the-fly).
        
        Args:
            model_name: Name of the model
            metric_name: Name of the metric
        
        Returns:
            Array of bootstrap samples
        """
        model_data = self.bootstrap_data[model_name]
        
        # Format 1: JSON with bootstrap_indices (from bootstrap_ci.json)
        if model_data.get('format') == 'json_with_indices':
            checkpoint_df = model_data['checkpoint_data']
            
            # Map metric name to JSON key
            metric_json_map = {
                'r2': 'r2',
                'r2_score': 'r2',
                'mae': 'mae',
                'rmse': 'rmse',
                'rel_l2_error': 'rel_l2',
                'rel_l2': 'rel_l2',
                'max_error': 'max_error',
                'mse': 'mse',
                'rel_l1': 'rel_l1',
            }
            
            json_key = metric_json_map.get(metric_name, metric_name)
            
            # Check if metric exists in JSON
            if json_key not in model_data:
                raise ValueError(f"Metric {metric_name} (JSON key: {json_key}) not found in bootstrap data for {model_name}. Available: {list(model_data.keys())}")
            
            # Get bootstrap indices for this specific metric
            bootstrap_indices = model_data[json_key]['bootstrap_indices']
            
            # Map metric name to checkpoint CSV column name
            metric_col_map = {
                'r2': 'r2_score',
                'mae': 'mae',
                'rmse': 'rmse',
                'rel_l2_error': 'rel_l2',
                'rel_l2': 'rel_l2',
                'max_error': 'max_error',
                'mse': 'mse',
                'rel_l1': 'rel_l1',
            }
            
            # Try to find the column - handle AB-UPT's _raw suffix
            col_name = metric_col_map.get(metric_name, metric_name)
            if col_name not in checkpoint_df.columns:
                col_name = f"{col_name}_raw"
            
            if col_name not in checkpoint_df.columns:
                # Try l2_error_raw and l1_error_raw for AB-UPT
                if metric_name == 'rel_l2_error' and 'l2_error_raw' in checkpoint_df.columns:
                    col_name = 'l2_error_raw'
                elif metric_name in ['rel_l2', 'rel_l2_error'] and 'l2_error_raw' in checkpoint_df.columns:
                    col_name = 'l2_error_raw'
                elif metric_name == 'rel_l1' and 'l1_error_raw' in checkpoint_df.columns:
                    col_name = 'l1_error_raw'
                elif metric_name not in checkpoint_df.columns:
                    raise ValueError(f"Column {metric_name} (tried {col_name}) not found in checkpoint data for {model_name}. Available: {checkpoint_df.columns.tolist()}")
                else:
                    col_name = metric_name
            
            # Compute bootstrap samples
            per_sample_values = checkpoint_df[col_name].values
            bootstrap_samples = []
            
            for indices in bootstrap_indices:
                resampled_values = per_sample_values[indices]
                bootstrap_samples.append(np.mean(resampled_values))
            
            return np.array(bootstrap_samples)
        
        # Format 2: Legacy CSV format (compute bootstrap samples from checkpoint data)
        if model_data.get('format') == 'legacy':
            checkpoint_df = model_data['checkpoint_data']
            bootstrap_indices = model_data['bootstrap_indices']
            
            # Map metric name to column name
            metric_col_map = {
                'r2': 'r2_score',
                'mae': 'mae',
                'rmse': 'rmse',
                'rel_l2_error': 'rel_l2',
                'max_error': 'max_error',
                'mse': 'mse',
                'rel_l1': 'rel_l1'
            }
            
            # Also try with _raw suffix for AB-UPT
            col_name = metric_col_map.get(metric_name, metric_name)
            if col_name not in checkpoint_df.columns:
                col_name = f"{col_name}_raw"
            
            if col_name not in checkpoint_df.columns:
                # Try original metric name
                if metric_name not in checkpoint_df.columns:
                    raise ValueError(f"Column {metric_name} (or {col_name}) not found in checkpoint data for {model_name}")
                col_name = metric_name
            
            # Compute bootstrap samples
            per_sample_values = checkpoint_df[col_name].values
            bootstrap_samples = []
            
            for indices in bootstrap_indices:
                resampled_values = per_sample_values[indices]
                bootstrap_samples.append(np.mean(resampled_values))
            
            return np.array(bootstrap_samples)
        
        raise ValueError(f"Cannot extract bootstrap samples for {model_name}, {metric_name}. Format: {model_data.get('format')}")
    
    def _paired_bootstrap_test(self, metric_name: str, model_a: str, model_b: str) -> Dict:
        """
        Perform paired bootstrap hypothesis test for a specific metric.
        
        Tests H0: metric(model_a) = metric(model_b)
        
        Args:
            metric_name: Name of metric to test (e.g., 'r2', 'mae')
            model_a: Name of first model
            model_b: Name of second model
        
        Returns:
            Dictionary with test results
        """
        # Get bootstrap samples for both models
        samples_a = self._get_bootstrap_samples(model_a, metric_name)
        samples_b = self._get_bootstrap_samples(model_b, metric_name)
        
        # Verify same number of bootstrap iterations
        if len(samples_a) != len(samples_b):
            raise ValueError(f"Bootstrap sample sizes don't match: {len(samples_a)} vs {len(samples_b)}")
        
        # Compute observed metrics (mean of bootstrap samples)
        metric_a = np.mean(samples_a)
        metric_b = np.mean(samples_b)
        observed_diff = metric_a - metric_b
        
        # Compute bootstrap distribution of differences
        # KEY: These are PAIRED differences (same bootstrap iteration)
        bootstrap_diffs = samples_a - samples_b
        
        # Compute confidence interval (percentile method)
        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha / 2))
        
        # Compute p-value (two-tailed test)
        # P-value = proportion of bootstrap samples where difference crosses zero
        if observed_diff >= 0:
            p_value = 2 * np.mean(bootstrap_diffs <= 0)
        else:
            p_value = 2 * np.mean(bootstrap_diffs >= 0)
        p_value = min(p_value, 1.0)  # Cap at 1.0
        
        # Effect size (standardized mean difference - Cohen's d)
        std_diff = np.std(bootstrap_diffs, ddof=1)
        effect_size = observed_diff / std_diff if std_diff > 0 else 0.0
        
        # Determine significance
        significant = p_value < self.alpha
        ci_excludes_zero = not (ci_lower <= 0 <= ci_upper)
        
        return {
            'model_a': model_a,
            'model_b': model_b,
            'metric': metric_name,
            'metric_a': metric_a,
            'metric_b': metric_b,
            'difference': observed_diff,
            'bootstrap_mean_diff': np.mean(bootstrap_diffs),
            'bootstrap_std_diff': std_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'significant': significant,
            'ci_excludes_zero': ci_excludes_zero,
            'effect_size': effect_size,
            'n_bootstrap': len(bootstrap_diffs),
            'alpha': self.alpha
        }
    
    def compare_all_pairs(self, metrics: List[str]) -> pd.DataFrame:
        """
        Perform pairwise comparisons between all models for specified metrics.
        
        Args:
            metrics: List of metric names to compare
        
        Returns:
            DataFrame with all pairwise test results
        """
        model_names = sorted(self.bootstrap_data.keys())
        results = []
        
        for metric in metrics:
            self.logger.info(f"\nTesting metric: {metric}")
            
            for i, model_a in enumerate(model_names):
                for j, model_b in enumerate(model_names):
                    if i >= j:  # Only upper triangle (avoid duplicates and self-comparison)
                        continue
                    
                    try:
                        result = self._paired_bootstrap_test(metric, model_a, model_b)
                        results.append(result)
                        
                        if result['significant']:
                            self.logger.info(
                                f"  {model_a} vs {model_b}: "
                                f"Δ={result['difference']:.4f}, "
                                f"p={result['p_value']:.4f} *"
                            )
                    except (KeyError, ValueError) as e:
                        self.logger.warning(f"  Skipping {model_a} vs {model_b}: {e}")
        
        return pd.DataFrame(results)
    
    def apply_holm_bonferroni_correction(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Holm-Bonferroni correction for multiple comparisons.
        
        The Holm-Bonferroni method is a step-down procedure that controls
        the family-wise error rate (FWER) while being more powerful than
        the standard Bonferroni correction.
        
        Args:
            results_df: DataFrame with pairwise test results
        
        Returns:
            DataFrame with corrected significance levels
        """
        df = results_df.copy()
        
        # Group by metric (apply correction separately for each metric)
        for metric in df['metric'].unique():
            mask = df['metric'] == metric
            metric_df = df[mask].copy()
            
            # Sort by p-value (ascending)
            metric_df = metric_df.sort_values('p_value')
            
            # Number of comparisons for this metric
            m = len(metric_df)
            
            # Apply Holm-Bonferroni correction
            holm_alpha = []
            holm_significant = []
            
            for i, (idx, row) in enumerate(metric_df.iterrows()):
                # Adjusted alpha for this comparison
                alpha_i = self.alpha / (m - i)
                holm_alpha.append(alpha_i)
                
                # Sequential testing: stop at first non-significant result
                if i == 0:
                    is_sig = row['p_value'] < alpha_i
                else:
                    # Only significant if all previous were significant
                    prev_all_sig = all(holm_significant)
                    is_sig = prev_all_sig and (row['p_value'] < alpha_i)
                
                holm_significant.append(is_sig)
            
            # Update dataframe
            df.loc[mask, 'holm_alpha'] = metric_df.index.map(
                dict(zip(metric_df.index, holm_alpha))
            )
            df.loc[mask, 'holm_significant'] = metric_df.index.map(
                dict(zip(metric_df.index, holm_significant))
            )
        
        return df
    
    def interpret_effect_size(self, effect_size: float) -> str:
        """
        Interpret effect size using Cohen's conventions.
        
        Args:
            effect_size: Standardized mean difference
        
        Returns:
            String interpretation
        """
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def create_significance_matrix(self, results_df: pd.DataFrame, metric: str,
                                   use_holm: bool = True) -> pd.DataFrame:
        """
        Create a significance matrix showing which model pairs differ significantly.
        
        Args:
            results_df: DataFrame with test results
            metric: Metric to visualize
            use_holm: Whether to use Holm-corrected p-values
        
        Returns:
            Square matrix with significance indicators
        """
        # Filter for specific metric
        metric_df = results_df[results_df['metric'] == metric].copy()
        
        # Get all unique model names
        models = sorted(set(metric_df['model_a'].unique()) | set(metric_df['model_b'].unique()))
        
        # Create empty matrix
        matrix = pd.DataFrame('', index=models, columns=models)
        
        sig_col = 'holm_significant' if use_holm else 'significant'
        
        for _, row in metric_df.iterrows():
            model_a = row['model_a']
            model_b = row['model_b']
            
            if row[sig_col]:
                # Determine which model is better
                if row['difference'] > 0:
                    # Model A is better
                    symbol = '+'  # A > B
                    matrix.loc[model_a, model_b] = symbol
                    matrix.loc[model_b, model_a] = '-'  # B < A
                else:
                    # Model B is better
                    symbol = '-'  # A < B
                    matrix.loc[model_a, model_b] = symbol
                    matrix.loc[model_b, model_a] = '+'  # B > A
            else:
                # No significant difference
                matrix.loc[model_a, model_b] = '≈'
                matrix.loc[model_b, model_a] = '≈'
        
        return matrix
    
    def plot_pairwise_differences(self, results_df: pd.DataFrame, metric: str,
                                  output_path: Path, use_holm: bool = True):
        """
        Create forest plot of pairwise differences with confidence intervals.
        
        Args:
            results_df: DataFrame with test results
            metric: Metric to visualize
            output_path: Path to save plot
            use_holm: Whether to use Holm-corrected significance
        """
        metric_df = results_df[results_df['metric'] == metric].copy()
        metric_df = metric_df.sort_values('difference', ascending=True)
        
        sig_col = 'holm_significant' if use_holm else 'significant'
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(metric_df) * 0.3)))
        
        for i, (_, row) in enumerate(metric_df.iterrows()):
            comparison = f"{row['model_a']} vs {row['model_b']}"
            color = 'darkred' if row[sig_col] else 'gray'
            alpha = 1.0 if row[sig_col] else 0.5
            
            # Plot confidence interval
            ax.plot([row['ci_lower'], row['ci_upper']], [i, i],
                   color=color, alpha=alpha, linewidth=2)
            
            # Plot point estimate
            ax.scatter(row['difference'], i, color=color, s=100, alpha=alpha, zorder=3)
            
            # Add p-value annotation
            p_str = f"p={row['p_value']:.3f}"
            if row[sig_col]:
                p_str += "*"
            ax.text(row['ci_upper'] + 0.001, i, p_str, va='center', fontsize=8)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Formatting
        ax.set_yticks(range(len(metric_df)))
        ax.set_yticklabels([f"{row['model_a']} vs {row['model_b']}" 
                           for _, row in metric_df.iterrows()])
        ax.set_xlabel(f'Difference in {metric.upper()} (with 95% CI)', fontsize=12)
        ax.set_title(f'Pairwise Model Comparisons: {metric.upper()}\n'
                    f'{"Holm-Bonferroni Corrected" if use_holm else "Uncorrected"}',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved forest plot to {output_path}")
    
    def create_summary_report(self, results_df: pd.DataFrame, output_dir: Path):
        """
        Create comprehensive summary report with all results.
        
        Args:
            results_df: DataFrame with test results (with Holm correction)
            output_dir: Directory to save reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results as CSV
        results_df.to_csv(output_dir / 'pairwise_bootstrap_tests.csv', index=False)
        
        # Create Markdown report
        report = self._generate_markdown_report(results_df)
        with open(output_dir / 'pairwise_comparison_report.md', 'w') as f:
            f.write(report)
        
        # Create LaTeX table
        latex_table = self._generate_latex_table(results_df)
        with open(output_dir / 'pairwise_comparison_table.tex', 'w') as f:
            f.write(latex_table)
        
        # Create significance matrices for each metric
        for metric in results_df['metric'].unique():
            # Uncorrected
            matrix = self.create_significance_matrix(results_df, metric, use_holm=False)
            matrix.to_csv(output_dir / f'{metric}_significance_matrix_uncorrected.csv')
            
            # Holm-corrected
            matrix_holm = self.create_significance_matrix(results_df, metric, use_holm=True)
            matrix_holm.to_csv(output_dir / f'{metric}_significance_matrix_holm.csv')
            
            # Plot forest plots
            self.plot_pairwise_differences(
                results_df, metric,
                output_dir / f'{metric}_forest_plot_uncorrected.png',
                use_holm=False
            )
            self.plot_pairwise_differences(
                results_df, metric,
                output_dir / f'{metric}_forest_plot_holm.png',
                use_holm=True
            )
        
        self.logger.info(f"Summary report saved to {output_dir}")
    
    def _generate_markdown_report(self, results_df: pd.DataFrame) -> str:
        """Generate Markdown report."""
        report = "# Paired Bootstrap Hypothesis Testing Report\n\n"
        report += f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Bootstrap Iterations:** {self.n_bootstrap}\n\n"
        report += f"**Significance Level (α):** {self.alpha}\n\n"
        report += f"**Multiple Comparison Correction:** Holm-Bonferroni\n\n"
        
        report += "## Summary Statistics\n\n"
        report += f"- Total comparisons: {len(results_df)}\n"
        report += f"- Significant (uncorrected): {results_df['significant'].sum()}\n"
        report += f"- Significant (Holm-corrected): {results_df['holm_significant'].sum()}\n\n"
        
        for metric in sorted(results_df['metric'].unique()):
            metric_df = results_df[results_df['metric'] == metric]
            
            report += f"## Metric: {metric.upper()}\n\n"
            
            # Holm-corrected significant differences
            if 'holm_significant' in metric_df.columns:
                sig_df = metric_df[metric_df['holm_significant']].sort_values('p_value')
            else:
                sig_df = pd.DataFrame()  # Empty if column doesn't exist
            
            if len(sig_df) > 0:
                report += "### Significant Differences (Holm-corrected)\n\n"
                report += "| Model A | Model B | Difference | 95% CI | p-value | Effect Size |\n"
                report += "|---------|---------|------------|--------|---------|-------------|\n"
                
                for _, row in sig_df.iterrows():
                    effect_interp = self.interpret_effect_size(row['effect_size'])
                    report += (
                        f"| {row['model_a']} | {row['model_b']} | "
                        f"{row['difference']:.4f} | "
                        f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | "
                        f"{row['p_value']:.4f} | "
                        f"{row['effect_size']:.3f} ({effect_interp}) |\n"
                    )
                report += "\n"
            else:
                report += "### No significant differences after Holm-Bonferroni correction\n\n"
            
            # Non-significant differences
            if 'holm_significant' in metric_df.columns:
                nonsig_mask = metric_df['holm_significant'] == False
                nonsig_df = metric_df.loc[nonsig_mask].sort_values('p_value')
                if len(nonsig_df) > 0:
                    report += "### Non-significant Differences\n\n"
                    report += f"Total: {len(nonsig_df)} comparisons\n\n"
        
        report += "\n## Interpretation Guide\n\n"
        report += "- **p-value < 0.05**: Strong evidence of genuine difference (reject H₀)\n"
        report += "- **p-value ≥ 0.05**: Insufficient evidence (fail to reject H₀)\n"
        report += "- **Effect Size**: Standardized difference (Cohen's d)\n"
        report += "  - Negligible: |d| < 0.2\n"
        report += "  - Small: 0.2 ≤ |d| < 0.5\n"
        report += "  - Medium: 0.5 ≤ |d| < 0.8\n"
        report += "  - Large: |d| ≥ 0.8\n\n"
        
        return report
    
    def _generate_latex_table(self, results_df: pd.DataFrame) -> str:
        """Generate LaTeX table."""
        latex = "% Pairwise Bootstrap Test Results\n"
        latex += "% Holm-Bonferroni corrected for multiple comparisons\n\n"
        
        for metric in sorted(results_df['metric'].unique()):
            metric_df = results_df[results_df['metric'] == metric]
            sig_df = metric_df[metric_df['holm_significant']].sort_values('p_value')
            
            if len(sig_df) == 0:
                continue
            
            latex += f"\\begin{{table}}[h]\n"
            latex += "\\centering\n"
            latex += f"\\caption{{Significant Pairwise Differences for {metric.upper()} "
            latex += "(Holm-Bonferroni Corrected)}}\n"
            latex += f"\\label{{tab:pairwise_{metric}}}\n"
            latex += "\\begin{tabular}{llcccc}\n"
            latex += "\\toprule\n"
            latex += "Model A & Model B & Difference & 95\\% CI & $p$-value & Effect Size \\\\\n"
            latex += "\\midrule\n"
            
            for _, row in sig_df.iterrows():
                p_str = f"{row['p_value']:.3f}"
                if row['p_value'] < 0.001:
                    p_str = "$<0.001$"
                    sig_marker = "***"
                elif row['p_value'] < 0.01:
                    sig_marker = "**"
                elif row['p_value'] < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = ""
                
                latex += (
                    f"{row['model_a']} & {row['model_b']} & "
                    f"{row['difference']:.4f} & "
                    f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] & "
                    f"{p_str} & {row['effect_size']:.3f} {sig_marker} \\\\\n"
                )
            
            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"
            latex += "\\end{table}\n\n"
        
        return latex


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Perform paired bootstrap hypothesis testing with Holm-Bonferroni correction'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Directory containing model results with bootstrap data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='statistical_comparison',
        help='Directory to save comparison results'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['r2', 'mae', 'rmse', 'rel_l2_error'],
        help='Metrics to compare'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=2000,
        help='Number of bootstrap iterations (should match saved results)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("PAIRED BOOTSTRAP HYPOTHESIS TESTING")
    logger.info("="*80)
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Metrics: {', '.join(args.metrics)}")
    logger.info(f"Significance level: {args.alpha}")
    logger.info(f"Bootstrap iterations: {args.n_bootstrap}")
    logger.info("="*80)
    
    # Initialize tester
    tester = PairedBootstrapTester(
        bootstrap_dir=Path(args.results_dir),
        alpha=args.alpha,
        n_bootstrap=args.n_bootstrap
    )
    
    # Perform pairwise comparisons
    logger.info("\nPerforming pairwise comparisons...")
    results_df = tester.compare_all_pairs(args.metrics)
    
    # Apply Holm-Bonferroni correction
    logger.info("\nApplying Holm-Bonferroni correction...")
    results_df = tester.apply_holm_bonferroni_correction(results_df)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total comparisons: {len(results_df)}")
    logger.info(f"Significant (uncorrected): {results_df['significant'].sum()}")
    logger.info(f"Significant (Holm-corrected): {results_df['holm_significant'].sum()}")
    
    for metric in sorted(results_df['metric'].unique()):
        metric_df = results_df[results_df['metric'] == metric]
        sig_uncorr = metric_df['significant'].sum()
        sig_holm = metric_df['holm_significant'].sum()
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  Uncorrected: {sig_uncorr}/{len(metric_df)} significant")
        logger.info(f"  Holm-corrected: {sig_holm}/{len(metric_df)} significant")
    
    # Create comprehensive report
    logger.info("\n" + "="*80)
    logger.info("Generating reports...")
    logger.info("="*80)
    tester.create_summary_report(results_df, Path(args.output_dir))
    
    logger.info("\n" + "="*80)
    logger.info("✓ COMPLETE!")
    logger.info("="*80)
    logger.info(f"Results saved to: {args.output_dir}/")
    logger.info(f"  - pairwise_bootstrap_tests.csv")
    logger.info(f"  - pairwise_comparison_report.md")
    logger.info(f"  - pairwise_comparison_table.tex")
    logger.info(f"  - *_significance_matrix_*.csv")
    logger.info(f"  - *_forest_plot_*.png")


if __name__ == "__main__":
    main()
