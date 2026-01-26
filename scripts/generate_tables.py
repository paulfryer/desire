#!/usr/bin/env python3
"""
Generate publication-ready LaTeX tables from test results.

Outputs:
- Table 1: Main results summary
- Table 2: Statistical comparisons
- Table 3: Effect sizes
- Table S1: Full predicate results
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load test results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def compute_stats(scores: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of scores."""
    if not scores:
        return {'mean': 0, 'std': 0, 'ci_low': 0, 'ci_high': 0, 'n': 0}

    n = len(scores)
    mean = np.mean(scores)
    std = np.std(scores, ddof=1) if n > 1 else 0

    if n > 1:
        ci = stats.sem(scores) * stats.t.ppf(0.975, n - 1)
        ci_low = mean - ci
        ci_high = mean + ci
    else:
        ci_low = ci_high = mean

    return {
        'mean': mean,
        'std': std,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n': n,
    }


def compute_cohens_d(g1: List[float], g2: List[float]) -> Optional[float]:
    """Compute Cohen's d effect size."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return None

    mean1, mean2 = np.mean(g1), np.mean(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return None

    return (mean1 - mean2) / pooled_std


def format_number(value: float, precision: int = 3) -> str:
    """Format number for LaTeX output."""
    if value is None:
        return '--'
    return f'{value:.{precision}f}'


def format_pvalue(p: float) -> str:
    """Format p-value for LaTeX output."""
    if p < 0.001:
        return '$<$0.001'
    elif p < 0.01:
        return f'{p:.3f}'
    else:
        return f'{p:.2f}'


def generate_table1_main_results(
    results: Dict[str, Any],
    output_path: Path,
) -> str:
    """
    Generate Table 1: Main results summary.

    Format: Agent | Predicate1 | Predicate2 | ... | Overall
    """
    test_results = results.get('results', {})

    # Get all agents and predicates
    all_agents = set()
    for test_data in test_results.values():
        all_agents.update(test_data.keys())
    agents = sorted(all_agents)
    predicates = sorted(test_results.keys())

    # Agent display names
    agent_names = {
        'heuristic': 'Heuristic',
        'rl_ppo': 'PPO',
        'active_inference': 'Active Inference',
        'rl_dqn': 'DQN',
        'model_based': 'Model-Based',
    }

    # Predicate abbreviations
    pred_abbrev = {
        'anticipatory_activation': 'AA',
        'frustration_intensification': 'FI',
        'means_end_decoupling': 'MED',
        'extinction_resistance': 'ER',
        'opportunity_cost_sensitivity': 'OCS',
        'state_dependent_valuation': 'SDV',
        'information_seeking': 'IS',
        'latent_learning': 'LL',
        'spontaneous_switching': 'SS',
    }

    # Build LaTeX table
    n_cols = len(predicates) + 2  # Agent + predicates + Overall
    col_format = 'l' + 'c' * (n_cols - 1)

    latex = []
    latex.append('\\begin{table}[htbp]')
    latex.append('\\centering')
    latex.append('\\caption{Main Results: Mean Scores by Agent and Predicate Test}')
    latex.append('\\label{tab:main_results}')
    latex.append(f'\\begin{{tabular}}{{{col_format}}}')
    latex.append('\\toprule')

    # Header
    header = ['Agent']
    for pred in predicates[:6]:  # Limit to 6 predicates for space
        abbrev = pred_abbrev.get(pred, pred[:3].upper())
        header.append(abbrev)
    header.append('Overall')
    latex.append(' & '.join(header) + ' \\\\')
    latex.append('\\midrule')

    # Data rows
    for agent in agents:
        row = [agent_names.get(agent, agent)]
        all_scores = []

        for pred in predicates[:6]:
            if agent in test_results[pred]:
                raw = test_results[pred][agent].get('raw_data', {})
                scores = raw.get('trial_scores', [test_results[pred][agent].get('score', 0)])
                mean = np.mean(scores)
                std = np.std(scores, ddof=1) if len(scores) > 1 else 0
                row.append(f'{mean:.2f} \\tiny{{$\\pm${std:.2f}}}')
                all_scores.extend(scores)
            else:
                row.append('--')

        # Overall
        if all_scores:
            overall_mean = np.mean(all_scores)
            row.append(f'\\textbf{{{overall_mean:.2f}}}')
        else:
            row.append('--')

        latex.append(' & '.join(row) + ' \\\\')

    latex.append('\\bottomrule')
    latex.append('\\end{tabular}')
    latex.append('\\end{table}')

    table_str = '\n'.join(latex)

    # Save to file
    with open(output_path / 'table1_main_results.tex', 'w') as f:
        f.write(table_str)

    return table_str


def generate_table2_statistical_tests(
    results: Dict[str, Any],
    output_path: Path,
) -> str:
    """
    Generate Table 2: Statistical test results.

    Format: Predicate | H-statistic | p-value | eta^2 | Significant
    """
    test_results = results.get('results', {})
    predicates = sorted(test_results.keys())

    latex = []
    latex.append('\\begin{table}[htbp]')
    latex.append('\\centering')
    latex.append('\\caption{Omnibus Statistical Tests (Kruskal-Wallis)}')
    latex.append('\\label{tab:statistical_tests}')
    latex.append('\\begin{tabular}{lcccc}')
    latex.append('\\toprule')
    latex.append('Predicate Test & $H$ & $p$ & $\\eta^2$ & Sig. \\\\')
    latex.append('\\midrule')

    for pred in predicates:
        agent_results = test_results[pred]

        # Collect groups
        groups = []
        for agent, result in agent_results.items():
            raw = result.get('raw_data', {})
            scores = raw.get('trial_scores', [result.get('score', 0)])
            if len(scores) >= 2:
                groups.append(scores)

        if len(groups) >= 2:
            # Kruskal-Wallis test
            h_stat, p_val = stats.kruskal(*groups)

            # Eta-squared
            n_total = sum(len(g) for g in groups)
            eta_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))
            eta_sq = max(0, eta_sq)

            sig = '*' if p_val < 0.05 else ''
            if p_val < 0.01:
                sig = '**'
            if p_val < 0.001:
                sig = '***'

            pred_name = pred.replace('_', ' ').title()
            latex.append(
                f'{pred_name[:25]} & {h_stat:.2f} & {format_pvalue(p_val)} & '
                f'{eta_sq:.3f} & {sig} \\\\'
            )

    latex.append('\\bottomrule')
    latex.append('\\multicolumn{5}{l}{\\footnotesize *$p<0.05$, **$p<0.01$, ***$p<0.001$} \\\\')
    latex.append('\\end{tabular}')
    latex.append('\\end{table}')

    table_str = '\n'.join(latex)

    with open(output_path / 'table2_statistical_tests.tex', 'w') as f:
        f.write(table_str)

    return table_str


def generate_table3_effect_sizes(
    results: Dict[str, Any],
    output_path: Path,
) -> str:
    """
    Generate Table 3: Pairwise effect sizes (Cohen's d).

    Focus on key comparisons.
    """
    test_results = results.get('results', {})

    # Focus on AA test (primary result)
    aa_results = test_results.get('anticipatory_activation', {})

    if not aa_results:
        return ''

    agents = sorted(aa_results.keys())

    latex = []
    latex.append('\\begin{table}[htbp]')
    latex.append('\\centering')
    latex.append('\\caption{Pairwise Effect Sizes for Anticipatory Activation (Cohen\'s $d$)}')
    latex.append('\\label{tab:effect_sizes}')
    latex.append('\\begin{tabular}{l' + 'c' * len(agents) + '}')
    latex.append('\\toprule')

    # Header
    agent_names = {
        'heuristic': 'Heur.',
        'rl_ppo': 'PPO',
        'active_inference': 'ActInf',
    }
    header = [''] + [agent_names.get(a, a[:5]) for a in agents]
    latex.append(' & '.join(header) + ' \\\\')
    latex.append('\\midrule')

    # Build effect size matrix
    for i, a1 in enumerate(agents):
        row = [agent_names.get(a1, a1[:5])]
        raw1 = aa_results[a1].get('raw_data', {})
        scores1 = raw1.get('trial_scores', [aa_results[a1].get('score', 0)])

        for j, a2 in enumerate(agents):
            if i == j:
                row.append('--')
            elif i < j:
                raw2 = aa_results[a2].get('raw_data', {})
                scores2 = raw2.get('trial_scores', [aa_results[a2].get('score', 0)])
                d = compute_cohens_d(scores1, scores2)
                if d is not None:
                    # Format with effect size interpretation
                    if abs(d) >= 0.8:
                        row.append(f'\\textbf{{{d:.2f}}}')
                    elif abs(d) >= 0.5:
                        row.append(f'{d:.2f}')
                    else:
                        row.append(f'{d:.2f}')
                else:
                    row.append('--')
            else:
                row.append('')  # Below diagonal, will be filled

        latex.append(' & '.join(row) + ' \\\\')

    latex.append('\\bottomrule')
    latex.append('\\multicolumn{' + str(len(agents) + 1) + '}{l}{\\footnotesize Bold: large effect ($|d| \\geq 0.8$)} \\\\')
    latex.append('\\end{tabular}')
    latex.append('\\end{table}')

    table_str = '\n'.join(latex)

    with open(output_path / 'table3_effect_sizes.tex', 'w') as f:
        f.write(table_str)

    return table_str


def generate_supplementary_table(
    results: Dict[str, Any],
    output_path: Path,
) -> str:
    """
    Generate Table S1: Full results with all statistics.
    """
    test_results = results.get('results', {})

    latex = []
    latex.append('\\begin{table}[htbp]')
    latex.append('\\centering')
    latex.append('\\caption{Full Results: All Predicate Tests with Descriptive Statistics}')
    latex.append('\\label{tab:full_results}')
    latex.append('\\small')
    latex.append('\\begin{tabular}{llccccc}')
    latex.append('\\toprule')
    latex.append('Predicate & Agent & Mean & SD & 95\\% CI & $n$ \\\\')
    latex.append('\\midrule')

    for pred in sorted(test_results.keys()):
        first_row = True
        for agent in sorted(test_results[pred].keys()):
            result = test_results[pred][agent]
            raw = result.get('raw_data', {})
            scores = raw.get('trial_scores', [result.get('score', 0)])

            stats_dict = compute_stats(scores)

            pred_name = pred.replace('_', ' ').title()[:20] if first_row else ''
            agent_name = agent.replace('_', ' ').title()

            latex.append(
                f'{pred_name} & {agent_name} & '
                f'{stats_dict["mean"]:.3f} & {stats_dict["std"]:.3f} & '
                f'[{stats_dict["ci_low"]:.3f}, {stats_dict["ci_high"]:.3f}] & '
                f'{stats_dict["n"]} \\\\'
            )
            first_row = False

        latex.append('\\midrule')

    # Remove last midrule and add bottomrule
    latex[-1] = '\\bottomrule'

    latex.append('\\end{tabular}')
    latex.append('\\end{table}')

    table_str = '\n'.join(latex)

    with open(output_path / 'table_s1_full_results.tex', 'w') as f:
        f.write(table_str)

    return table_str


def main():
    parser = argparse.ArgumentParser(description='Generate publication tables')
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to test_results.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/tables',
        help='Output directory for tables'
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generating publication tables...")
    print(f"  Results: {results_path}")
    print(f"  Output: {output_path}")
    print()

    # Load results
    results = load_results(results_path)

    # Generate tables
    print("Generating Table 1: Main results...")
    generate_table1_main_results(results, output_path)

    print("Generating Table 2: Statistical tests...")
    generate_table2_statistical_tests(results, output_path)

    print("Generating Table 3: Effect sizes...")
    generate_table3_effect_sizes(results, output_path)

    print("Generating Table S1: Full results...")
    generate_supplementary_table(results, output_path)

    print()
    print(f"All tables saved to: {output_path}")
    print()
    print("Generated files:")
    for f in sorted(output_path.glob("*.tex")):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
