"""
Analyze and visualize test results.

Usage:
    python experiments/analyze_results.py --results outputs/tests_xxx/test_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.tests.base_test import TestResult
from src.analysis.metrics import compare_agents, aggregate_metrics
from src.analysis.fitting import compare_fits, fit_exponential_decay, fit_power_law_decay
from src.analysis.visualization import (
    plot_predicate_scores,
    plot_comparison_grid,
    plot_frustration_response,
    save_figure,
    create_summary_report,
)


def load_results(results_path: Path) -> Dict:
    """Load test results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def reconstruct_test_results(data: Dict) -> Dict[str, Dict[str, TestResult]]:
    """Reconstruct TestResult objects from JSON data."""
    results = {}

    for test_name, test_data in data.get("results", {}).items():
        results[test_name] = {}
        for agent_name, result_data in test_data.items():
            results[test_name][agent_name] = TestResult(
                test_name=result_data["test_name"],
                agent_name=result_data["agent_name"],
                score=result_data["score"],
                raw_data=result_data.get("raw_data", {}),
                metadata=result_data.get("metadata", {}),
            )

    return results


def analyze_frustration_dynamics(results: Dict[str, TestResult]) -> Dict:
    """Analyze the dynamics of frustration responses."""
    analysis = {}

    for agent_name, result in results.items():
        raw_data = result.raw_data

        if "post_intensity_curve" not in raw_data:
            continue

        post_curve = np.array(raw_data["post_intensity_curve"])
        baseline_mean = raw_data.get("baseline_mean_intensity", 1.0)

        if len(post_curve) < 5:
            continue

        x = np.arange(len(post_curve))

        # Fit decay curves
        exp_fit = fit_exponential_decay(x, post_curve)
        power_fit = fit_power_law_decay(x, post_curve)

        # Compare fits
        comparison = compare_fits(x, post_curve)

        analysis[agent_name] = {
            "peak_intensity": float(np.max(post_curve)),
            "time_to_peak": int(np.argmax(post_curve)),
            "final_intensity": float(post_curve[-1]),
            "decay_type": comparison.get("best_model", "unknown"),
            "exponential_r2": comparison["exponential"]["r_squared"] if comparison["exponential"]["success"] else None,
            "power_law_r2": comparison["power_law"]["r_squared"] if comparison["power_law"]["success"] else None,
            "interpretation": _interpret_frustration(comparison, post_curve, baseline_mean),
        }

    return analysis


def _interpret_frustration(comparison: Dict, post_curve: np.ndarray, baseline: float) -> str:
    """Generate interpretation of frustration dynamics."""
    peak = np.max(post_curve)
    fi_ratio = peak / baseline if baseline > 0 else 0

    interpretations = []

    # Intensity interpretation
    if fi_ratio > 2.0:
        interpretations.append("Strong frustration response (>2x baseline)")
    elif fi_ratio > 1.5:
        interpretations.append("Moderate frustration response (1.5-2x baseline)")
    elif fi_ratio > 1.0:
        interpretations.append("Mild frustration response (>1x baseline)")
    else:
        interpretations.append("No frustration response detected")

    # Decay interpretation
    decay_type = comparison.get("decay_type_indicator", "undetermined")
    if decay_type == "power_law_preferred":
        interpretations.append("Power-law decay suggests persistent goal-seeking")
    elif decay_type == "exponential_preferred":
        interpretations.append("Exponential decay suggests reward-driven adaptation")

    return "; ".join(interpretations)


def analyze_means_end_transfer(results: Dict[str, TestResult]) -> Dict:
    """Analyze means-end decoupling transfer efficiency."""
    analysis = {}

    for agent_name, result in results.items():
        raw_data = result.raw_data

        p1_times = raw_data.get("p1_reach_times", [])
        p2_times = raw_data.get("p2_reach_times", [])

        if not p1_times and not p2_times:
            continue

        analysis[agent_name] = {
            "p1_success_rate": raw_data.get("p1_success_rate", 0),
            "p2_success_rate": raw_data.get("p2_success_rate", 0),
            "transfer_efficiency": raw_data.get("transfer_efficiency", 0),
            "mean_p1_time": raw_data.get("mean_p1_time"),
            "mean_p2_time": raw_data.get("mean_p2_time"),
            "optimal_p2_length": raw_data.get("optimal_p2_length"),
            "interpretation": _interpret_transfer(
                raw_data.get("p1_success_rate", 0),
                raw_data.get("p2_success_rate", 0),
                result.score,
            ),
        }

    return analysis


def _interpret_transfer(p1_success: float, p2_success: float, med_score: float) -> str:
    """Generate interpretation of transfer performance."""
    interpretations = []

    # Success rate interpretation
    if p2_success > 0.8:
        interpretations.append("High transfer success")
    elif p2_success > 0.5:
        interpretations.append("Moderate transfer success")
    else:
        interpretations.append("Low transfer success")

    # MED score interpretation
    if med_score > 0.8:
        interpretations.append("Near-optimal path efficiency")
    elif med_score > 0.5:
        interpretations.append("Suboptimal but functional transfer")
    else:
        interpretations.append("Poor means-end decoupling")

    # Comparison interpretation
    if p2_success > 0 and p1_success > 0:
        transfer_drop = (p1_success - p2_success) / p1_success
        if transfer_drop < 0.1:
            interpretations.append("Robust goal representation")
        elif transfer_drop < 0.3:
            interpretations.append("Some path-specific learning")
        else:
            interpretations.append("High path-dependence")

    return "; ".join(interpretations)


def generate_report(data: Dict, output_dir: Path) -> None:
    """Generate a comprehensive analysis report."""
    results = reconstruct_test_results(data)

    report_lines = [
        "# Desire Research - Analysis Report",
        f"\nGenerated: {data.get('timestamp', 'Unknown')}",
        "\n## Configuration",
    ]

    # Configuration summary
    config = data.get("config", {})
    report_lines.extend([
        f"- Experiment: {config.get('experiment_name', 'Unknown')}",
        f"- Grid size: {config.get('environment', {}).get('grid_size', 'Unknown')}",
        f"- Trials per test: {config.get('test', {}).get('n_trials', 'Unknown')}",
    ])

    # Test results
    report_lines.append("\n## Test Results\n")

    for test_name, test_results in results.items():
        report_lines.append(f"### {test_name.replace('_', ' ').title()}\n")

        report_lines.append("| Agent | Score | Std |")
        report_lines.append("|-------|-------|-----|")

        for agent_name, result in test_results.items():
            std = result.raw_data.get("std_score", 0)
            report_lines.append(f"| {agent_name} | {result.score:.3f} | {std:.3f} |")

        report_lines.append("")

    # Statistical analysis
    report_lines.append("\n## Statistical Analysis\n")

    analysis = data.get("analysis", {})
    for test_name, test_analysis in analysis.items():
        report_lines.append(f"### {test_name.replace('_', ' ').title()}\n")

        if "kruskal_wallis" in test_analysis:
            kw = test_analysis["kruskal_wallis"]
            sig_text = "SIGNIFICANT" if kw["significant"] else "not significant"
            report_lines.append(
                f"**Kruskal-Wallis H-test**: H = {kw['statistic']:.2f}, "
                f"p = {kw['p_value']:.4f} ({sig_text})\n"
            )

        if "pairwise_comparisons" in test_analysis:
            report_lines.append("**Pairwise Comparisons:**\n")
            for comp_name, comp_data in test_analysis["pairwise_comparisons"].items():
                sig_marker = "*" if comp_data["significant"] else ""
                report_lines.append(
                    f"- {comp_name}: U = {comp_data['u_statistic']:.1f}, "
                    f"p = {comp_data['p_value']:.4f}{sig_marker}, "
                    f"effect = {comp_data['effect_size']:.3f}"
                )
            report_lines.append("")

    # Detailed analysis
    if "frustration_intensification" in results:
        report_lines.append("\n## Frustration Dynamics Analysis\n")
        fi_analysis = analyze_frustration_dynamics(results["frustration_intensification"])

        for agent_name, agent_analysis in fi_analysis.items():
            report_lines.append(f"### {agent_name}\n")
            report_lines.append(f"- Peak intensity: {agent_analysis['peak_intensity']:.3f}")
            report_lines.append(f"- Time to peak: {agent_analysis['time_to_peak']} steps")
            report_lines.append(f"- Decay type: {agent_analysis['decay_type']}")
            report_lines.append(f"- **Interpretation**: {agent_analysis['interpretation']}\n")

    if "means_end_decoupling" in results:
        report_lines.append("\n## Transfer Efficiency Analysis\n")
        med_analysis = analyze_means_end_transfer(results["means_end_decoupling"])

        for agent_name, agent_analysis in med_analysis.items():
            report_lines.append(f"### {agent_name}\n")
            report_lines.append(f"- P1 success rate: {agent_analysis['p1_success_rate']:.2%}")
            report_lines.append(f"- P2 success rate: {agent_analysis['p2_success_rate']:.2%}")
            report_lines.append(f"- Transfer efficiency: {agent_analysis['transfer_efficiency']:.3f}")
            report_lines.append(f"- **Interpretation**: {agent_analysis['interpretation']}\n")

    # Conclusions
    report_lines.append("\n## Conclusions\n")
    report_lines.append(_generate_conclusions(results, analysis))

    # Write report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to {report_path}")


def _generate_conclusions(
    results: Dict[str, Dict[str, TestResult]],
    analysis: Dict,
) -> str:
    """Generate overall conclusions from the analysis."""
    conclusions = []

    # Check for significant differences
    has_significant = False
    for test_name, test_analysis in analysis.items():
        if test_analysis.get("kruskal_wallis", {}).get("significant", False):
            has_significant = True
            break

    if has_significant:
        conclusions.append(
            "The analysis reveals **statistically significant differences** "
            "between agent types on at least one predicate test."
        )
    else:
        conclusions.append(
            "No statistically significant differences were found between agent types. "
            "This may indicate insufficient test sensitivity or genuine behavioral similarity."
        )

    # Agent-specific observations
    for test_name, test_results in results.items():
        scores = {name: r.score for name, r in test_results.items()}
        if scores:
            best_agent = max(scores.keys(), key=lambda k: scores[k])
            worst_agent = min(scores.keys(), key=lambda k: scores[k])

            if scores[best_agent] > scores[worst_agent] * 1.2:
                conclusions.append(
                    f"On {test_name.replace('_', ' ')}, **{best_agent}** showed the "
                    f"strongest response (score: {scores[best_agent]:.3f}), while "
                    f"{worst_agent} showed the weakest (score: {scores[worst_agent]:.3f})."
                )

    return "\n\n".join(conclusions)


def main():
    parser = argparse.ArgumentParser(description="Analyze test results")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to test_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    # Load results
    print(f"Loading results from {results_path}")
    data = load_results(results_path)

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / "analysis"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct TestResult objects
    results = reconstruct_test_results(data)

    # Generate visualizations
    if not args.no_plots:
        print("\nGenerating visualizations...")

        try:
            import matplotlib.pyplot as plt

            for test_name, test_results in results.items():
                fig = plot_predicate_scores(
                    test_results,
                    title=f"{test_name.replace('_', ' ').title()} Scores",
                )
                save_figure(fig, output_dir / f"{test_name}_scores")
                plt.close(fig)
                print(f"  Saved {test_name}_scores.png")

        except ImportError:
            print("Warning: matplotlib not available, skipping plots")

    # Generate report
    print("\nGenerating analysis report...")
    generate_report(data, output_dir)

    print(f"\nAnalysis complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
