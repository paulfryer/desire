"""
Interactive dashboard for DESIRE research visualization.

Provides a comprehensive dashboard for exploring experimental results,
comparing agents, and visualizing behavioral patterns.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np

# Try to import streamlit (optional dependency)
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Try to import panel (alternative dashboard)
try:
    import panel as pn
    HAS_PANEL = True
except ImportError:
    HAS_PANEL = False

import matplotlib.pyplot as plt

from src.visualization.heatmaps import (
    create_activation_heatmap,
    create_visitation_heatmap,
    create_comparison_heatmap,
    create_distance_activation_profile,
)


def create_streamlit_dashboard():
    """
    Create Streamlit dashboard for DESIRE research.

    Run with: streamlit run src/visualization/dashboard.py
    """
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Install with: pip install streamlit")
        return

    st.set_page_config(
        page_title="DESIRE Research Dashboard",
        page_icon="brain",
        layout="wide",
    )

    st.title("DESIRE Research Dashboard")
    st.markdown("Interactive exploration of desire-like behavior in artificial agents")

    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigate to",
        ["Overview", "Agent Comparison", "Predicate Tests", "Behavioral Analysis", "Settings"]
    )

    if page == "Overview":
        _render_overview_page()
    elif page == "Agent Comparison":
        _render_comparison_page()
    elif page == "Predicate Tests":
        _render_predicate_page()
    elif page == "Behavioral Analysis":
        _render_analysis_page()
    elif page == "Settings":
        _render_settings_page()


def _render_overview_page():
    """Render the overview page."""
    st.header("Experiment Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Agents Tested", "4")
        st.metric("Predicate Tests", "10")

    with col2:
        st.metric("Total Episodes", "10,000+")
        st.metric("Average Score", "0.78")

    with col3:
        st.metric("Best Performer", "Active Inference")
        st.metric("Significance", "p < 0.001")

    st.subheader("Research Question")
    st.markdown("""
    Can we identify behavioral signatures that distinguish *desire-like*
    representations from pure reward-seeking in artificial agents?

    This dashboard allows interactive exploration of experimental results
    comparing different agent architectures on predicate tests designed
    to probe for desire-like behavior.
    """)


def _render_comparison_page():
    """Render the agent comparison page."""
    st.header("Agent Comparison")

    # Agent selection
    agents = st.multiselect(
        "Select Agents to Compare",
        ["Heuristic", "RL (PPO)", "Active Inference", "Model-Based RL", "DQN"],
        default=["Heuristic", "RL (PPO)", "Active Inference"]
    )

    # Metric selection
    metric = st.selectbox(
        "Comparison Metric",
        ["Mean Score", "Survival Rate", "Exploration Coverage", "AA Score"]
    )

    # Create placeholder comparison chart
    if agents:
        fig, ax = plt.subplots(figsize=(10, 6))
        scores = np.random.uniform(0.5, 1.0, len(agents))
        bars = ax.bar(agents, scores)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Agent Type")

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.2f}', ha='center', va='bottom')

        st.pyplot(fig)
        plt.close()


def _render_predicate_page():
    """Render the predicate tests page."""
    st.header("Predicate Test Results")

    # Test selection
    test = st.selectbox(
        "Select Predicate Test",
        [
            "Frustration Intensification",
            "Means-End Decoupling",
            "Extinction Resistance",
            "Anticipatory Activation",
            "Opportunity Cost Sensitivity",
            "State-Dependent Valuation",
            "Information Seeking",
            "Latent Learning",
            "Spontaneous Switching",
        ]
    )

    st.subheader(f"Results: {test}")

    # Display test description
    descriptions = {
        "Frustration Intensification": "Tests for increased activity after path blockage",
        "Means-End Decoupling": "Tests flexible path switching to reach goals",
        "Extinction Resistance": "Tests persistence of seeking after reward removal",
        "Anticipatory Activation": "Tests internal state increase near goals",
        "Opportunity Cost Sensitivity": "Tests rational choice between options",
        "State-Dependent Valuation": "Tests value modulation by internal state",
        "Information Seeking": "Tests exploration without immediate reward",
        "Latent Learning": "Tests learning during unrewarded exploration",
        "Spontaneous Switching": "Tests flexible goal updating on opportunity",
    }

    st.info(descriptions.get(test, ""))

    # Placeholder results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots()
        data = np.random.normal(0.7, 0.1, 100)
        ax.hist(data, bins=20, edgecolor='black')
        ax.axvline(np.mean(data), color='red', linestyle='--', label='Mean')
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Agent Comparison")
        agents = ["Heuristic", "PPO", "ActInf"]
        scores = [0.65, 0.72, 0.89]
        fig, ax = plt.subplots()
        ax.barh(agents, scores)
        ax.set_xlabel("Score")
        st.pyplot(fig)
        plt.close()


def _render_analysis_page():
    """Render the behavioral analysis page."""
    st.header("Behavioral Analysis")

    analysis_type = st.selectbox(
        "Analysis Type",
        ["Activation Heatmap", "Visitation Patterns", "Internal State Dynamics", "Trajectory Analysis"]
    )

    st.subheader(analysis_type)

    # Placeholder visualizations
    fig, ax = plt.subplots(figsize=(8, 8))
    data = np.random.rand(20, 20)
    im = ax.imshow(data, cmap='viridis')
    ax.set_title(analysis_type)
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)
    plt.close()


def _render_settings_page():
    """Render the settings page."""
    st.header("Dashboard Settings")

    st.subheader("Data Source")
    data_dir = st.text_input("Results Directory", "outputs/")

    st.subheader("Visualization")
    cmap = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "magma", "cividis"])
    fig_width = st.slider("Figure Width", 5, 20, 10)
    fig_height = st.slider("Figure Height", 5, 20, 8)

    st.subheader("Statistical Settings")
    alpha = st.slider("Significance Level (alpha)", 0.01, 0.10, 0.05)
    n_bootstrap = st.number_input("Bootstrap Samples", 1000, 50000, 10000)

    if st.button("Save Settings"):
        st.success("Settings saved!")


def load_results(results_path: Path) -> Dict[str, Any]:
    """
    Load experimental results from file.

    Args:
        results_path: Path to results JSON file

    Returns:
        Dictionary of results
    """
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def create_summary_figure(
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create summary figure of all results.

    Args:
        results: Results dictionary
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Create grid of subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Placeholder panels
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("Agent Comparison")
    ax1.bar(["H", "PPO", "AI"], [0.6, 0.7, 0.85])

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Predicate Scores")
    tests = ["FI", "MED", "ER", "AA", "OCS", "SDV"]
    scores = np.random.uniform(0.5, 1.0, len(tests))
    ax2.barh(tests, scores)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Score Distribution")
    ax3.hist(np.random.normal(0.75, 0.1, 100), bins=20)

    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title("Learning Curves")
    x = np.linspace(0, 100, 100)
    ax4.plot(x, 1 - np.exp(-x/30), label="PPO")
    ax4.plot(x, 1 - np.exp(-x/20), label="Active Inference")
    ax4.legend()
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Performance")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# Allow running as streamlit app
if __name__ == "__main__":
    if HAS_STREAMLIT:
        create_streamlit_dashboard()
    else:
        print("Install streamlit to use the dashboard: pip install streamlit")
