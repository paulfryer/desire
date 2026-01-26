"""
Representational Similarity Analysis (RSA) and Centered Kernel Alignment (CKA).

Tools for analyzing and comparing neural representations across agents and conditions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

# Try to import UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    UMAP = None


@dataclass
class RSAResult:
    """Result from Representational Similarity Analysis."""

    rsa_correlation: float
    p_value: float
    rdm1: np.ndarray
    rdm2: np.ndarray
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rsa_correlation": self.rsa_correlation,
            "p_value": self.p_value,
            "n_samples": self.n_samples,
        }


@dataclass
class CKAResult:
    """Result from Centered Kernel Alignment."""

    cka_score: float
    kernel: str
    n_samples: int
    hsic_xy: float
    hsic_xx: float
    hsic_yy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cka_score": self.cka_score,
            "kernel": self.kernel,
            "n_samples": self.n_samples,
            "hsic_xy": self.hsic_xy,
            "hsic_xx": self.hsic_xx,
            "hsic_yy": self.hsic_yy,
        }


@dataclass
class ProjectionResult:
    """Result from dimensionality reduction projection."""

    projections: np.ndarray
    labels: np.ndarray
    method: str
    explained_variance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_samples": len(self.labels),
            "explained_variance": self.explained_variance,
        }


def compute_rdm(representations: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """
    Compute Representational Dissimilarity Matrix (RDM).

    Args:
        representations: Matrix of shape (n_samples, n_features)
        metric: Distance metric ('correlation', 'euclidean', 'cosine')

    Returns:
        RDM of shape (n_samples, n_samples)
    """
    if metric == "correlation":
        # 1 - correlation = dissimilarity
        distances = pdist(representations, metric="correlation")
    elif metric == "cosine":
        distances = pdist(representations, metric="cosine")
    else:
        distances = pdist(representations, metric=metric)

    return squareform(distances)


def compute_rsa(
    representations1: np.ndarray,
    representations2: np.ndarray,
    metric: str = "correlation",
) -> RSAResult:
    """
    Compute Representational Similarity Analysis between two sets of representations.

    RSA measures the correlation between Representational Dissimilarity Matrices.

    Args:
        representations1: First set of representations (n_samples, n_features1)
        representations2: Second set of representations (n_samples, n_features2)
        metric: Distance metric for RDMs

    Returns:
        RSAResult with correlation and significance
    """
    assert len(representations1) == len(representations2), \
        "Representations must have same number of samples"

    n_samples = len(representations1)

    # Compute RDMs
    rdm1 = compute_rdm(representations1, metric=metric)
    rdm2 = compute_rdm(representations2, metric=metric)

    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n_samples, k=1)
    rdm1_vec = rdm1[triu_idx]
    rdm2_vec = rdm2[triu_idx]

    # Compute Spearman correlation
    correlation, p_value = stats.spearmanr(rdm1_vec, rdm2_vec)

    return RSAResult(
        rsa_correlation=float(correlation),
        p_value=float(p_value),
        rdm1=rdm1,
        rdm2=rdm2,
        n_samples=n_samples,
    )


def _center_kernel_matrix(K: np.ndarray) -> np.ndarray:
    """Center a kernel matrix."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _compute_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Compute Hilbert-Schmidt Independence Criterion."""
    n = K.shape[0]
    Kc = _center_kernel_matrix(K)
    Lc = _center_kernel_matrix(L)
    return np.trace(Kc @ Lc) / ((n - 1) ** 2)


def compute_cka(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",
    sigma: Optional[float] = None,
) -> CKAResult:
    """
    Compute Centered Kernel Alignment between two representations.

    CKA measures similarity of representations invariant to isotropic scaling
    and orthogonal transformation.

    Args:
        X: First representations (n_samples, n_features1)
        Y: Second representations (n_samples, n_features2)
        kernel: Kernel type ('linear' or 'rbf')
        sigma: RBF kernel bandwidth (auto if None)

    Returns:
        CKAResult with CKA score
    """
    assert len(X) == len(Y), "X and Y must have same number of samples"

    n_samples = len(X)

    # Compute kernel matrices
    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == "rbf":
        if sigma is None:
            # Auto bandwidth selection using median heuristic
            sigma_x = np.median(pdist(X)) or 1.0
            sigma_y = np.median(pdist(Y)) or 1.0
            sigma = (sigma_x + sigma_y) / 2
        K = rbf_kernel(X, gamma=1.0 / (2 * sigma ** 2))
        L = rbf_kernel(Y, gamma=1.0 / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    # Compute HSIC values
    hsic_xy = _compute_hsic(K, L)
    hsic_xx = _compute_hsic(K, K)
    hsic_yy = _compute_hsic(L, L)

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    denominator = np.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-10:
        cka_score = 0.0
    else:
        cka_score = hsic_xy / denominator

    return CKAResult(
        cka_score=float(cka_score),
        kernel=kernel,
        n_samples=n_samples,
        hsic_xy=float(hsic_xy),
        hsic_xx=float(hsic_xx),
        hsic_yy=float(hsic_yy),
    )


def create_projection_visualization(
    internal_states: np.ndarray,
    labels: np.ndarray,
    methods: List[str] = None,
    n_components: int = 2,
    perplexity: int = 30,
    n_neighbors: int = 15,
    random_state: int = 42,
) -> Dict[str, ProjectionResult]:
    """
    Create dimensionality reduction projections of internal states.

    Args:
        internal_states: Internal state representations (n_samples, n_features)
        labels: Labels for coloring (e.g., goal distances)
        methods: Projection methods to use (default: ["pca", "tsne", "umap"])
        n_components: Number of components (usually 2)
        perplexity: t-SNE perplexity
        n_neighbors: UMAP n_neighbors
        random_state: Random seed

    Returns:
        Dictionary of method -> ProjectionResult
    """
    if methods is None:
        methods = ["pca", "tsne"]
        if HAS_UMAP:
            methods.append("umap")

    results = {}
    n_samples = len(internal_states)

    # Adjust perplexity for small datasets
    effective_perplexity = min(perplexity, n_samples // 4, 50)
    effective_perplexity = max(effective_perplexity, 5)

    for method in methods:
        if method == "pca":
            pca = PCA(n_components=n_components, random_state=random_state)
            projections = pca.fit_transform(internal_states)
            explained_var = float(sum(pca.explained_variance_ratio_))

            results["pca"] = ProjectionResult(
                projections=projections,
                labels=labels,
                method="pca",
                explained_variance=explained_var,
            )

        elif method == "tsne":
            tsne = TSNE(
                n_components=n_components,
                perplexity=effective_perplexity,
                random_state=random_state,
                init="pca",
            )
            projections = tsne.fit_transform(internal_states)

            results["tsne"] = ProjectionResult(
                projections=projections,
                labels=labels,
                method="tsne",
            )

        elif method == "umap" and HAS_UMAP:
            effective_neighbors = min(n_neighbors, n_samples // 2)
            effective_neighbors = max(effective_neighbors, 2)

            umap = UMAP(
                n_components=n_components,
                n_neighbors=effective_neighbors,
                random_state=random_state,
            )
            projections = umap.fit_transform(internal_states)

            results["umap"] = ProjectionResult(
                projections=projections,
                labels=labels,
                method="umap",
            )

    return results


def compare_agent_representations(
    agent1_states: np.ndarray,
    agent2_states: np.ndarray,
    agent1_name: str = "agent1",
    agent2_name: str = "agent2",
) -> Dict[str, Any]:
    """
    Compare representations between two agents.

    Args:
        agent1_states: Internal states from first agent
        agent2_states: Internal states from second agent
        agent1_name: Name of first agent
        agent2_name: Name of second agent

    Returns:
        Comparison results including RSA and CKA
    """
    # Ensure same number of samples
    min_samples = min(len(agent1_states), len(agent2_states))
    states1 = agent1_states[:min_samples]
    states2 = agent2_states[:min_samples]

    # RSA
    rsa_result = compute_rsa(states1, states2)

    # CKA (linear and RBF)
    cka_linear = compute_cka(states1, states2, kernel="linear")
    cka_rbf = compute_cka(states1, states2, kernel="rbf")

    # Procrustes analysis - how well do aligned representations match?
    from scipy.linalg import orthogonal_procrustes
    if states1.shape[1] == states2.shape[1]:
        R, scale = orthogonal_procrustes(states1, states2)
        aligned_states1 = states1 @ R
        procrustes_dist = np.linalg.norm(aligned_states1 - states2)
        procrustes_normalized = procrustes_dist / np.linalg.norm(states2)
    else:
        procrustes_normalized = None

    return {
        "agents": [agent1_name, agent2_name],
        "n_samples": min_samples,
        "rsa": rsa_result.to_dict(),
        "cka_linear": cka_linear.to_dict(),
        "cka_rbf": cka_rbf.to_dict(),
        "procrustes_distance": procrustes_normalized,
    }


def analyze_representation_geometry(
    internal_states: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Analyze geometric properties of internal representations.

    Args:
        internal_states: Internal state matrix
        labels: Labels (e.g., distances to goal)

    Returns:
        Dictionary of geometric metrics
    """
    n_samples, n_dims = internal_states.shape

    # Intrinsic dimensionality estimation (participation ratio)
    centered = internal_states - internal_states.mean(axis=0)
    cov = centered.T @ centered / n_samples
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Participation ratio
    if len(eigenvalues) > 0:
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        else:
            participation_ratio = 0.0
    else:
        participation_ratio = 0.0

    # Effective dimensionality (how many PCs needed for 95% variance)
    if len(eigenvalues) > 0:
        sorted_eigs = np.sort(eigenvalues)[::-1]
        cumvar = np.cumsum(sorted_eigs) / np.sum(sorted_eigs)
        effective_dim = np.searchsorted(cumvar, 0.95) + 1
    else:
        effective_dim = n_dims

    # Representational spread (average distance from centroid)
    centroid = internal_states.mean(axis=0)
    distances_to_centroid = np.linalg.norm(internal_states - centroid, axis=1)
    spread = float(np.mean(distances_to_centroid))

    # Cluster tendency (if labels are categorical)
    unique_labels = np.unique(labels)
    if len(unique_labels) < n_samples / 2:
        # Compute between-class / within-class variance ratio
        between_var = 0
        within_var = 0

        for label in unique_labels:
            mask = labels == label
            class_samples = internal_states[mask]
            if len(class_samples) > 0:
                class_mean = class_samples.mean(axis=0)
                between_var += len(class_samples) * np.sum((class_mean - centroid) ** 2)
                within_var += np.sum((class_samples - class_mean) ** 2)

        cluster_ratio = between_var / within_var if within_var > 0 else 0
    else:
        cluster_ratio = None

    return {
        "participation_ratio": participation_ratio,
        "effective_dimensionality": effective_dim,
        "spread": spread,
        "n_dimensions": n_dims,
        "cluster_ratio": cluster_ratio,
    }
