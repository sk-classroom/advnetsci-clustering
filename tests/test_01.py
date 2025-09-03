# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.6",
#     "numpy==2.2.6",
#     "pandas==2.3.2",
#     "python-igraph==0.11.9",
#     "scikit-learn==1.7.1",
#     "seaborn==0.13.2",
#     "tqdm==4.67.1",
# ]
# ///

# %% Import
import numpy as np
import sys
import os
import igraph
from scipy import sparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assignment.assignment import generate_sbm

# %% Test ------------
# Create initial ring lattice (p=0)
n = 300
k = 10
n_nodes = n * 2

def prep_data(edges, community_labels, n_nodes):
    uids, cids = np.unique(community_labels, return_inverse = True)
    n_labels = len(uids)

    # Construct the adjacency matrix A
    src, trg = np.array(edges).T
    A = sparse.csr_matrix((np.ones(len(src)), (src, trg)), shape=(n_nodes, n_nodes))
    A = A + A.T
    A.sort_indices()
    A.eliminate_zeros()

    # Membership matrix
    U = sparse.csr_matrix((np.ones(n_nodes), (np.arange(n_nodes), cids)), shape=(n_nodes, n_labels))

    # Number of edges between communities
    Ac = U.T @ A @ U # community communication matrix (2,2)
    Ac = Ac.toarray()
    return A, U, Ac

# ------------------------------------------------------------
# Test 1 : Check the number of edges between and within communities
# ------------------------------------------------------------
edges, community_labels = generate_sbm(k = k, n = n, delta_k = k, q = 2)
A, U, Ac = prep_data(edges, community_labels, n_nodes)

assert U.shape[1] == 2, (
    f"Test 1 failed: The number of communities detected is {U.shape[1]}, "
    f"but it should be 2. Make sure your community_labels output from generate_sbm "
    f"assigns exactly two unique community labels (e.g., [0]*n + [1]*n for n=300)."
)

assert Ac[0, 1] == 0, (
    f"Test 1 failed: When delta_k=k (all edges are within-community), "
    f"there should be no edges running between communities, but Ac[0, 1]={Ac[0, 1]}. "
    f"Check your probability matrix and SBM generation logic to ensure that between-community "
    f"edges are not created when delta_k=k."
)


# ------------------------------------------------------------
# Test 2 : Check the number of edges between and within communities match with the expected value
# ------------------------------------------------------------

edges, community_labels = generate_sbm(k = k, n = n, delta_k = k // 2, q = 2)
A, U, Ac = prep_data(edges, community_labels, n_nodes)

mu = k * n_nodes
var = mu # Since this is a poisson distribution.
std = np.sqrt(var)

n_edges_generated = np.sum(Ac)
assert n_edges_generated > mu - 3 * std and n_edges_generated < mu + 3 * std, (
    f"Test 2 failed: Total # of edges generated={n_edges_generated}, "
    f"which should be between {mu - 3 * std} and {mu + 3 * std}. "
    f"Expected mean number of edges is {mu} (k={k} * n_nodes={n_nodes}), "
    f"with standard deviation {std:.2f}. "
    f"Check your calculation of p_in and p_out, and ensure your SBM generation "
    f"produces the correct expected number of edges."
)

# ------------------------------------------------------------
# Test 3 : Check the communities are not detectable when delta_k is too small
# ------------------------------------------------------------
edges, community_labels = generate_sbm(k = k, n = n, delta_k = 0, q = 2)
A, U, Ac = prep_data(edges, community_labels, n_nodes)

p = Ac[np.triu_indices(2)]
p = p / np.sum(p)
entropy = -np.sum(p * np.log2(p))
p_uniform = np.ones(3) / 3
uniform_entropy = -np.sum(p_uniform * np.log2(p_uniform))  # For 2x2 matrix, 3 entries, uniform distribution
entropy_ratio = entropy / uniform_entropy

assert np.isclose(entropy_ratio, 1, atol=1e-1), (
    f"Test 3 failed: Entropy ratio={entropy_ratio:.3f}. "
    f"The communities should not be detectable when delta_k is too small (delta_k=0). "
    f"Your SBM should generate a random graph with no community structure in this case, "
    f"so the number of edges between and within communities should be close to uniform. "
    f"Check the distribution of edges between and within communities and SBM generation for delta_k=0.\n"
    f"Here is the observed edge count matrix (Ac):\n{Ac}\n"
    f"Each entry Ac[i, j] gives the number of edges between community i and community j. "
    f"For delta_k=0, these should be close to uniform (all entries similar). "
)


