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
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assignment.assignment import generate_sbm, detect_communities

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
# Test 1 : Community detection must fail when delta_k is too small
# ------------------------------------------------------------
edges, community_labels = generate_sbm(k = k, n = n, delta_k = 0, q = 2)
detected_community_labels = detect_communities(edges, n_nodes)

nmi = normalized_mutual_info_score(detected_community_labels, community_labels)

assert nmi < 0.8, (
    f"Test 1 failed: NMI={nmi:.3f}. "
    f"The communities should not be detectable when delta_k is too small (delta_k=0). "
    f"Your community detection should fail to find any meaningful communities in this case. "
    f"Check your community detection logic for delta_k=0."
)


# ------------------------------------------------------------
# Test 2 : Community detection must be successful when delta_k is large
# ------------------------------------------------------------
edges, community_labels = generate_sbm(k = k, n = n, delta_k = k, q = 2)
detected_community_labels = detect_communities(edges, n_nodes)

nmi = normalized_mutual_info_score(detected_community_labels, community_labels)

assert nmi > 0.95, (
    f"Test 2 failed: NMI={nmi:.3f}. "
    f"The communities should be detectable when delta_k is large (delta_k=k). "
    f"Your community detection should find meaningful communities in this case. "
    f"Check your community detection logic for delta_k=k."
)
