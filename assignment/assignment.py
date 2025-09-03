# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair==5.5.0",
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

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    # Initialization code that runs before all other cells
    import numpy as np
    import igraph
    from sklearn.metrics import normalized_mutual_info_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Community Detectability Limit Assignment

    Welcome to the Community Detection assignment!

    In this assignment, you will explore the **detectability limit** of community detection algorithms using the Stochastic Block Model (SBM). You will learn when communities become detectable as we vary the strength of community structure, and discover the threshold that determines successful community detection.


    # What is the detectability limit?

    The detectability limit is the maximum level of noise that can be tolerated by the community detection algorithm while still being able to detect the communities better than random guessing. In other words, suppose you have a network with communties (in a sense that on average each node has more edges within its own community than across communities). Since the communities are there, we expect the community detection algorithm to detect them. But the algorithm can fail to do so because some edges are running between communities, blurring the community boundary. The detectability limit asks the maximum level of the noise above which the community detection algorithm fails to detect them.



    Let us consider a more specific case, using the stochastic block model (SBM). We consider a very simple SBM in which we have two communities with equal size $n$ with average degree $k$.
    Keeping the average degree $k$ fixed, we then vary the "strength" of the community structure by varying the gap between the within-community and between-community degree $delta_k$.

    $$
    k_{in} = \frac{k + \delta_k}{2}, k_{out} = \frac{k - \delta_k}{2}
    $$


    As $\delta_k$ increases, more edges are running "within-community" than "across-community", and the communities become more detectable. We should expect to see the following profile, when plotting the accuracy of the detected communities as a function of $\delta_k$.

    ![](https://github.com/user-attachments/assets/2e942ed1-5479-4ebf-8296-f7cdcf1eca0c)

    When $\delta_k = 0$, the network is a random graph, where nodes have an equal probability of being connected to any other node regarsdless of the block structure.
    As a result, the algorithm is no better than random guessing (accuracy of 0.5).
    When $\delta_k > 0$, the network has a higher density of edges within-community than across-community.
    While the communities are present, the algorithm still failed due to the noise in the network until $\delta_k$ is large enough.
    The detectability limit is the minimum $\delta_k$ above which the algorithm finds the communities better than random guessing.




    /// admonition | Why do we care?

    The detectability limit allows us to discuss the optimality of an algorithm for a given type of community structure. For example, for the stochastic block model, it has been shown theoretically that the modularity maximization is in fact an optimal algorithm that can identify the assortative communities down to the information-theoretic limit. No algorithm can do better than this.

    ///

    # The assignment

    In this assignment, we will identify the detectability limit for the stochastic block model.
    You will implement the SBM generation function and the community detection function.
    You will also generate hypothesis about the functional form of the detectability limit $\delta_k = f(k)$ as a function of the average degree $k$, based on the observations.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 📋 Assignment Instructions & Grading""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "📋 Assignment Tasks": mo.md(
                r"""
            Complete the following tasks and upload your notebook to your GitHub repository.

            1. **Task 1**: Implement Stochastic Block Model (SBM) generation function
            2. **Task 2**: Implement community detection function using your preferred algorithm
            3. **Task 3**: Find the theoretical detectability threshold and compare with your results
            4. Update this notebook by using `git add`, `git commit`, and then `git push`.
            5. The notebook will be automatically graded, and your score will be shown on GitHub.
            """
            ),
            "🔒 Protected Files": mo.md(
                r"""
            Protected files are test files and configuration files you cannot modify. They appear in your repository but don't make any changes to them.
            """
            ),
            "⚖️ Academic Integrity": mo.md(
                r"""
            There is a system that automatically checks code similarity across all submissions and online repositories. Sharing code or submitting copied work will result in zero credit and disciplinary action.

            While you can discuss concepts, each student must write their own code. Cite any external resources, including AI tools, in your comments.
            """
            ),
            "📚 Allowed Libraries": mo.md(
                r"""
            You **cannot** import any other libraries that result in the grading script failing or a zero score. Only use: `numpy`, `igraph`, `sklearn.metrics.normalized_mutual_info_score`, `pandas`, `altair`, `tqdm`
            """
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 1: Stochastic Block Model Generation

    To start with, we need networks with which to test the community detection algorithm.
    We will use the stochastic block model (SBM) to generate networks with community structure.

    ### Using igraph.Graph.SBM API

    The `igraph.Graph.SBM` function creates a Stochastic Block Model graph with the following key parameters:

    - **`n`**: Total number of nodes in the graph (for two communities of size n each, this is 2*n)
    - **`pref_matrix`**: The probability matrix as a list of lists. For two communities, this is a 2×2 matrix:
      ```python
      pref_matrix = [[p_in, p_out], [p_out, p_in]]
      ```
    - **`block_sizes`**: List specifying the size of each community/block. For equal communities: `[n, n]`
    - **`directed`**: Set to `False` for undirected networks


    The function returns an igraph Graph object where:

    - Nodes 0 to n-1 belong to community 0
    - Nodes n to 2n-1 belong to community 1
    - You can extract the edge list using `g.get_edgelist()`

    ### Computing the Probability Values

    Now, how do we compute the values $p_{in}$ and $p_{out}$ for our probability matrix? The Stochastic Block Model generates networks by controlling the probability of connections within and between communities.

    A node in a community of size $n$ can potentially connect to $(n-1)$ other nodes within its community. If we want the expected within-community degree to be $k_{in}$, then:

    $$
    p_{in} = \frac{k_{in}}{n-1}
    $$

    Similarly, a node can connect to $n$ nodes in the other community. If we want the expected between-community degree to be $k_{out}$, then:

    $$
    p_{out} = \frac{k_{out}}{n}
    $$

    Now, let's implement the SBM generation function!
    """
    )
    return


@app.function
# Task 1
def generate_sbm(k, n, delta_k, q=2):
    """
    Generate a Stochastic Block Model with two communities.

    Args:
        k (float): Average degree (fixed at 20)
        n (int): Community size (number of nodes per community)
        delta_k (float): Gap between within-community and between-community connections
        q (int): Number of communities (default: 2)

    Returns:
        - edge_list: list of tuples [(node1, node2), ...]
        - community_membership: list of integers [0, 0, 1, 1, ...] indicating community membership
    """

    # Calculate connection probabilities
    k_in = (k + delta_k) / 2  # within-community degree
    k_out = (k - delta_k) / 2  # between-community degree

    # Connection probabilities
    p_in = k_in / n  # probability of within-community edges
    p_out = k_out / n  # probability of between-community edges

    # Your code here
    g = ...

    edges = ...

    community_labels = ...

    return edges, community_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 2: Community Detection Function

    Implement a community detection function that takes a network as an **edge list** and returns detected community labels.

    **Requirements:**

    - Input: `edge_list` - list of tuples representing edges [(node1, node2), (node1, node3), ...]
    - Input: `num_nodes` - total number of nodes in the network
    - Output: `numpy.ndarray` of community labels (integers starting from 0)
    - **Use the Leiden algorithm with modularity objective only**:

    **Hint:** Create igraph Graph from edge list, apply community detection, then return the membership vector.
    You can generate the graph by using `g.add_vertices(num_nodes)` and `g.add_edges(edge_list)`.
    """
    )
    return


@app.function
# Task 2
def detect_communities(edge_list, num_nodes):
    """
    Detect communities in a network from edge list.

    Args:
        edge_list (list): List of tuples representing edges [(node1, node2), ...]
        num_nodes (int): Total number of nodes in the network

    Returns:
        numpy.ndarray: Community labels (integers starting from 0)
    """
    # Create igraph Graph from edge list
    g = igraph.Graph(directed=False)

    # Your code here

    # Return community membership as numpy array
    membership = ...
    return membership


@app.cell
def _():
    k = 100  # Average degree. Change it when working with the task 3.
    max_delta_k = 15  # Reduce this value to zoom in the transition point.
    return k, max_delta_k


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Expand the following cell to see the other parameters:""")
    return


@app.cell(hide_code=True)
def _():
    # Fixed experimental parameters
    n = 500  # Community size
    q = 2  # Number of communities
    n_samples = 10  # Number of simulation runs
    return n, n_samples, q


@app.cell(hide_code=True)
def _(k, plt, results, sns):
    sns.set(style="white", font_scale=1.2)
    sns.set_style("ticks")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.lineplot(
        data=results,
        x="deltaK",
        y="nmi",
    )

    ax.set_xlabel(r"$\delta k$")
    ax.set_ylabel("Normalized Mutual Information")
    if k == 20:
        ax.axvline(
            x=8.94427191 / 2,
            color="red",
            linestyle="--",
            label="Theoretiacl Detectability Threshold",
        )
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The dashed line represents the information theoretic detectability limit, and the modularity maximization can detect the communities down to this limit.

    /// admonition | Finite size effect

    You would observe that the detectability threshold did not sharply separate the two regimes.
    It would be that the algorithm still finds communities slightly better than random guessing below the threshold. This is because in the theoretical limit, the network is assumed to have inifite number of nodes, while our network have a finite number of nodes.
    In principle, the transition should be sharper as the number of nodes increases.

    ///

    ## Task 3: Discover the Relationship Between Average Degree and Detectability Threshold

    Your goal is to discover the functional relationship between average degree `k` and the detectability threshold `δk` through experimentation.

    **Your task:**

    1. Repeat the experiments with different values of `k`
    2. Record the observed detectability threshold `δk` (no need to be precise; we just need to know the functional form)
    3. Identify the functional form of the relationship and write a python function that takes the average degree $k$ and computes the detectability limit `δk`

    Be mindful about the finite size effect, i.e., you should NOT expect that the community detection algorithm should be **exactly** at the same level as the random guessing below the threshold. Even if the algorithm finds communities precisely better than random guessing, as long as the accuracy is still close to zero, take it as a failure.
    """
    )
    return


@app.function
# Task 3
def calc_detectability_limit(average_k):
    """
    Find the detectability threshold functional form of the average degree.

    Args:
        average_k (float): Average degree of the network

    Returns:
        float: The minimum δk above which communities are undetectable
    """
    # Your code here

    detectability_limit = ...

    return detectability_limit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To get an idea of the functional form, record your observations (i.e., the pairs of the average degree k and your observed detectability limit) here. You can move the cell that specifies the $k$ value to here to avoid going up and down repeatedly.""")
    return


@app.cell
def _():
    # Place the average degree values you have tested here
    observed_k_values = [10, 20, 40, 100, 5]

    # Place the corresponding threshold values you observed
    observed_thresholds = [
        2,
        4.2,
        5,
        6,
        0.7,
    ]
    return observed_k_values, observed_thresholds


@app.cell(hide_code=True)
def _(observed_k_values, observed_thresholds, plt, sns):
    # Create relationship plot from student's manual data entry using seaborn
    sns.set(style="white", font_scale=1.2)
    sns.set_style("ticks")

    order = np.argsort(observed_k_values)

    _fig, _ax = plt.subplots(figsize=(8, 6))

    # Plot observed data points
    _ax.scatter(
        np.array(observed_k_values)[order],
        np.array(observed_thresholds)[order],
        color="blue",
        s=100,
        alpha=0.7,
        label="Your Observations",
        zorder=3,
    )

    # Add theoretical curve
    theoretical_thresholds = [calc_detectability_limit(_k) for _k in observed_k_values]

    sns.lineplot(
        x=np.array(observed_k_values)[order],
        y=np.array(theoretical_thresholds)[order],
        linewidth=3,
        alpha=0.8,
        label="Your prediction",
        color="red",
        ax=_ax,
    )

    _ax.set_xlabel("Average Degree (k)")
    _ax.set_ylabel("Detectability Threshold (δk)")
    _ax.set_title("Relationship Discovery: Observations vs Your Prediction")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    sns.despine()

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Libraries""")
    return


@app.cell
def _():
    # All imports in one place to avoid conflicts
    import altair as alt
    import pandas as pd
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from itertools import product
    return pd, plt, product, sns, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Generate Networks and Test Community Detection""")
    return


@app.cell
def _(k, max_delta_k, n, n_samples, nmi_scores, pd, product, q, tqdm):
    results = []
    delta_k_values = np.linspace(0, max_delta_k, 20)
    for delta_k, _sample_id in tqdm(
        product(delta_k_values, range(n_samples)),
        desc="Testing δk values",
        total=len(delta_k_values) * n_samples,
        leave=True,
    ):
        # Generate SBM network
        result = generate_sbm(k, n, delta_k, q)

        if result is None:  # Student hasn't implemented the function yet
            nmi_scores.append(0)
            continue

        # Handle both return formats
        edge_list, true_labels = result
        true_labels = np.array(true_labels)

        # Detect communities using edge list
        detected_labels = detect_communities(edge_list, q * n)

        if detected_labels is None:  # Student hasn't implemented the function yet
            nmi_scores.append(0)
            continue

        # Calculate NMI
        nmi = normalized_mutual_info_score(true_labels, detected_labels)
        results.append({"nmi": nmi, "deltaK": delta_k, "sample": _sample_id})

    results = pd.DataFrame(results)
    return (results,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
