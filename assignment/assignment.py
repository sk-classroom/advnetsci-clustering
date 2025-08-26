import marimo

__generated_with = "0.14.16"
app = marimo.App(width="full", theme="dark")

with app.setup(hide_code=True):
    # Initialization code that runs before all other cells
    import numpy as np
    import igraph
    import scipy.sparse
    from sklearn.metrics import normalized_mutual_info_score

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Community Detectability Limit Assignment

    Welcome to the Community Detection assignment!

    In this assignment, you will explore the **detectability limit** of community detection algorithms using the Stochastic Block Model (SBM). You will learn when communities become detectable as we vary the strength of community structure, and discover the theoretical threshold that determines successful community detection.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 📋 Assignment Instructions & Grading
    """
    )
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
            You **cannot** import any other libraries that result in the grading script failing or a zero score. Only use: `numpy`, `igraph`, `scipy`, `sklearn.metrics.normalized_mutual_info_score`, `pandas`, `altair`
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

    Implement a function that generates networks using the **Stochastic Block Model (SBM)** with two communities.

    **Parameters:**
    - `k`: Average degree (fixed at 20)
    - `n`: Community size (number of nodes per community)
    - `q`: Number of communities (fixed at 2)
    - `delta_k`: Gap between within-community and between-community connections

    **Formulas:**
    - `k_in = (k + delta_k) / 2` (average within-community degree)
    - `k_out = (k - delta_k) / 2` (average between-community degree)
    - `p_in = k_in / n` (probability of within-community edges)
    - `p_out = k_out / n` (probability of between-community edges)

    **Return:** An igraph Graph object with community labels stored as vertex attribute "community"
    
    **Alternative return format (if preferred):** You may also return a tuple containing:
    - Edge list: [(node1, node2), (node1, node3), ...] 
    - Community membership: list of integers [0, 0, 1, 1, ...] indicating community membership
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
        igraph.Graph: Graph with community labels stored as vertex attribute "community"
        OR tuple: (edge_list, community_membership) where:
            - edge_list: list of tuples [(node1, node2), ...]
            - community_membership: list of integers [0, 0, 1, 1, ...] indicating community membership
    """
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 2: Community Detection Function

    Implement a community detection function that takes a network in **scipy sparse adjacency matrix** format and returns detected community labels.

    **Requirements:**
    - Input: `scipy.sparse` matrix (adjacency matrix)
    - Output: `numpy.ndarray` of community labels (integers starting from 0)
    - You can use any community detection algorithm you prefer:
      - **Leiden algorithm** (recommended for best performance)
      - **Louvain algorithm** (good balance of speed and quality)  
      - **Infomap** (information-theoretic approach)
      - Any other algorithm available in igraph

    **Hint:** Convert the sparse matrix to igraph format, apply community detection, then return the membership vector.
    """
    )
    return


@app.function
# Task 2
def detect_communities(adj_matrix):
    """
    Detect communities in a network from adjacency matrix.

    Args:
        adj_matrix (scipy.sparse matrix): Sparse adjacency matrix

    Returns:
        numpy.ndarray: Community labels (integers starting from 0)
    """
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 3: Theoretical Detectability Threshold

    After running your experiments, determine the **theoretical detectability threshold** for the Stochastic Block Model.

    **Your task:**
    1. Run experiments with `delta_k` values from 0 to 8 (10 intermediate points)
    2. Plot the Normalized Mutual Information (NMI) vs. `delta_k`
    3. Find the minimum `delta_k` where your algorithm performs better than random guessing (NMI > ~0.1)
    4. **Compare with theory:** The theoretical threshold for detectability in SBM with two equal-sized communities is:
    
    **Formula:** `delta_k_threshold = sqrt(k)` where k is the average degree
    
    For k=20, the theoretical threshold is `delta_k_threshold = sqrt(20) ≈ 4.47`

    **Win condition:** Find the empirical threshold from your results and compare it with the theoretical prediction!
    """
    )
    return


@app.function
# Task 3 - Helper function to compute theoretical threshold
def compute_theoretical_threshold(k):
    """
    Compute the theoretical detectability threshold for SBM.

    Args:
        k (float): Average degree

    Returns:
        float: Theoretical threshold delta_k_threshold = sqrt(k)
    """
    pass


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Community Detectability Experiment

    This experiment will test your implementations by generating SBM networks with different `delta_k` values and measuring community detection performance.
    """
    )
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Experimental Results and Analysis

    The performance curve shows how community detection accuracy (NMI) changes with the community structure strength (`delta_k`).

    **Key observations to discuss:**
    - **Random regime** (`delta_k < threshold`): NMI ≈ 0, algorithm performs no better than random guessing
    - **Detectable regime** (`delta_k > threshold`): NMI increases, communities become detectable
    - **Threshold location**: Compare your empirical threshold with the theoretical prediction `sqrt(20) ≈ 4.47`

    **Questions to consider:**
    - How sharp is the phase transition in your results?
    - Does your chosen algorithm match the theoretical prediction?
    - How does performance vary across different random realizations?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Libraries""")
    return


@app.cell
def _():
    # All imports in one place to avoid conflicts
    import numpy as np
    import igraph
    import altair as alt
    import pandas as pd
    return alt, igraph, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Experimental Parameters""")
    return


@app.cell
def _(np):
    # Experimental parameters
    k = 20          # Average degree (fixed)
    n = 50          # Community size
    q = 2           # Number of communities
    
    # Range of delta_k values to test (10 points from 0 to 8)
    delta_k_values = np.linspace(0, 8, 10)
    
    # Theoretical threshold
    theoretical_threshold = np.sqrt(k)
    
    print(f"Average degree: {k}")
    print(f"Community size: {n}")
    print(f"Total network size: {q * n}")
    print(f"Delta_k range: {delta_k_values}")
    print(f"Theoretical threshold: {theoretical_threshold:.2f}")
    
    return delta_k_values, k, n, q, theoretical_threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Generate Networks and Test Community Detection""")
    return


@app.cell
def _(delta_k_values, k, n, q, generate_sbm, detect_communities, normalized_mutual_info_score, scipy, np):
    # Run experiments for each delta_k value
    nmi_scores = []
    
    for delta_k in delta_k_values:
        print(f"Testing delta_k = {delta_k:.2f}")
        
        # Generate SBM network
        result = generate_sbm(k, n, delta_k, q)
        
        if result is None:  # Student hasn't implemented the function yet
            nmi_scores.append(0)
            continue
        
        # Handle both return formats
        if isinstance(result, tuple) and len(result) == 2:
            # Format: (edge_list, community_membership)
            edge_list, true_labels = result
            true_labels = np.array(true_labels)
            
            # Create igraph from edge list
            g = igraph.Graph()
            g.add_vertices(q * n)  # Total number of nodes
            g.add_edges(edge_list)
            
        else:
            # Format: igraph.Graph object
            g = result
            true_labels = np.array(g.vs["community"])
            
        # Get adjacency matrix in sparse format
        adj_matrix = scipy.sparse.csr_matrix(g.get_adjacency().data)
        
        # Detect communities
        detected_labels = detect_communities(adj_matrix)
        
        if detected_labels is None:  # Student hasn't implemented the function yet
            nmi_scores.append(0)
            continue
        
        # Calculate NMI
        nmi = normalized_mutual_info_score(true_labels, detected_labels)
        nmi_scores.append(nmi)
    
    nmi_scores = np.array(nmi_scores)
    print("Experiments completed!")
    
    return nmi_scores,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Performance Visualization""")
    return


@app.cell
def _(delta_k_values, nmi_scores, theoretical_threshold, pd, alt):
    # Create dataframe for plotting
    df = pd.DataFrame({
        'delta_k': delta_k_values,
        'NMI': nmi_scores
    })
    
    # Create the main performance curve
    line_chart = alt.Chart(df).mark_line(
        point=True, 
        color='steelblue', 
        strokeWidth=3
    ).encode(
        x=alt.X('delta_k:Q', title='Δk (Community Structure Strength)', scale=alt.Scale(domain=[0, 8])),
        y=alt.Y('NMI:Q', title='Normalized Mutual Information', scale=alt.Scale(domain=[0, 1])),
        tooltip=['delta_k:Q', 'NMI:Q']
    ).properties(
        width=600,
        height=400,
        title='Community Detectability: NMI vs Community Structure Strength'
    )
    
    # Add theoretical threshold line
    threshold_line = alt.Chart(pd.DataFrame({
        'threshold': [theoretical_threshold, theoretical_threshold],
        'y': [0, 1]
    })).mark_line(
        color='red',
        strokeDash=[5, 5],
        strokeWidth=2
    ).encode(
        x=alt.X('threshold:Q'),
        y=alt.Y('y:Q')
    )
    
    # Add threshold annotation
    threshold_text = alt.Chart(pd.DataFrame({
        'threshold': [theoretical_threshold + 0.2],
        'y': [0.8],
        'label': [f'Theoretical\nThreshold\n√k = {theoretical_threshold:.2f}']
    })).mark_text(
        align='left',
        color='red',
        fontSize=12
    ).encode(
        x=alt.X('threshold:Q'),
        y=alt.Y('y:Q'),
        text='label:N'
    )
    
    # Combine all elements
    chart = (line_chart + threshold_line + threshold_text).resolve_scale(
        x='shared',
        y='shared'
    )
    
    return chart, df


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
