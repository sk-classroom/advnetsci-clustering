import marimo

__generated_with = "0.14.17"
app = marimo.App(width="full")

with app.setup(hide_code=True):
    def _():
        # Initialization code that runs before all other cells
        import numpy as np
        import igraph
        from sklearn.metrics import normalized_mutual_info_score

        return


    _()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Community Detectability Limit Assignment

    Welcome to the Community Detection assignment!

    In this assignment, you will explore the **detectability limit** of community detection algorithms using the Stochastic Block Model (SBM). You will learn when communities become detectable as we vary the strength of community structure, and discover the threshold that determines successful community detection.


    # What is the detectability limit?

    

    Suppose that we have a network with communities



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


@app.cell
def _(igraph, np):
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

        # Create probability matrix for q=2 communities
        prob_matrix = np.array([[p_in, p_out], [p_out, p_in]])

        # Community sizes
        community_sizes = [n, n]

        # Generate SBM using igraph
        g = igraph.Graph.SBM(
            n=q * n,
            pref_matrix=prob_matrix.tolist(),
            block_sizes=community_sizes,
            directed=False,
        )

        # Add community labels as vertex attribute
        community_labels = [0] * n + [
            1
        ] * n  # first n nodes in community 0, next n in community 1

        edges = g.get_edgelist()

        return edges, community_labels
    return (generate_sbm,)


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
    - You can use any community detection algorithm you prefer:
      - **Leiden algorithm** (recommended for best performance)
      - **Louvain algorithm** (good balance of speed and quality)
      - **Infomap** (information-theoretic approach)
      - Any other algorithm available in igraph

    **Hint:** Create igraph Graph from edge list, apply community detection, then return the membership vector.
    """
    )
    return


@app.cell
def _(igraph, np):
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
        g.add_vertices(num_nodes)

        # Add edges if there are any
        if len(edge_list) > 0:
            g.add_edges(edge_list)

        communities = g.community_leiden(objective_function="modularity")

        # Return community membership as numpy array
        return np.array(communities.membership)
    return (detect_communities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Task 3: Discover the Theoretical Relationship

    Your goal is to discover the functional relationship between average degree `k` and the detectability threshold `δk` through experimentation.

    **Your task:**
    1. Implement a function to find the empirical threshold where NMI drops to nearly zero (~0.1)
    2. Use the interactive tool below to run experiments with different values of `k`
    3. For each `k`, record the threshold `δk` where communities become undetectable
    4. Observe how the relationship between `k` and `δk` emerges as you add more data points
    5. **Predict the functional form**: What mathematical relationship do you see?

    **Hint:** The theoretical relationship is `δk_threshold = f(k)`. Can you guess what `f(k)` is?
    """
    )
    return


@app.cell
def _(np):
    # Task 3
    def find_empirical_threshold(average_k):
        """
        Find the detectability threshold functional form of the average degree.

        Args:
            average_k (float): Average degree of the network

        Returns:
            float: The minimum δk above which communities are undetectable
        """
        return 2 * np.sqrt(2 * average_k)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ---
    ## Interactive Detectability Explorer

    Use this tool to explore how the detectability threshold changes with average degree `k`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Manual Data Entry

    Record your observed threshold values here for different average degrees `k`.
    """
    )
    return


@app.cell
def _(np):
    # Students can modify these lists to record their observations
    # Add your observed (k, threshold) pairs here
    observed_k_values = [
        5,
        10,
        15,
        20,
        25,
        30,
        36,
        49,
    ]  # Average degree values you tested
    observed_thresholds = [
        2.3,
        3.1,
        3.9,
        4.4,
        5.1,
        5.5,
        6.0,
        7.0,
    ]  # Corresponding threshold values you observed

    # Convert to numpy arrays for analysis
    k_data = np.array(observed_k_values)
    threshold_data = np.array(observed_thresholds)

    print("Current data points:")
    for _k, thresh in zip(k_data, threshold_data):
        print(f"  k = {_k}, threshold = {thresh:.2f}")
    return k_data, threshold_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Relationship Discovery Plot""")
    return


@app.cell
def _(alt, k_data, np, pd, threshold_data):
    # Create relationship plot from student's manual data entry
    if len(k_data) >= 2:
        student_df = pd.DataFrame({"k": k_data, "threshold": threshold_data})

        # Create scatter plot of student observations
        student_points = (
            alt.Chart(student_df)
            .mark_circle(size=120, color="blue", opacity=0.7)
            .encode(
                x=alt.X("k:Q", title="Average Degree (k)"),
                y=alt.Y("threshold:Q", title="Detectability Threshold (δk)"),
                tooltip=["k:Q", "threshold:Q"],
            )
        )

        # Add theoretical curve y = sqrt(x)
        k_range = np.linspace(5, max(50, np.max(k_data) + 5), 100)
        theory_df = pd.DataFrame({"k": k_range, "threshold": np.sqrt(k_range)})

        theory_curve = (
            alt.Chart(theory_df)
            .mark_line(color="red", strokeDash=[5, 5], strokeWidth=3)
            .encode(x=alt.X("k:Q"), y=alt.Y("threshold:Q"))
        )

        # Add legend
        legend_df = pd.DataFrame(
            {
                "x": [0, 0],
                "y": [0, 0],
                "category": ["Your Observations", "Theory: √k"],
            }
        )

        discovery_chart = (
            (student_points + theory_curve)
            .properties(
                width=700,
                height=450,
                title="Relationship Discovery: Your Observations vs Theory",
            )
            .resolve_scale(x="shared", y="shared")
        )

        # Analysis feedback
        print("📊 Analysis of your data:")
        print(f"Number of data points: {len(k_data)}")

        if len(k_data) >= 3:
            # Calculate correlation with sqrt(k)
            theoretical_values = np.sqrt(k_data)
            correlation = np.corrcoef(threshold_data, theoretical_values)[0, 1]
            print(f"Correlation with √k: {correlation:.3f}")

            if correlation > 0.9:
                print(
                    "🎉 Excellent! Your data strongly matches the theoretical relationship √k"
                )
            elif correlation > 0.7:
                print(
                    "👍 Good correlation with √k theory - you're on the right track!"
                )
            else:
                print(
                    "🤔 Moderate correlation - try collecting more precise threshold values"
                )

    else:
        discovery_chart = (
            alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
            .mark_text(
                text="Add at least 2 data points to see the relationship",
                fontSize=16,
            )
            .encode(x="x:Q", y="y:Q")
            .properties(width=700, height=450, title="Relationship Discovery")
        )
        print("📝 Add your observed (k, threshold) values to the lists above")

    discovery_chart
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
    from tqdm import tqdm
    from sklearn.metrics import normalized_mutual_info_score
    return alt, igraph, normalized_mutual_info_score, np, pd, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Code""")
    return


@app.cell
def _(np):
    # Fixed experimental parameters
    k = 20  # Average degree
    n = 500  # Community size
    q = 2  # Number of communities
    n_samples = 10  # Number of simulation runs
    # Range of delta_k values to test (10 points from 0 to 8)
    delta_k_values = np.linspace(0, 10, 20)

    return delta_k_values, k, n, n_samples, q


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Generate Networks and Test Community Detection""")
    return


@app.cell
def _(
    delta_k_values,
    detect_communities,
    generate_sbm,
    k,
    n,
    n_samples,
    nmi_scores,
    normalized_mutual_info_score,
    np,
    pd,
    q,
    tqdm,
):
    from itertools import product
    # Run experiments for each delta_k value

    print(f"Running experiments with k = {k}...")

    results = []
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
    print("✅ Experiments completed!")
    return (results,)


@app.cell
def _(plt, results, sns):
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
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, plt, sns


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
