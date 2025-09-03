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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assignment.assignment import calc_detectability_limit
# ------------------------------------------------------------
# Test 1 : calc_detectability_limit returns correct threshold values for hand-picked average_k
# ------------------------------------------------------------
ks = [1, 13, 25]
expected_thresholds = [1.0, 3.6, 5.0]

for k, expected in zip(ks, expected_thresholds):
    threshold = calc_detectability_limit(k)
    assert np.isclose(threshold, expected, atol=1e-2), (
        f"Test 1 failed: For average_k={k}, expected threshold {expected}, got {threshold}"
    )

# ------------------------------------------------------------
# Test 2 : calc_detectability_limit is monotonic and positive
# ------------------------------------------------------------
thresholds = [calc_detectability_limit(k) for k in ks]

# All thresholds should be positive
assert all(t > 0 for t in thresholds), "Test 2 failed: All thresholds should be positive"

# Thresholds should increase as average_k increases
for i in range(1, len(thresholds)):
    assert thresholds[i] > thresholds[i-1], (
        f"Test 2 failed: Threshold did not increase: {thresholds[i-1]} (k={ks[i-1]}) -> {thresholds[i]} (k={ks[i]})"
    )

# ------------------------------------------------------------
# Test 3 : calc_detectability_limit is not linear
# ------------------------------------------------------------
diffs = [thresholds[i+1] - thresholds[i] for i in range(len(thresholds)-1)]
assert not all(np.isclose(diffs[0], d) for d in diffs[1:]), (
    "Test 3 failed: calc_detectability_limit appears to be linear, which is unlikely for this task."
)
