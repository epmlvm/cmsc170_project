# manlulu_exer08 <br>
# Elvira P. Manlulu <br>
# CMSC 170 W1L <br>
# October 21, 2026 (Revised) <br>
# Exercise 08 : Perceptron

# DECLARATION ON GENERATIVE AI During the preparation of this work, author used her old python base code to adapt to Google Colab. Author used Gemini to get guidance for importing and writing files from and to the specified Google Drive directly to be processed for this program, correct the errors caused by indentation. Iteration to debug and decode the code was also done using Claude to arrive at the final program. AI was used for plotting and decision_boundary functions


import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Parsing

def parse_input(input_lines):
  if isinstance(input_lines, str):                                          # If input is a single string, split it into a list of lines
    input_lines = input_lines.strip().splitlines()

  input_lines = [line.strip() for line in input_lines if line.strip()]      # Clean and filter empty lines

  eta = float(input_lines[0])                                                # Parse Line 0: Learning rate (eta)
  w0 = float(input_lines[1])                                                 # Parse Line 1: Bias (w0)

  training_data = []                                                         # Parse Line 2 onwards: Training data (x1, x2, ..., xm, d)
  for line in input_lines[2:]:
    parts = [float(val) for val in re.split(r'[\s,]+', line) if val]         # Split line into parts and convert to floats
    if parts:
      training_data.append(parts)

  return eta, w0, training_data

# Step 1. Initialization. Set w(0) = 0. Then perform the following computations for time-step n = 1,2,...)

def init_weights(num_features, b):
  weights = np.zeros(num_features + 1)                                       # Initialize weights: w0 is the bias, w1 to wm are initialized to 0
  weights[0] = b
  return weights


# Step 2. Activation.  At time-step n, activate the perceptron by applying continuous-valued input vector x(n) and desired response d(n).

def activate(row, num_features):
  x = np.ones(num_features + 1)
  x[1:] = row[:-1]
  d = row[-1]                                             # Desired response or target label, d(n)
  return x, d

# Step 3. Computation of Actual Response. Compute the actual response of the perceptron as _y(x) = sgn [summation of w(n)x(n)]__ where sgn(·) is the signum function.

def signum(v):
  return 1 if v >= 0 else -1

def linear_combiner(weights, x):
  m = len(x)
  v = 0.0
  for i in range(m):
    v += weights[i] * x[i]
  return v                                    # Linear combiner output, v(n)

# Step 4. Adaptation of Weight Vector. Update the weight vector of the perceptron to obtain w(n+1) = w(n) + eta[d(n)-y(n)]x(n)

def update_weights(weights, eta, d, y, x):
  return weights + eta * (d - y) * x

def create_headers(num_features):                              # Set up headers for the step table # x0, x1, ..., xm, w0, w1, ..., wm, v, y, d
  headers = [ f"x{i}" for i in range(num_features + 1)]
  headers += [ f"w{i}" for i in range(num_features + 1)]
  headers += ["v", "y", "d"]
  return headers

def decision_boundary(weights):
    w0, w1, w2 = weights[:3]
    print(f"From the final weights, we have :  ({w0:g}) + ({w1:g})x1 + ({w2:g})x2 = 0")
    if w2 != 0:
        m = - w1 / w2 + 0.0
        c = - w0 / w2 + 0.0
        print(f"Simplify      :  ({w2:g})x2 = ({-w0 + 0.0:g}) + ({-w1 + 0.0:g})x1")
        print(f"Solve for x2  :  x2 = ({m:g})x1 + ({c:g})")
        if w1 == 0:
            print(f"\nThis is a horizontal line at x2 = {c:g}")
        else:
            print(f"\nThis is a line with slope {m:g} and x2-intercept {c:g}")
    elif w1 != 0:
        print(f"Solve for x1:\n  x1 = {-w0 / w1:g}")
        print(f"\nThis is a vertical line at x1 = {-w0 / w1:g}")
    else:
        print("w1 = w2 = 0: no boundary defined.")

# Plotting

def plot_boundary_2d(training_data, weights, converged, title="Perceptron"):
  X = np.array([row[:-1] for row in training_data])          # inputs (x1, x2)
  D = np.array([row[-1] for row in training_data])           # desired responses d

  fig, ax = plt.subplots(figsize=(5, 5))
  ax.scatter(X[D == 1, 0],  X[D == 1, 1],  marker='o', s=120, c='tab:blue',   label='Class C1 (d = +1)')
  ax.scatter(X[D == -1, 0], X[D == -1, 1], marker='x', s=120, c='tab:orange', label='Class C2 (d = -1)')

  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  w0, w1, w2 = weights[:3]

  if converged:
    if w2 != 0:                                                # x2 = -(w1/w2)x1 - (w0/w2)
      x1_line = np.linspace(x_min, x_max, 100)
      x2_line = -(w1 / w2) * x1_line - (w0 / w2)
      ax.plot(x1_line, x2_line, 'k--', label='Decision boundary')
    elif w1 != 0:                                              # vertical line x1 = -w0/w1
      ax.axvline(-w0 / w1, color='k', linestyle='--', label='Decision boundary')
    ax.set_title(title)
  else:
    ax.set_title(f"{title}\n(no separator: did not converge)")

  ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
  ax.set_xlabel('x1'); ax.set_ylabel('x2')
  ax.axhline(0, color='gray', linewidth=0.5); ax.axvline(0, color='gray', linewidth=0.5)
  ax.grid(True, alpha=0.3)
  ax.legend(loc='upper right', fontsize=8)
  plt.show()

def plot_boundary_3d(training_data, weights, converged, title="Perceptron"):
  X = np.array([row[:-1] for row in training_data])          # inputs (x1, x2, x3)
  D = np.array([row[-1] for row in training_data])           # desired responses d
  w0, w1, w2, w3 = weights[:4]
  lo, hi = X.min() - 0.5, X.max() + 0.5

  fig = go.Figure()                                          # NEW var: plotly figure
  fig.add_trace(go.Scatter3d(x=X[D == 1, 0], y=X[D == 1, 1], z=X[D == 1, 2], mode='markers',
                             marker=dict(size=7, symbol='circle', color='blue'), name='Class C1 (d = +1)'))
  fig.add_trace(go.Scatter3d(x=X[D == -1, 0], y=X[D == -1, 1], z=X[D == -1, 2], mode='markers',
                             marker=dict(size=7, symbol='x', color='orange'), name='Class C2 (d = -1)'))

  if converged and w3 != 0:                                  # plane: x3 = -(w0 + w1x1 + w2x2) / w3
    x1_grid, x2_grid = np.meshgrid(np.linspace(lo, hi, 60), np.linspace(lo, hi, 60))
    x3_grid = -(w0 + w1 * x1_grid + w2 * x2_grid) / w3
    x3_grid[(x3_grid < lo) | (x3_grid > hi)] = np.nan        # clip plane to the plot box
    fig.add_trace(go.Surface(x=x1_grid, y=x2_grid, z=x3_grid, opacity=0.4,
                             colorscale=[[0, 'gray'], [1, 'gray']], showscale=False, name='Decision plane'))
    subtitle = f"Plane: x3 = {-w0/w3 + 0.0:g} + ({-w1/w3 + 0.0:g})x1 + ({-w2/w3 + 0.0:g})x2"
  elif converged:
    subtitle = "w3 = 0: plane is parallel to the x3 axis, not drawn"
  else:
    subtitle = "no separator: did not converge"

  fig.update_layout(title=f"{title}<br><sub>{subtitle}</sub>",
                    scene=dict(xaxis=dict(title='x1', range=[lo, hi]),
                               yaxis=dict(title='x2', range=[lo, hi]),
                               zaxis=dict(title='x3', range=[lo, hi]),
                               aspectmode='cube'),
                    width=650, height=600)
  fig.show()

def plot_boundary(training_data, weights, converged, title="Perceptron"):
  num_features = len(training_data[0]) - 1
  if num_features == 2:
    plot_boundary_2d(training_data, weights, converged, title)
  elif num_features == 3:
    plot_boundary_3d(training_data, weights, converged, title)
  else:
    print(f"Plotting skipped: {num_features} features cannot be drawn in 2D/3D.")

# **Perceptron**

def perceptron(input_lines, verbose=True, max_iter=100, title="Perceptron"):

  eta, b, training_data = parse_input(input_lines)                          # Unpack the parsed values
  num_features = len(training_data[0]) - 1                                   # Get the number of features (excluding the desired label d)
  weights = init_weights(num_features, b)                                   # Initialize weights
  headers = create_headers(num_features)                                     # Set up headers for the step table

  P_table = []

  iter = 0
  converged = False

  while not converged and iter < max_iter:                    # Apply the Perceptron Convergence Algorithm iteratively until the weights converge or the stopping condition is met.
    converged = True
    for row in training_data:
      print(f"\nIteration : {iter}")
      print(f"Weights : {weights}")
      print(pd.DataFrame(P_table, columns=headers))
      print(f"Training Sample : {row}")

      x, d = activate(row, num_features)
      v = linear_combiner(weights, x)                         # Linear combiner output, v(n)
      y = signum(v)                                           # Actual response or predicted output, y(n) hard limiter, +1 if v>=0, -1 otherwise
      P_table.append(list(x) + list(weights) + [v, y, d])
      if y != d:                                              # if y or quantized response is not equal to the desired response
        converged = False                                     # Continue until all training samples (per row) are correctly classified.
        weights = update_weights(weights, eta, d, y, x)       # Bias weight and the weights for each feature, [b, w1(n), w2(n), ... , wm(n)]
      #print(f"Updated Weights : {weights}")
      if verbose:
        print(f"Next Iteration {iter + 1} | x = {x} | d = {d:+g} | v = {v:+.2f} | y = {y:+d} | updated w = {weights}")
    iter += 1

  if converged:
    print(f"\nConverged, Iterations : {iter}.")
  else:
    print(f"\nDid not converge within {max_iter}.")
  print(f"\nFinal Adjusted weight : {weights}\n")
  if converged and num_features == 2:
    decision_boundary(weights)
  plot_boundary(training_data, weights, converged, title)

  #PCA = pd.DataFrame(P_table, columns=create_headers(num_features))

  return #PCA

# Lab Example

TEST = """
0.1
0.5
1 0 1
0 1 -1
"""

perceptron(TEST,verbose=True,title="TEST")

# AND Gate

AND = """
0.1
0.5
0 0 -1
0 1 -1
1 0 -1
1 1 1
"""

perceptron(AND,verbose=False,title="AND")

#OR Gate

OR = """
0.1
0.5
0 0 -1
0 1 1
1 0 1
1 1 1
"""

perceptron(OR,verbose=False,title="OR")

# NAND Gate

NAND = """
0.1
0.5
0 0 1
0 1 1
1 0 1
1 1 -1
"""

perceptron(NAND,verbose=False,title="NAND")

# NOR Gate

NOR = """
0.1
0.5
0 0 1
0 1 -1
1 0 -1
1 1 -1
"""

perceptron(NOR)

# XOR Gate

XOR = """
0.1
0.5
0 0 -1
0 1 1
1 0 1
1 1 -1
"""

perceptron(XOR)

# Test cases with m=3 features

# at least have 2 1's
TEST_7 = """
0.1
0.5
0 0 0 -1
0 0 1 -1
0 1 0 -1
0 1 1 1
1 0 0 -1
1 0 1 1
1 1 0 1
1 1 1 1
"""

perceptron(TEST_7,verbose=True, title="TEST_7")

# not((x0 and x1) or x2)
TEST_8 = """
0.1
0.5
0 0 0 1
0 0 1 -1
0 1 0 1
0 1 1 -1
1 0 0 1
1 0 1 -1
1 1 0 -1
1 1 1 -1
"""

perceptron(TEST_8, verbose=True, title="TEST_8")