import os
import math
import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Load the character connection data
with open(os.path.join("data", "char_counts.pkl"), "rb") as f:
    char_counts = pickle.load(f)

# Define QWERTY keyboard layout positions
keyboard_layout = {
    "q": (0, 2),
    "w": (1, 2),
    "e": (2, 2),
    "r": (3, 2),
    "t": (4, 2),
    "y": (5, 2),
    "u": (6, 2),
    "i": (7, 2),
    "o": (8, 2),
    "p": (9, 2),
    "a": (0.3, 1),
    "s": (1.3, 1),
    "d": (2.3, 1),
    "f": (3.3, 1),
    "g": (4.3, 1),
    "h": (5.3, 1),
    "j": (6.3, 1),
    "k": (7.3, 1),
    "l": (8.3, 1),
    "z": (0.6, 0),
    "x": (1.6, 0),
    "c": (2.6, 0),
    "v": (3.6, 0),
    "b": (4.6, 0),
    "n": (5.6, 0),
    "m": (6.6, 0),
}

# Define adjacent letter pairs
adjacent_pairs = set()
rows = [
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    ["z", "x", "c", "v", "b", "n", "m"],
]

# Add horizontal adjacents
for row in rows:
    for i in range(len(row) - 1):
        adjacent_pairs.add((row[i], row[i + 1]))
        adjacent_pairs.add((row[i + 1], row[i]))

# Add vertical adjacents
for i in range(len(rows) - 1):
    for j in range(min(len(rows[i]), len(rows[i + 1]))):
        adjacent_pairs.add((rows[i][j], rows[i + 1][j]))
        adjacent_pairs.add((rows[i + 1][j], rows[i][j]))

# Add diagonal adjacents (fixed to handle varying row lengths)
for i in range(len(rows) - 1):
    current_row = rows[i]
    next_row = rows[i + 1]

    # Forward diagonal (top-left to bottom-right)
    for j in range(len(current_row)):
        if j + 1 < len(next_row):  # Check if the diagonal position exists
            adjacent_pairs.add((current_row[j], next_row[j + 1]))
            adjacent_pairs.add((next_row[j + 1], current_row[j]))

    # Backward diagonal (top-right to bottom-left)
    for j in range(len(current_row)):
        if j > 0 and j - 1 < len(next_row):  # Check if the diagonal position exists
            adjacent_pairs.add((current_row[j], next_row[j - 1]))
            adjacent_pairs.add((next_row[j - 1], current_row[j]))

# Create a directed graph
G = nx.DiGraph()

# Add nodes (only letters)
letters = sorted([char for char in char_counts.keys() if char.isalpha()])
G.add_nodes_from(letters)

# Calculate threshold based on percentiles
all_weights = []
for source in char_counts:
    if not source.isalpha():
        continue
    for target, weight in char_counts[source].items():
        if not target.isalpha():
            continue
        if (
            weight > 0 and (source, target) not in adjacent_pairs
        ):  # Exclude adjacent pairs
            all_weights.append(weight)

# Set threshold to 75th percentile
threshold = np.percentile(all_weights, 75)

# Add edges with weights above threshold
for source in char_counts:
    if not source.isalpha():
        continue
    for target, weight in char_counts[source].items():
        if not target.isalpha():
            continue
        if (
            weight >= threshold and (source, target) not in adjacent_pairs
        ):  # Only add edges above threshold and not adjacent
            G.add_edge(source, target, weight=weight)

# Calculate edge weights for visualization
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
max_weight = max(edge_weights)
edge_weights = [
    math.log(w / max_weight * 5 + 1) for w in edge_weights
]  # Log scale for better visualization

# Create position dictionary for all letters
pos = {}
for letter in letters:
    if letter in keyboard_layout:
        pos[letter] = keyboard_layout[letter]
    else:
        # This should never happen now since we filtered non-letters
        pos[letter] = (0, 3)  # Fallback position

# Set up the plot with a dark background
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(-1, 10)
ax.set_ylim(-1, 4)  # Adjusted y-axis limits to accommodate non-letter characters

# Draw the network with improved aesthetics
nodes = nx.draw_networkx_nodes(
    G,
    pos,
    node_size=500,  # Reduced node size
    node_color="#1f78b4",
    alpha=0.8,
    edgecolors="white",
    linewidths=2,
    ax=ax,
)

# Draw edges with gradient colors based on weight
edges = nx.draw_networkx_edges(
    G,
    pos,
    width=edge_weights,
    edge_color=edge_weights,
    edge_cmap=plt.cm.viridis,
    alpha=0.6,
    arrows=True,
    arrowsize=12,  # Smaller arrow size
    arrowstyle="-|>",  # More compact arrow style
    connectionstyle="arc3,rad=0.2",  # More pronounced curve
    node_size=500,  # Match node size
    ax=ax,
)

# Add node labels with improved styling
nx.draw_networkx_labels(
    G,
    pos,
    font_size=12,  # Slightly smaller font
    font_weight="bold",
    font_color="white",
    ax=ax,
)

# Add edge labels for weights (only for strongest connections)
edge_labels = {
    (u, v): f"{G[u][v]['weight']:.1f}"
    for u, v in G.edges()
    if G[u][v]["weight"] >= np.percentile(all_weights, 90)
}

# Add title and adjust layout
ax.set_title(
    f"Typo Network (non-adjacent keys only)", fontsize=20, pad=20, color="white"
)

ax.axis("off")

# Add a colorbar for edge weights
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.viridis,
    norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)),
)
sm.set_array([])  # This line is important for the colorbar to work
cbar = plt.colorbar(
    sm,
    ax=ax,
    label="Connection Strength",
    fraction=0.012,  # Make the colorbar shorter
    pad=0.01,
)  # Reduce padding
cbar.ax.tick_params(labelsize=8)  # Make the tick labels smaller

# Save the plot
plt.savefig(
    os.path.join("data", "letter_network_no_adjacent.png"),
    dpi=300,
    bbox_inches="tight",
    facecolor="black",
)
plt.close()
