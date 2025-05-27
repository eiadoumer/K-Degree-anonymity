import networkx as nx
import random
from collections import Counter
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import pulp
import urllib.request
import io
import zipfile
import time
start_time = time.time()



def anonymize_degree_sequence(G, k, t):
    """
    Generates a target anonymized degree sequence using (k, t)-degree anonymity.
    Groups node degrees in chunks of size k and assigns an adjusted degree within ±t of the group's average.
    
    Parameters:
        G (networkx.Graph): Input graph
        k (int): Minimum group size for degree anonymity
        t (int): Tolerance range (degrees within ±t can be grouped)

    Returns:
        List[int]: Anonymized degree sequence
    """
    degrees = sorted([deg for _, deg in G.degree()], reverse=True)  # Sorted degrees
    n = len(degrees)
    new_degrees = []

    for i in range(0, n, k):
        group = degrees[i:i + k]

        if len(group) < k:
    # Just assign remaining nodes to previous group's degree
            new_degrees.extend([new_degrees[-1]] * len(group))
            break

        avg_degree = np.mean(group)
        adjusted_degree = max(avg_degree - t, min(avg_degree + t, group[0]))
        new_degrees.extend([adjusted_degree] * k)

    return new_degrees[:n]


def ilp_vertex_split_realization(G, k, t):
    """
    Performs vertex splitting to transform graph G into a (k, t)-anonymized version.
    Splits nodes whose actual degree exceeds the target anonymized degree.

    Parameters:
        G (networkx.Graph): Input graph
        k (int): Minimum group size for degree anonymity
        t (int): Tolerance range (degrees within ±t can be grouped)

    Returns:
        Tuple[Graph, Graph, Dict]: Original graph, anonymized graph, mapping of split nodes
    """
    G = G.copy()
    target_degrees = anonymize_degree_sequence(G, k, t)
    nodes = list(G.nodes())
    actual_degrees = dict(G.degree())
    degree_diff = {
        node: actual_degrees[node] - target_degrees[i]
        for i, node in enumerate(nodes)
    }

    to_split = [node for node in nodes if degree_diff[node] > 0]
    print(f"Nodes to split for k={k}, t={t}: {to_split}")

    # ILP to minimize number of splits
    prob = pulp.LpProblem("Vertex_Splitting", pulp.LpMinimize)
    split_vars = {v: pulp.LpVariable(f"split_{v}", cat='Binary') for v in to_split}

    for node in to_split:
        if degree_diff[node] > 0:
            prob += split_vars[node] == 1, f"must_split_{node}"

    prob += pulp.lpSum(split_vars[v] for v in to_split)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    G_prime = G.copy()
    split_map = {}
    for v in to_split:
        if split_vars[v].varValue == 1:
            neighbors = list(G_prime.neighbors(v))
            G_prime.remove_node(v)

            # Try all 2-partitions and pick the one with most balanced degrees
            best_split = None
            best_diff = float('inf')
            for i in range(1, len(neighbors)):
                part1 = neighbors[:i]
                part2 = neighbors[i:]
                diff = abs(len(part1) - len(part2))
                if diff < best_diff:
                    best_split = (part1, part2)
                    best_diff = diff

            v1, v2 = f"{v}_1", f"{v}_2"
            G_prime.add_node(v1)
            G_prime.add_node(v2)
            for u in best_split[0]:
                G_prime.add_edge(v1, u)
            for u in best_split[1]:
                G_prime.add_edge(v2, u)

            split_map[v] = [v1, v2]

    print("Original Degree Sequence:", sorted(actual_degrees.values(), reverse=True))
    print("New Degree Sequence:", sorted([deg for _, deg in G_prime.degree()], reverse=True))

    return G, G_prime, split_map


def visualize_graphs(G_orig, G_split, split_map):
    """
    Plots the original and anonymized graphs side by side, coloring split nodes in orange.

    Parameters:
        G_orig (Graph): Original graph
        G_split (Graph): Graph after vertex splitting
        split_map (dict): Mapping of original to split nodes
    """
    pos = nx.spring_layout(G_orig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title("Original Graph")
    nx.draw(G_orig, pos, ax=axs[0], with_labels=True, node_color="lightblue", edge_color='gray')

    axs[1].set_title("Anonymized (Vertex-Split) Graph")
    pos_prime = nx.spring_layout(G_split, seed=42)
    colors = ['lightblue' if not any(n in v for v in split_map.values()) else 'orange' for n in G_split.nodes()]
    nx.draw(G_split, pos_prime, ax=axs[1], with_labels=True, node_color=colors, edge_color='gray')

    plt.tight_layout()
    plt.show()


def verify_k_t_anonymity(G):
    """
    Verifies that the input graph satisfies (k, t)-degree anonymity.

    Parameters:
        G (Graph): The graph to verify

    Returns:
        List[Tuple]: Nodes violating (k, t)-anonymity (if any)
    """
    degrees = [deg for _, deg in G.degree()]
    node_deg = dict(G.degree())

    violations = []
    for node, d in node_deg.items():
        in_group = sum(1 for deg in degrees if abs(deg - d) <= t)
        if in_group < k:
            violations.append((node, d, in_group))
            
        

    if not violations:
        print(f" The graph satisfies ({k}, {t}) degree anonymity.")
    else:
        print(f" The graph violates ({k}, {t}) degree anonymity at {len(violations)} node(s):")
        for node, deg, count in violations:
            print(f"  - Node {node} (deg={deg}) only has {count} nodes within ±{t} of its degree.")

    return violations


# --- Parameters and Execution ---

k = 4  # Minimum number of similar-degree nodes
t = 5  # Degree tolerance

# Generate example graph
G = nx.erdos_renyi_graph(1000, 0.4, seed=42)


# Run anonymization
G_orig, G_anonymized, split_map = ilp_vertex_split_realization(G, k, t)
violations = verify_k_t_anonymity(G_anonymized)
print(violations)

if len(G_orig.nodes()) < 100:
    visualize_graphs(G_orig, G_anonymized, split_map)
else:
    print("Graph too large to visualize.")
print("--- %s seconds to Run ---" % (time.time() - start_time))