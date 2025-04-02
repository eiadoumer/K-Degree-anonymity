

import networkx as nx
import numpy as np
import pulp
import matplotlib.pyplot as plt
from collections import Counter

def anonymize_degree_sequence(G, k, t):
    
    # Get degrees sorted in descending order
    degrees = sorted([deg for _, deg in G.degree()], reverse=True)  
    n = len(degrees)
    new_degrees = []
   
    # Process degrees in groups of size k
    for i in range(0, n, k):
        group = degrees[i:i + k]  # Get current group of k degrees
        
        # Handle final group if smaller than k
        if len(group) < k:
            # Pad with last adjusted degree to maintain group size
            new_degrees.extend([new_degrees[-1]] * (k - len(group)))
            new_degrees.extend(group)  # Add actual remaining degrees
            break

        # Calculate average and adjust degrees within [avg-t, avg+t] range
        avg_degree = np.mean(group)
        adjusted_degree = max(avg_degree - t, min(avg_degree + t, group[0]))

        # Apply adjusted degree to all in group
        new_degrees.extend([adjusted_degree] * k)
        
    return new_degrees[:n] # Return only n degrees (remove padding if any)


def ilp_graph_realization(G, target_degrees):
    
    G_prime = G.copy()  # Work on a copy
    current_degrees = dict(G.degree())
    
    # Calculate required degree changes for each node
    degree_changes = {node: target_degrees[i] - current_degrees[node] 
                     for i, node in enumerate(G.nodes())}

    # Setup ILP problem
    prob = pulp.LpProblem("Graph_Realization", pulp.LpMinimize)
    
    # All possible edges to add (non-edges) and remove (existing edges)
    edges = list(nx.non_edges(G)) + list(G.edges())
    
    # Binary variables for each possible edge modification
    x = {e: pulp.LpVariable(f"x_{e[0]}_{e[1]}", cat=pulp.LpBinary) for e in edges}

    # Objective: minimize total edge modifications
    prob += pulp.lpSum(x[e] for e in edges)

    # Constraints: net degree change must match target for each node
    for node in G.nodes():
        prob += (
            pulp.lpSum(x[e] for e in edges if node in e) == degree_changes[node]
        )

    # Solve the ILP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Apply the solution to the graph
    for e in edges:
        if x[e].varValue == 1:  # If edge modification is selected
            if G_prime.has_edge(*e):
                G_prime.remove_edge(*e)  # Delete existing edge
            else:
                G_prime.add_edge(*e)  # Add new edge

    return G_prime


def plot_graph():
    # Create a random graph with 8 nodes and 40% connection probability
    G = nx.erdos_renyi_graph(8, 0.4)
    
    # Anonymization parameters
    k = 3  # Each degree should appear at least k times
    t = 3  # Degrees can vary ±t from group average

    print("Original Degree Sequence:", sorted([d for _, d in G.degree()], reverse=True))

    # Generate anonymized degree sequence
    new_degrees = anonymize_degree_sequence(G, k, t)
    print("Anonymized Degree Sequence:", new_degrees)

    # Create anonymized graph
    G_anonymized = ilp_graph_realization(G, new_degrees)
    
    # Print comparison statistics
    print("Final Degree Sequence:", sorted([d for _, d in G_anonymized.degree()], reverse=True))
    print(f"Number of nodes (Original): {G.number_of_nodes()}, (Anonymized): {G_anonymized.number_of_nodes()}")
    print(f"Number of edges (Original): {G.number_of_edges()}, (Anonymized): {G_anonymized.number_of_edges()}")
    print(f'Average degree (Original): {np.mean([d for n, d in G.degree()]):.2f}, (Anonymized): {np.mean([d for n, d in G_anonymized.degree()]):.2f}')
    print(f'Degree distribution (Original): {dict(Counter(sorted([d for n, d in G.degree()], reverse=True)))}')

    # Check connectivity and path lengths
    if nx.is_connected(G) and nx.is_connected(G_anonymized):
        print(f"Average shortest path (Original): {nx.average_shortest_path_length(G):.3f}, (Anonymized): {nx.average_shortest_path_length(G_anonymized):.3f}")
    else:
        print("Graphs are not connected, average shortest path cannot be computed.")
    
    print(f'Maximum degree (Original): {max(dict(G.degree()).values())}, (Anonymized): {max(dict(G_anonymized.degree()).values())}')

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Original Graph")

    plt.subplot(1, 2, 2)
    nx.draw(G_anonymized, with_labels=True, node_color='lightgreen', edge_color='gray')
    plt.title("Anonymized Graph")

    plt.show()
    
plot_graph()