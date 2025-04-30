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




# Step 3: ILP - splitting model
def ilp_vertex_split_realization(G, k, t):
    G = G.copy()
   
    target_degrees = anonymize_degree_sequence(G, k, t)
    nodes = list(G.nodes())
    actual_degrees = dict(G.degree())
    degree_diff = {
        node: actual_degrees[node] - target_degrees[i]
        for i, node in enumerate(nodes)
    }

    # Directly identify nodes that need splitting
    to_split = [node for node in nodes if degree_diff[node] > 0]

    print(f"Nodes to split for k={k}, t={t}: {to_split}")

    # 1. Generate target degree sequence
    target_degrees = anonymize_degree_sequence(G, k, t)
    nodes = list(G.nodes())
    actual_degrees = dict(G.degree())
    degree_diff = {
        node: actual_degrees[node] - target_degrees[i]
        for i, node in enumerate(nodes)
    }

    # ILP model: minimize number of splits
    prob = pulp.LpProblem("Vertex_Splitting", pulp.LpMinimize)
    split_vars = {v: pulp.LpVariable(f"split_{v}", cat='Binary') for v in to_split}

    # 2. Force splits when degree > target
    for node in to_split:
        if degree_diff[node] > 0:
            prob += split_vars[node] == 1, f"must_split_{node}"

    # 3. Objective: minimize optional splits
    prob += pulp.lpSum(split_vars[v] for v in to_split)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Apply splits
    G_prime = G.copy()
    split_map = {}
    for v in to_split:
        if split_vars[v].varValue == 1:
            neighbors = list(G_prime.neighbors(v))
            G_prime.remove_node(v)


            # splits the node into two nodes, v1 taking half, the rest goes to v2
            mid = len(neighbors) // 2
            v1, v2 = f"{v}_1", f"{v}_2"
            G_prime.add_node(v1)
            G_prime.add_node(v2)
            for u in neighbors[:mid]:
                G_prime.add_edge(v1, u)
            for u in neighbors[mid:]:
                G_prime.add_edge(v2, u)

            split_map[v] = [v1, v2]

    print("Original Degree Sequence:", sorted(actual_degrees.values(), reverse=True))
    print("New Degree Sequence:", sorted([deg for _, deg in G_prime.degree()], reverse=True))

    return G, G_prime, split_map




def visualize_graphs(G_orig, G_split, split_map):
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
    degrees = [deg for _, deg in G.degree()]
    node_deg = dict(G.degree())

    violations = []

    for node, d in node_deg.items():
        # Count how many nodes (including itself) have degree within ±t
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
  



# Anonymization parameters
 
k=3
t=2


# url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

# sock = urllib.request.urlopen(url)  # open URL
# s = io.BytesIO(sock.read())  # read into BytesIO "file"
# sock.close()

# zf = zipfile.ZipFile(s)  # zipfile object
# txt = zf.read("football.txt").decode()  # read info file
# gml = zf.read("football.gml").decode()  # read gml data
# # throw away bogus first line with # from mejn files
# gml = gml.split("\n")[1:]
# G = nx.parse_gml(gml)

G =nx.erdos_renyi_graph(15,0.4, seed=42)  # Example graph
G_orig, G_anonymized, split_map = ilp_vertex_split_realization(G, k, t)
violations = verify_k_t_anonymity(G_anonymized)
visualize_graphs(G_orig, G_anonymized, split_map)

