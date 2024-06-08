import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pulp

def solve_transshipment_problem(nodes, supply, demand, cost, plot_network=False, plot_bar_chart=False):
    """
    Solves the transhipment problem given nodes, supply, demand, and cost data.
    Optionally plots the transhipment network and a stacked bar chart.

    Parameters:
    nodes (dict): Dictionary containing lists of source nodes, transhipment nodes, and destination nodes.
    supply (dict): Supply values for each source node.
    demand (dict): Demand values for each destination node.
    cost (dict): Cost of transportation between nodes.
    plot_network (bool): Whether to plot the transhipment network. Default is False.
    plot_bar_chart (bool): Whether to plot the stacked bar chart. Default is False.

    Returns:
    results_df (pd.DataFrame): DataFrame containing the flow values.
    total_cost (float): Total cost of the solution.
    """
    sources = nodes['sources']
    transshipments = nodes['transshipments']
    destinations = nodes['destinations']

    # Define the problem
    problem = pulp.LpProblem("Transshipment_Problem", pulp.LpMinimize)

    # Decision variables
    flow = {(i, j): pulp.LpVariable(f"flow_{i}_{j}", lowBound=0, cat='Continuous') for (i, j) in cost}

    # Objective function
    problem += pulp.lpSum(cost[i, j] * flow[i, j] for (i, j) in cost), "Total Cost"

    # Supply constraints
    for i in sources:
        problem += pulp.lpSum(flow[i, j] for j in transshipments) <= supply[i], f"Supply_{i}"

    # Demand constraints
    for j in destinations:
        problem += pulp.lpSum(flow[i, j] for i in transshipments) >= demand[j], f"Demand_{j}"

    # Transshipment constraints
    for k in transshipments:
        problem += pulp.lpSum(flow[i, k] for i in sources) == pulp.lpSum(flow[k, j] for j in destinations), f"Transshipment_{k}"

    # Solve the problem
    problem.solve()

    # Store results in DataFrame
    results = [(v.name, v.varValue) for v in problem.variables()]
    results_df = pd.DataFrame(results, columns=['Variable', 'Value'])

    # Extract total cost
    total_cost = pulp.value(problem.objective)

    # Optionally plot the network
    if plot_network:
        plot_transshipment_network(nodes, supply, demand, cost, flow)

    # Optionally plot the stacked bar chart
    if plot_bar_chart:
        plot_stacked_bar_chart(flow, nodes)
    
    return results_df, total_cost

def plot_transshipment_network(nodes, supply, demand, cost, flow):
    """
    Plots the transhipment network with the given nodes, supply, demand, cost, and flow values.

    Parameters:
    nodes (dict): Dictionary containing lists of source nodes, transhipment nodes, and destination nodes.
    supply (dict): Supply values for each source node.
    demand (dict): Demand values for each destination node.
    cost (dict): Cost of transportation between nodes.
    flow (dict): Flow values between nodes as decision variables from the optimisation result.
    """
    sources = nodes['sources']
    transshipments = nodes['transshipments']
    destinations = nodes['destinations']

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for node in sources + transshipments + destinations:
        G.add_node(node)

    # Add edges with weights
    for (i, j) in cost:
        G.add_edge(i, j, weight=cost[i, j], capacity=flow[i, j].varValue)

    # Custom positions for the nodes
    pos = {
        'F1': (0, 3), 'F2': (0, 2), 'F3': (0, 1), 'F4': (0, 0), 'F5': (0, -1),
        'XD1': (1, 1), 'XD2': (1, -1),
        'DC1': (2, 2), 'DC2': (2, 1), 'DC3': (2, 0), 'DC4': (2, -1), 'DC5': (2, -2)
    }

    # Draw the network graph
    plt.figure(figsize=(14, 8))  # Adjust the figure size as needed
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

    # Draw edge labels with the flow values
    edge_labels = {(i, j): f"{flow[i, j].varValue:.1f}" for (i, j) in cost}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('Transshipment Network Flow')
    plt.show()

def plot_stacked_bar_chart(flow, nodes):
    """
    Plots a stacked bar chart showing the flow values between nodes.

    Parameters:
    flow (dict): Flow values between nodes as decision variables from the optimisation result.
    nodes (dict): Dictionary containing lists of source nodes, transhipment nodes, and destination nodes.
    """
    sources = nodes['sources']
    transshipments = nodes['transshipments']
    destinations = nodes['destinations']

    # Create a DataFrame to hold flow values for plotting
    flow_data = {
        'From': [],
        'To': [],
        'Flow': []
    }

    for (i, j), var in flow.items():
        flow_data['From'].append(i)
        flow_data['To'].append(j)
        flow_data['Flow'].append(var.varValue)

    flow_df = pd.DataFrame(flow_data)

    # Create separate DataFrames for each category
    flow_sources_to_transshipments = flow_df[flow_df['From'].isin(sources)]
    flow_transshipments_to_destinations = flow_df[flow_df['From'].isin(transshipments)]

    # Plot the stacked bar chart for sources to transshipments
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bottom = pd.Series([0] * len(transshipments), index=transshipments)
    for source in sources:
        values = flow_sources_to_transshipments[flow_sources_to_transshipments['From'] == source].set_index('To')['Flow']
        ax.bar(transshipments, values, bottom=bottom[transshipments], label=source)
        bottom += values

    ax.set_xlabel('Transshipment Nodes')
    ax.set_ylabel('Flow Values')
    ax.set_title('Flow from Sources to Transshipment Nodes')
    ax.legend(title='From')

    plt.show()

    # Plot the stacked bar chart for transshipments to destinations
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = pd.Series([0] * len(destinations), index=destinations)
    for transshipment in transshipments:
        values = flow_transshipments_to_destinations[flow_transshipments_to_destinations['From'] == transshipment].set_index('To')['Flow']
        ax.bar(destinations, values, bottom=bottom[destinations], label=transshipment)
        bottom += values

    ax.set_xlabel('Destination Nodes')
    ax.set_ylabel('Flow Values')
    ax.set_title('Flow from Transshipment Nodes to Destinations')
    ax.legend(title='From')

    plt.show()