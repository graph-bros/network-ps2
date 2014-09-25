# Reference:
# http://networkx.github.io/documentation/networkx-1.9.1/
# reference/algorithms.centrality.html

import operator
import functools
import itertools
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

families = ["Acciaiuoli", "Albizzi", "Barbadori", "Bischeri", "Castellani",
            "Ginori", "Guadagni", "Lamberteschi", "Medici", "Pazzi",
            "Peruzzi", "Pucci", "Ridolfi", "Salviati", "Strozzi",
            "Tornabuoni"]

relationships = [[8, 9], [5,6,8], [4,8], [6,10,14], [2,10,14],
                 [1], [1,3,7,15], [6], [0,1,2,12,13,15], [13],
                 [3,4,14], [], [8,14,15], [8,9], [3,4,10,12],
                 [6,8,12]]

def draw_graph(vertices, edges):
    G = nx.Graph()
    G.add_nodes_from(vertices)

    for idx, edge in enumerate(edges):
        for dest in list(edge):
            G.add_edge(families[idx], families[dest])

    return G

def descending(node_value_dict):
    return sorted(node_value_dict.iteritems(),
                  key=operator.itemgetter(1),
                  reverse=True)


def comb_small_networks(vertices, edges, n):
    vertices_list = []
    edges_list = []
    for combi in itertools.combinations(vertices, n):
        vertices_index = []
        for vertex in combi:
            vertices_index.append(vertices.index(vertex))

        rel_set = []
        for index in vertices_index:
            remain = set(edges[index]).intersection(set(vertices_index))
            rel_set.append(list(remain))

        vertices_list.append(vertices_index)
        edges_list.append(rel_set)

    not_empty_vertices_list = []
    not_empty_edges_list = []
    for index, edge_list in enumerate(edges_list):
        if len([edge for edges in edge_list for edge in edges]) != 0:
            not_empty_vertices_list.append(vertices_list[index])
            not_empty_edges_list.append(edges_list[index])

    return not_empty_vertices_list, not_empty_edges_list

def find_max_degree(vertices, edges, n=2, centrality_type="degree"):
    max_val = 0
    max_vertice_list = []
    max_edges_list = []
    vertice_lists, edges_lists = comb_small_networks(vertices, edges, n)
    for index, vertice_list in enumerate(vertice_lists):
        g = draw_graph(vertice_list, edges_lists[index])

        if centrality_type == "degree":
            centrality = nx.degree_centrality(g)
        elif centrality_type == "eigenvector":
            centrality = nx.eigenvector_centrality(g)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(g)
        elif centrality_type == "closeness":
            centrality = nx.harmonic_centrality(g)

        if max(centrality.values()) > max_val:
            max_val = max(centrality.values())
            max_vertice_list = vertice_list
            max_edges_list = edges_lists[index]

    return max_val, max_vertice_list

def harmonic_centrality(G, u=None, distance=None, normalized=True):
    """
    Disclaimer: This code is almost same with closeness_centrality by networkx.
                The only difference is sum and devide by number of node for
                calculate harmonic centrality.
    """
    if distance is not None:
        # use Dijkstra's algorithm with specified attribute as edge weight
        path_length = functools.partial(nx.single_source_dijkstra_path_length,
                                        weight=distance)
    else:
        path_length = nx.single_source_shortest_path_length

    if u is None:
        nodes = G.nodes()
    else:
        nodes = [u]
    harmonic_centrality = {}
    for n in nodes:
        sp = path_length(G,n)
        totsp = sum([1/x for x in sp.values() if x!=0])
        if totsp > 0.0 and len(G) > 1:
            harmonic_centrality[n] = totsp / (len(G.nodes())-1.0)
            # normalize to number of nodes-1 in connected part
            if normalized:
                s = (len(sp)-1.0) / ( len(G) - 1 )
                harmonic_centrality[n] *= s
        else:
            harmonic_centrality[n] = 0.0
    if u is not None:
        return harmonic_centrality[u]
    else:
        return harmonic_centrality

def graph_centraliity(g, centrality_type):
    if centrality_type == "degree":
        centrality = nx.degree_centrality(g)
    elif centrality_type == "eigenvector":
        centrality = nx.eigenvector_centrality(g)
    elif centrality_type == "betweenness":
        centrality = nx.betweenness_centrality(g)
    elif centrality_type == "harmonic":
        centrality = harmonic_centrality(g)

    return pd.Series(centrality, name=centrality_type)

def do_ps2_five_one():
    centrality_types = ["degree", "eigenvector", "betweenness", "harmonic"]
    g = draw_graph(families, relationships)
    centralities = []
    for centrality_type in centrality_types:
        centrality = graph_centraliity(g, centrality_type)
        print centrality.order(ascending=False)


if __name__=="__main__":
    do_ps2_five_one()
