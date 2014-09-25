from operator import itemgetter
from itertools import permutations
import networkx as nx

def full_degree_sequences(n):
    """
    n: number of vertex
    """

    first = []
    for i in range(1, n+1):
        first.extend([j for j in range(1, n)])

    return set(permutations(first, n))
    #return permutations(first, n)

def maximum(d):
    max_val = max(d.iteritems(), key=itemgetter(1))[1]
    max_vertices = []
    for vertex, value in d.iteritems():
        if value == max_val:
            max_vertices.append(vertex)
    return [max_val, max_vertices]

def distinct(g):
    bc = nx.betweenness_centrality(g)
    cc = nx.closeness_centrality(g)
    dc = nx.degree_centrality(g)
    ec = nx.eigenvector_centrality(g)

    return [maximum(bc), maximum(cc), maximum(dc), maximum(ec)]

def job(degree_sequence):
    try:
        G = nx.configuration_model(degree_sequence)
        G = nx.Graph(G) # to remove patallel edges
        G.remove_edges_from(G.selfloop_edges())
        distinct_prop = distinct(G)

        only_best_list = [i for x in zip(*distinct_prop)[1] if len(x)==1 for i in x]
        if len(set(only_best_list)) >= 2:
            len_best = [len(x) for x in zip(*distinct_prop)[1]]
            num_uniq = len(set([i for x in zip(*distinct_prop)[1] for i in x]))
            if sum(len_best) < 7 and len_best.count(0) == 0 and num_uniq >= 3:
                with open("result.txt", "a") as f:
                    print ""
                    print num_uniq, zip(*distinct_prop)
                    print G.nodes(), G.edges()
                    print ""
                    print>>f, num_uniq, zip(*distinct_prop)
                    print>>f, G.nodes(), G.edges()
    except nx.exception.NetworkXError, e:
        #print e, degree_sequence
        pass

if __name__=="__main__":
    max_n = 7

    for n in range(4, max_n+1):
        print "=== num of vertex:", n
        for degree_sequence in full_degree_sequences(n):
            job(degree_sequence)