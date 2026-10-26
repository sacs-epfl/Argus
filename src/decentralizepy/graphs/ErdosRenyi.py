import networkx as nx

from decentralizepy.graphs.Graph import Graph


class ErdosRenyi(Graph):
    """
    The class for generating an Erdős-Rényi-Gilbert G(n, p) random graph.

    """

    def __init__(self, n_procs, p, seed=None):
        """
        Parameters
        ----------
        n_procs : int
            total number of nodes in the graph
        p : float
            Probability of edge creation between any pair of nodes.
            Expected degree per node is p * (n_procs - 1).
        seed : int, optional
            Seed for random number generator

        """
        super().__init__(n_procs)
        G = nx.erdos_renyi_graph(n_procs, p, seed=seed)
        for i, l in G.adjacency():
            self.adj_list[i] = set(l)
        if not nx.is_connected(G):
            self.connect_graph()
