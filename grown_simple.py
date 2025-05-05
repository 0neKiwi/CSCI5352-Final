import random
import igraph as ig
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

n = 1000

def get_grown_graph(args):
    p, n = args
    g = ig.Graph()
    g.add_vertex()
    for i in range(n-1):
        g.add_vertex()
        if random.random() < p:
            v1, v2 = random.sample(range(i+2), 2)
            g.add_edge(v1, v2)
    return g

def get_grown_graph_lcc(args):
    p, n = args
    g = ig.Graph()
    g.add_vertex()
    for i in range(n-1):
        g.add_vertex()
        if random.random() < p:
            v1, v2 = random.sample(range(i+2), 2)
            g.add_edge(v1, v2)
    return g.ecount()/g.vcount(), g.components().giant().vcount()/(n+1)

def get_grown_graph_cc(args):
    p, n = args
    g = ig.Graph()
    g.add_vertex()
    for i in range(n-1):
        g.add_vertex()
        if random.random() < p:
            v1, v2 = random.sample(range(i+2), 2)
            g.add_edge(v1, v2)
    return g.ecount()/g.vcount(), g.transitivity_undirected()

def get_grown_graph_dist(args):
    p, n = args
    g = ig.Graph()
    g.add_vertex()
    for i in range(n-1):
        g.add_vertex()
        if random.random() < p:
            v1, v2 = random.sample(range(i+2), 2)
            g.add_edge(v1, v2)
    return g.ecount()/g.vcount(), g.average_path_length()

if __name__ == "__main__":
    results = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, n) for i in np.linspace(0,1,200)]
        for _ in range(10):
            results.append(pool.map(get_grown_graph_lcc, iters))
    final_res = sorted(zip(np.array(results)[:,:,0].mean(axis = 0), np.array(results)[:,:,1].mean(axis = 0)))
    x = np.array([x for x,_ in final_res if x != 0])
    y = np.array([y for x,y in final_res if x != 0])
    plt.plot(x, y)
    plt.xlabel("Edges per Vertex")
    plt.ylabel("Largest Connected Component Size")
    plt.title("Size of Largest Connected Component v Edges per Vertex in Randomly Grown Graph")
    plt.show()