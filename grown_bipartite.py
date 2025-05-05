import random
import igraph as ig
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

n = 1000

def get_grown_bipartite(args):
    p, n = args
    g = ig.Graph()
    g.add_vertex(type=False)
    g.add_vertex(type=True)
    for i in range(n-1):
        g.add_vertex(type=False)
        g.add_vertex(type=True)
        if random.random() < p:
            g.add_edge(random.randint(0, i+1)*2, random.randint(0, i+1)*2+1)
        if random.random() < p:
            g.add_edge(random.randint(0, i+1)*2, random.randint(0, i+1)*2+1)
    return g

def get_grown_bipartite_lcc(args):
    p, n = args
    g = ig.Graph()
    g.add_vertex(type=False)
    g.add_vertex(type=True)
    for i in range(n-1):
        g.add_vertex(type=False)
        g.add_vertex(type=True)
        if random.random() < p:
            g.add_edge(random.randint(0, i+1)*2, random.randint(0, i+1)*2+1)
        if random.random() < p:
            g.add_edge(random.randint(0, i+1)*2, random.randint(0, i+1)*2+1)
    return g.ecount()/g.vcount(), g.components().giant().vcount()/(2*n)

def get_grown_bipartite_dist(args):
    p, n = args
    g = ig.Graph()
    g.add_vertex(type=False)
    g.add_vertex(type=True)
    for i in range(n-1):
        g.add_vertex(type=False)
        g.add_vertex(type=True)
        if random.random() < p:
            g.add_edge(random.randint(0, i+1)*2, random.randint(0, i+1)*2+1)
        if random.random() < p:
            g.add_edge(random.randint(0, i+1)*2, random.randint(0, i+1)*2+1)
    return g.ecount()/g.vcount(), g.average_path_length()

if __name__ == "__main__":
    results = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, n) for i in np.linspace(0,1,200)]
        for _ in range(10):
            results.append(pool.map(get_grown_bipartite_lcc, iters))
    final_res = sorted(zip(np.array(results)[:,:,0].mean(axis = 0), np.array(results)[:,:,1].mean(axis = 0)))
    x = np.array([x for x,_ in final_res if x != 0])
    y = np.array([y for x,y in final_res if x != 0])
    plt.plot(x, y)
    plt.xlabel("Edges per Vertex")
    plt.ylabel("Largest Connected Component Size")
    plt.title("Size of Largest Connected Component v Edges per Vertex in Randomly Grown Bipartite Graph")
    plt.show()