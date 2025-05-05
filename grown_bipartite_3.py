import random
import igraph as ig
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

n = 250

def get_grown_bipartite_3(args):
    p, n, n2 = args
    g = ig.Graph()
    p1 = n/(n+n2)
    type1 = []
    type2 = []
    while g.vcount() < n + n2:
        if random.random() < p1:
            type1.append(g.vcount())
            g.add_vertex(type=False)
        else:
            type2.append(g.vcount())
            g.add_vertex(type=True)
        if len(type1) > 0 and len(type2) > 0:
            for _ in range(len(type1)*len(type2)):
                if random.random() < p:
                    g.add_edge(random.choice(type1), random.choice(type2))
    return g

def get_grown_bipartite_3_lcc(args):
    p, n, n2 = args
    g = ig.Graph()
    p1 = n/(n+n2)
    type1 = []
    type2 = []
    while g.vcount() < n + n2:
        if random.random() < p1:
            type1.append(g.vcount())
            g.add_vertex(type=False)
        else:
            type2.append(g.vcount())
            g.add_vertex(type=True)
        if len(type1) > 0 and len(type2) > 0:
            for _ in range(len(type1)*len(type2)):
                if random.random() < p:
                    g.add_edge(random.choice(type1), random.choice(type2))
    return g.ecount()/g.vcount(), g.components().giant().vcount()/(n+n2)

def get_grown_bipartite_3_dist(args):
    p, n, n2 = args
    g = ig.Graph()
    p1 = n/(n+n2)
    type1 = []
    type2 = []
    while g.vcount() < n + n2:
        if random.random() < p1:
            type1.append(g.vcount())
            g.add_vertex(type=False)
        else:
            type2.append(g.vcount())
            g.add_vertex(type=True)
        if len(type1) > 0 and len(type2) > 0:
            for _ in range(len(type1)*len(type2)):
                if random.random() < p:
                    g.add_edge(random.choice(type1), random.choice(type2))
    return g.ecount()/g.vcount(), g.average_path_length()

if __name__ == "__main__":
    results = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, n, n) for i in np.linspace(0,2.85/(n*n),200)]
        for _ in range(1):
            results.append(pool.map(get_grown_bipartite_3_lcc, iters))
    final_res = sorted(zip(np.array(results)[:,:,0].mean(axis = 0), np.array(results)[:,:,1].mean(axis = 0)))
    x = np.array([x for x,_ in final_res if x != 0])
    y = np.array([y for x,y in final_res if x != 0])
    plt.plot(x, y)
    plt.xlabel("Edges per Vertex")
    plt.ylabel("Largest Connected Component Size")
    plt.title("Size of Largest Connected Component v Edges per Vertex in Randomly Grown Bipartite Graph")
    plt.show()