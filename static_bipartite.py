import igraph as ig
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np

p = 0.008
n = 250

def get_static_bipartite(args):
    p, n, n2 = args
    g = ig.Graph.Random_Bipartite(n, n2, p=p)
    return g

def get_static_bipartite_lcc(args):
    p, n, n2 = args
    g = ig.Graph.Random_Bipartite(n, n2, p=p)
    return g.ecount()/g.vcount(), g.components().giant().vcount()/(n+n2)

def get_static_bipartite_dist(args):
    p, n, n2 = args
    g = ig.Graph.Random_Bipartite(n, n2, p=p)
    return g.ecount()/g.vcount(), g.average_path_length()

if __name__ == "__main__":
    results = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, n, n) for i in np.linspace(0,p,200)]
        for _ in range(10):
            results.append(pool.map(get_static_bipartite_lcc, iters))
    final_res = sorted(zip(np.array(results)[:,:,0].mean(axis = 0), np.array(results)[:,:,1].mean(axis = 0)))
    x = np.array([x for x,_ in final_res if x != 0])
    y = np.array([y for x,y in final_res if x != 0])
    plt.plot(x, y)
    plt.xlabel("Edges per Vertex")
    plt.ylabel("Largest Connected Component Size")
    plt.title("Size of Largest Connected Component v Edges per Vertex in Static Random Bipartite Graph")
    plt.show()