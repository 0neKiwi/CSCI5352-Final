import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import sys
from tqdm import tqdm

from static_bipartite import get_static_bipartite as static_graph
from grown_bipartite import get_grown_bipartite as sy_graph
from grown_bipartite_2 import get_grown_bipartite_2 as asy1_graph
from grown_bipartite_3 import get_grown_bipartite_3 as asy2_graph

p = 0.008
n = 250
trials = 10
colors = ["royalblue", "lightseagreen", "yellowgreen", "gold"]

def get_lcc(args):
    g_f, f_args = args
    g = g_f(f_args)
    proj1, _ = g.bipartite_projection()
    proj1_lcc = proj1.components().giant().vcount()
    proj1_nodes = proj1.vcount()
    return g.ecount()*2/g.vcount(), proj1.ecount()*2/proj1_nodes, proj1_lcc/proj1_nodes

def get_cc(args):
    g_f, f_args = args
    g = g_f(f_args)
    proj1, _ = g.bipartite_projection()
    proj1_cc = proj1.transitivity_undirected()
    return g.ecount()*2/g.vcount(), proj1.ecount()*2/proj1.vcount(), proj1_cc

def get_apl(args):
    g_f, f_args = args
    g = g_f(f_args)
    proj1, _ = g.bipartite_projection()
    proj1_apl = proj1.average_path_length()
    return g.ecount()*2/g.vcount(), proj1.ecount()*2/proj1.vcount(), proj1_apl

def get_deg_assort(args):
    g_f, f_args = args
    g = g_f(f_args)
    proj1, _ = g.bipartite_projection()
    proj1_deg_assort = proj1.assortativity_degree(directed=False)
    return g.ecount()*2/g.vcount(), proj1.ecount()*2/proj1.vcount(), proj1_deg_assort

def sort_results(res):
    sorted_res = sorted(zip(
        np.array(res)[:,:,0].mean(axis = 0), 
        np.array(res)[:,:,1].mean(axis = 0),
        np.array(res)[:,:,2].mean(axis = 0)
    ))
    x1 = np.array([x for x,_,_ in sorted_res if x != 0])
    y1 = np.array([y for x,y,_ in sorted_res if x != 0])
    y2 = np.array([y for x,_,y in sorted_res if x != 0])
    return x1, y1, y2

if len(sys.argv) != 2:
    print("Usage: python plot_bipartite.py {lcc | cc | apl | assort}")
    exit(-1)
else:
    if sys.argv[1] == "lcc":
        text = "Largest Connected Component"
        f = get_lcc
    elif sys.argv[1] == "cc":
        text = "Clustering Coefficient"
        f = get_cc
    elif sys.argv[1] == "apl":
        text = "Average Path Length"
        f = get_apl
    elif sys.argv[1] == "assort":
        text = "Degree Assortativity"
        f = get_deg_assort
    else:
        exit(-1)

if __name__ == "__main__":
    print(f"Getting {text} for Projection of Static Random Bipartite Graph...")
    results_static = []
    with mp.Pool(processes=10) as pool:
        iters = [(static_graph, (i, n, n)) for i in np.linspace(0,p,200)]
        for _ in tqdm(range(trials)):
            results_static.append(pool.map(f, iters))
    print(f"Getting {text} for Projection of Randomly Grown Bipartite Graph...")
    results_sy = []
    with mp.Pool(processes=10) as pool:
        iters = [(sy_graph, (i, n)) for i in np.linspace(0,1,200)]
        for _ in tqdm(range(trials)):
            results_sy.append(pool.map(f, iters))
    print(f"Getting {text} for Projection of Randomly Grown Bipartite Graph (Method 2)...")
    results_asy1 = []
    with mp.Pool(processes=10) as pool:
        iters = [(asy1_graph, (i, n, n)) for i in np.linspace(0,0.985/n,200)]
        for _ in tqdm(range(trials)):
            results_asy1.append(pool.map(f, iters))
    print(f"Getting {text} for Projection of Randomly Grown Bipartite Graph (Method 3)...")
    results_asy2 = []
    with mp.Pool(processes=10) as pool:
        iters = [(asy2_graph, (i, n, n)) for i in np.linspace(0,3/(n*n),200)]
        for _ in tqdm(range(trials)):
            results_asy2.append(pool.map(f, iters))
    print("Starting Plotting...")
    x_static, y1_static, y2_static = sort_results(results_static)
    x_sy, y1_sy, y2_sy = sort_results(results_sy)
    x_asy1, y1_asy1, y2_asy1 = sort_results(results_asy1)
    x_asy2, y1_asy2, y2_asy2 = sort_results(results_asy2)
    plt.figure(figsize=(8,5))
    plt.plot(x_static, y1_static, label="Static Random Bipartite Graph (Method B)", color=colors[0])
    plt.plot(x_sy, y1_sy, label="Randomly Grown Bipartite Graph (Method C)", color=colors[1])
    plt.plot(x_asy1, y1_asy1, label="Randomly Grown Bipartite Graph (Method D)", color=colors[2])
    plt.plot(x_asy2, y1_asy2, label="Randomly Grown Bipartite Graph (Method E)", color=colors[3])
    plt.legend()
    plt.xlabel("Mean Degree in Original Bipartite Graph")
    plt.ylabel(f"Mean Degree in Projection")
    plt.title(f"Mean Degree in Projection v Mean Degree in Original Bipartite Graph")
    plt.tight_layout()
    plt.savefig(f"projection_deg.png")
    plt.show()
    plt.figure(figsize=(8,5))
    plt.plot(x_static, y2_static, label="Projection of Static Random Bipartite Graph (Method B)", color=colors[0])
    plt.plot(x_sy, y2_sy, label="Projection of Randomly Grown Bipartite Graph (Method C)", color=colors[1])
    plt.plot(x_asy1, y2_asy1, label="Projection of Randomly Grown Bipartite Graph (Method D)", color=colors[2])
    plt.plot(x_asy2, y2_asy2, label="Projection of Randomly Grown Bipartite Graph (Method E)", color=colors[3])
    plt.legend()
    plt.xlabel("Mean Degree in Original Bipartite Graph")
    plt.ylabel(f"{text} in Projection")
    plt.title(f"{text} in Projection v Mean Degree in Original Bipartite Graph")
    plt.tight_layout()
    plt.savefig(f"projection_{sys.argv[1]}.png")
    plt.show()