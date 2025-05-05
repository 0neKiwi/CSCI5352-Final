import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import sys
from tqdm import tqdm

p = 0.008
n = 250
trials = 10
colors = ["mediumpurple", "royalblue", "lightseagreen", "yellowgreen", "gold"]

def sort_results(res):
    sorted_res = sorted(zip(np.array(res)[:,:,0].mean(axis = 0), np.array(res)[:,:,1].mean(axis = 0)))
    x = np.array([x for x,_ in sorted_res if x != 0])
    y = np.array([y for x,y in sorted_res if x != 0])
    return x, y

if len(sys.argv) != 2:
    print("Usage: python plot_bipartite.py {lcc | apl}")
    exit(-1)
else:
    if sys.argv[1] == "lcc":
        from grown_simple import get_grown_graph_lcc as simple_f
        from static_bipartite import get_static_bipartite_lcc as static_f
        from grown_bipartite import get_grown_bipartite_lcc as sy_f
        from grown_bipartite_2 import get_grown_bipartite_2_lcc as asy1_f
        from grown_bipartite_3 import get_grown_bipartite_3_lcc as asy2_f
        text = "Largest Connected Component"
    elif sys.argv[1] == "apl":
        from grown_simple import get_grown_graph_dist as simple_f
        from static_bipartite import get_static_bipartite_dist as static_f
        from grown_bipartite import get_grown_bipartite_dist as sy_f
        from grown_bipartite_2 import get_grown_bipartite_2_dist as asy1_f
        from grown_bipartite_3 import get_grown_bipartite_3_dist as asy2_f
        text = "Average Path Length"
    else:
        exit(-1)

if __name__ == "__main__":
    print(f"Getting {text} for Randomly Grown Simple Graph...")
    results_simple = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, 2*n) for i in np.linspace(0,1,200)]
        for _ in tqdm(range(trials)):
            results_simple.append(pool.map(simple_f, iters))
    print(f"Getting {text} for Static Random Bipartite Graph...")
    results_static = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, n, n) for i in np.linspace(0,p,200)]
        for _ in tqdm(range(trials)):
            results_static.append(pool.map(static_f, iters))
    print(f"Getting {text} for Randomly Grown Bipartite Graph...")
    results_sy = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, n) for i in np.linspace(0,1,200)]
        for _ in tqdm(range(trials)):
            results_sy.append(pool.map(sy_f, iters))
    print(f"Getting {text} for Randomly Grown Bipartite Graph (Method 2)...")
    results_asy1 = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, n, n) for i in np.linspace(0,0.985/n,200)]
        for _ in tqdm(range(trials)):
            results_asy1.append(pool.map(asy1_f, iters))
    print(f"Getting {text} for Randomly Grown Bipartite Graph (Method 3)...")
    results_asy2 = []
    with mp.Pool(processes=10) as pool:
        iters = [(i, n, n) for i in np.linspace(0,3/(n*n),200)]
        for _ in tqdm(range(trials)):
            results_asy2.append(pool.map(asy2_f, iters))
    print("Starting Plotting...")
    x_simple, y_simple = sort_results(results_simple)
    x_static, y_static = sort_results(results_static)
    x_sy, y_sy = sort_results(results_sy)
    x_asy1, y_asy1 = sort_results(results_asy1)
    x_asy2, y_asy2 = sort_results(results_asy2)
    plt.plot(x_simple*2, y_simple, label="Randomly Grown Graph (Method A)", color=colors[0])
    plt.plot(x_static*2, y_static, label="Static Random Bipartite Graph (Method B)", color=colors[1])
    plt.plot(x_sy*2, y_sy, label="Randomly Grown Bipartite Graph (Method C)", color=colors[2])
    plt.plot(x_asy1*2, y_asy1, label="Randomly Grown Bipartite Graph (Method D)", color=colors[3])
    plt.plot(x_asy2*2, y_asy2, label="Randomly Grown Bipartite Graph (Method E)", color=colors[4])
    plt.legend()
    plt.xlabel("Mean Degree")
    plt.ylabel(text)
    plt.title(f"{text} v Mean Degree in Random Networks")
    plt.savefig(f"bipartite_{sys.argv[1]}.png")
    plt.show()