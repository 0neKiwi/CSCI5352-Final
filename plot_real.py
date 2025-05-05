import sys
import igraph as ig
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from scipy.stats import ttest_1samp as ttest
from grown_bipartite_3 import get_grown_bipartite_3 as asy2_graph
from grown_bipartite_3 import get_grown_bipartite_3_lcc as asy2_lcc_f
from grown_bipartite_3 import get_grown_bipartite_3_dist as asy2_apl_f
from grown_bipartite_2 import get_grown_bipartite_2 as asy1_graph
from grown_bipartite_2 import get_grown_bipartite_2_lcc as asy1_lcc_f
from grown_bipartite_2 import get_grown_bipartite_2_dist as asy1_apl_f
from static_bipartite import get_static_bipartite as static_graph
from static_bipartite import get_static_bipartite_lcc as static_lcc_f
from static_bipartite import get_static_bipartite_dist as static_apl_f

networks = {"dupont":"Dupont Plant Pollinator Network", "brunson":"Brunson CEO Club Membership Network"}

if len(sys.argv) != 2:
    print("Usage: python plot_real.py {dupont | brunson}")
    exit(-1)
else:
    network = sys.argv[1]
    if network == "dupont":
        with open(network+".txt") as f:
            lines = f.readlines()
        adj = [l.split() for l in lines]
        adj = [[int(plant) for plant in pol] for pol in adj]
        nodes1 = len(adj)
        nodes2 = len(adj[0])
        g_actual = ig.Graph.Biadjacency(adj)
        asy1_p = 0.08775
        asy2_p = 0.01553
        static_p = 0.2535
    elif network == "brunson":
        with open(network+".txt") as f:
            lines = f.readlines()
        _, _, nodes1, nodes2 = lines[1].split()
        nodes1 = int(nodes1)
        nodes2 = int(nodes2)
        edges = [l.split() for l in lines[2:]]
        edges = [(int(e[0])-1, nodes1+int(e[1])-1) for e in edges]
        g_actual = ig.Graph(edges=edges)
        g_actual.vs["type"] = [False]*nodes1 + [True]*nodes2
        asy1_p = 0.1167
        asy2_p = 0.01903
        static_p = 0.25342
    else:
        exit(-1)

def proj_f(args):
    g_f, f_args = args
    g = g_f(f_args)
    proj1, proj2 = g.bipartite_projection()
    proj1_nodes = proj1.vcount()
    proj2_nodes = proj2.vcount()
    proj1_lcc = proj1.components().giant().vcount()
    proj2_lcc = proj2.components().giant().vcount()
    proj1_cc = proj1.transitivity_undirected()
    proj2_cc = proj2.transitivity_undirected()
    proj1_apl = proj1.average_path_length()
    proj2_apl = proj2.average_path_length()
    proj1_deg_assort = proj1.assortativity_degree()
    proj2_deg_assort = proj2.assortativity_degree()
    return [proj1.ecount()*2/proj1_nodes, 
            proj2.ecount()*2/proj2_nodes, 
            proj1_lcc, proj2_lcc, 
            proj1_cc, proj2_cc, 
            proj1_apl, proj2_apl,
            proj1_deg_assort, proj2_deg_assort]

def plot_hist(axis, xlabel, static, asy1, asy2, actual, bins=None, xticks=None, projection=None, debug=True):
    if debug:
        print(xlabel, f"Projection (Side {projection})" if projection is not None else "")
        if (np.isnan(np.sum(static))):
            print("Removing NaN from static...")
            static = static[~np.isnan(static)]
        if (np.isnan(np.sum(asy1))):
            print("Removing NaN from asy1...")
            asy1 = asy1[~np.isnan(asy1)]
        if (np.isnan(np.sum(asy2))):
            print("Removing NaN from asy2...")
            asy2 = asy2[~np.isnan(asy2)]
        res = ttest(static, actual)
        print("Static:", np.mean(static), "p:", res.pvalue)
        res = ttest(asy1, actual)
        print("Asy1:", np.mean(asy1), "p:", res.pvalue)
        res = ttest(asy2, actual)
        print("Asy2:", np.mean(asy2), "p:", res.pvalue)
        print("Actual:", actual)
    prefix = f"Projection of " if projection is not None else ""
    label_static = f"{prefix}Static Random Bipartite Graph (Method B)"
    label_asy1 = f"{prefix}Randomly Grown Bipartite Graph (Method D)"
    label_asy2 = f"{prefix}Randomly Grown Bipartite Graph (Method E)"
    label_act = networks[network]
    colors_static = ["deeppink", "white"]
    colors_asy1 = ["mediumpurple", "white"]
    colors_asy2 = ["mediumseagreen", "white"]
    axis.hist(static, label=label_static, bins=bins, align="mid", color=colors_static[0], edgecolor=colors_static[1], alpha=1)
    axis.hist(asy1, label=label_asy1, bins=bins, align="mid", color=colors_asy1[0], edgecolor=colors_asy1[1], alpha=0.5)
    axis.hist(asy2, label=label_asy2, bins=bins, align="mid", color=colors_asy2[0], edgecolor=colors_asy2[1], alpha=0.25)
    axis.axvline(x=actual, label=label_act, color="mediumvioletred")
    axis.set_xlabel(xlabel)
    if xticks is not None:
        axis.set_xticks(xticks)

if __name__ == "__main__":
    
    nodes = nodes1 + nodes2
    g_deg = sum(g_actual.degree())/nodes
    g_lcc = g_actual.components().giant().vcount()
    g_apl = g_actual.average_path_length()

    g1, g2 = g_actual.bipartite_projection()
    g1_deg = sum(g1.degree())/g1.vcount()
    g1_lcc = g1.components().giant().vcount()
    g1_cc = g1.transitivity_undirected()
    g1_apl = g1.average_path_length()
    g1_assort = g1.assortativity_degree()
    g2_deg = sum(g2.degree())/g2.vcount()
    g2_lcc = g2.components().giant().vcount()
    g2_cc = g2.transitivity_undirected()
    g2_apl = g2.average_path_length()
    g2_assort = g2.assortativity_degree()

    #g_grown = grown_graph((0.0875, len(adj), len(adj[0])))
    #g_static = ig.Graph.Random_Bipartite(len(adj), len(adj[0]), p=0.253)

    with mp.Pool(processes=10) as p:
        iters = [(static_p, nodes1, nodes2)]*1000
        static_lcc = np.array(p.map(static_lcc_f, iters))
        static_apl = np.array(p.map(static_apl_f, iters))

        iters2 = [(asy1_p, nodes1, nodes2)]*1000
        asy1_lcc = np.array(p.map(asy1_lcc_f, iters2))
        asy1_apl = np.array(p.map(asy1_apl_f, iters2))

        iters3 = [(asy2_p, nodes1, nodes2)]*1000
        asy2_lcc = np.array(p.map(asy2_lcc_f, iters3))
        asy2_apl = np.array(p.map(asy2_apl_f, iters3))

    with mp.Pool(processes=10) as p:
        iters = [(static_graph, (static_p, nodes1, nodes2))]*1000
        p_static = np.array(p.map(proj_f, iters))
        iters2 = [(asy1_graph, (asy1_p, nodes1, nodes2))]*1000
        p_asy1 = np.array(p.map(proj_f, iters2))
        iters3 = [(asy2_graph, (asy2_p, nodes1, nodes2))]*1000
        p_asy2 = np.array(p.map(proj_f, iters3))
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.15, hspace=0.3)
    fig.supylabel("Frequency")
    bins = np.linspace(3, 6, 16)
    plot_hist(ax[0], "Mean Degree", static_lcc[:,0]*2, asy1_lcc[:,0]*2, asy2_lcc[:,0]*2, g_deg, bins, debug=True)
    fig.legend(loc="lower right")
    bins = np.linspace(0.5, 1, 26)
    plot_hist(ax[1], "Proportion of Nodes in Large Component", static_lcc[:,1], asy1_lcc[:,1], asy2_lcc[:,1], g_lcc/nodes, bins, np.linspace(0.5, 1, 6))
    bins = np.linspace(2, 3, 26)
    plot_hist(ax[2], "Average Path Length", static_apl[:,1], asy1_apl[:,1], asy2_apl[:,1], g_apl, bins)
    fig.suptitle(f"""Network Measures for Random Bipartite Networks\n v {networks[network]}""")
    plt.savefig(f"real_{network}.png")
    #plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(5, 15))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.925, bottom=0.15, hspace=0.4)
    fig.supylabel("Frequency")
    bins = np.linspace(5,35,31)
    plot_hist(ax[0], "Mean Degree", p_static[:,0], p_asy1[:,0], p_asy2[:,0], g1_deg, bins, projection=1)
    fig.legend(loc="lower right")
    bins = np.linspace(0,nodes1,nodes1+1)
    plot_hist(ax[1], "Large Component Size", p_static[:,2], p_asy1[:,2], p_asy2[:,2], g1_lcc, bins, projection=1)
    bins = np.linspace(0,1,26)
    plot_hist(ax[2], "Clustering Coefficient", p_static[:,4], p_asy1[:,4], p_asy2[:,4], g1_cc, bins, projection=1)
    bins = np.linspace(1,2,26)
    plot_hist(ax[3], "Average Path Length", p_static[:,6], p_asy1[:,6], p_asy2[:,6], g1_apl, bins, projection=1)
    bins = np.linspace(-1,1,26)
    plot_hist(ax[4], "Degree Assortativity", p_static[:,8], p_asy1[:,8], p_asy2[:,8], g1_assort, bins, projection=1)
    fig.suptitle(f"""Network Measures of Projections of Random Networks\n v {networks[network]} (Side One)""")
    plt.savefig(f"real_p1_{network}.png")
    #plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(5, 15))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.925, bottom=0.15, hspace=0.4)
    fig.supylabel("Frequency")
    bins = np.linspace(1,20,21)
    plot_hist(ax[0], "Mean Degree", p_static[:,1], p_asy1[:,1], p_asy2[:,1], g2_deg, bins, projection=2)
    fig.legend(loc="lower right")
    bins = np.linspace(0,nodes2,nodes2+1)
    plot_hist(ax[1], "Large Component Size", p_static[:,3], p_asy1[:,3], p_asy2[:,3], g2_lcc, bins, projection=2)
    bins = np.linspace(0,1,26)
    plot_hist(ax[2], "Clustering Coefficient", p_static[:,5], p_asy1[:,5], p_asy2[:,5], g2_cc, bins, projection=2)
    bins = np.linspace(1,2,26)
    plot_hist(ax[3], "Average Path Length", p_static[:,7], p_asy1[:,7], p_asy2[:,7], g2_apl, bins, projection=2)
    bins = np.linspace(-1,1,26)
    plot_hist(ax[4], "Degree Assortativity", p_static[:,9], p_asy1[:,9], p_asy2[:,9], g2_assort, bins, projection=2)
    fig.suptitle(f"""Network Measures of Projections of Random Networks\n v {networks[network]} (Side Two)""")
    plt.savefig(f"real_p2_{network}.png")
    #plt.show()
    