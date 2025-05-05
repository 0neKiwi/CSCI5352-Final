### Networks
There are two networks in this repository. In the interest of time and page length, only one was explored in the paper.  
The Dupont plant-pollinator web's adjacency matrix is stored in `dupont.txt`, and the Brunson CEO club membership's edgelist is stored in `brunson.txt`.  
### Scripts
Run `python plot_bipartite.py` with each type of network measure (`lcc` for largest connected component size and `apl` for average path length).  
Run `python plot_projection.py` with each type of network measure (`lcc` for largest connected component size, `cc` for clustering coefficient, `apl` for average path length, and `assort` for degree assortativity).  
Run `python plot_real.py` with either `dupont` or `brunson`. You can pipe the output into a text file. Sample output is indicated in `real_dupont.txt` and `real_brunson.txt`.
