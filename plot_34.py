import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb
import seaborn as sns
sns.set()
sns.set_style("ticks")
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font',size=45)
def graph_pareto_init(p, d, filename,arg=None):
    plt.figure()
    init_idx = 64
    plt.scatter(-1/d[:init_idx, 0], d[:init_idx, 1], c='cyan', s= 10, label ='Initial designs ')
    plt.scatter(-1/d[init_idx:, 0], d[init_idx:, 1], c= 'blue', s = 10, label = 'New designs')
    plt.scatter(-1/p[:, 0], p[:, 1] , c= 'r', s= 10, label = 'Pareto fronts')
    plt.xlim(-240, -80)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim(0, 1.8)
    plt.legend(loc='lower left', fontsize=15)
    plt.xlabel('-Mass', fontsize=18)
    plt.ylabel('Elapsed Time', fontsize=18)
    plt.savefig(filename+".pdf",bbox_inches='tight',pad_inches=0)

def pareto_max(costs ):
    pareto_set = []
    non_pareto_set = []
    pareto_arg = []
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True  # And keep self

    for i in range(costs.shape[0]):
        if is_efficient[i]:
            pareto_set.append(costs[i])
            pareto_arg.append(i)
        else:
            non_pareto_set.append(costs[i])
    return np.array(pareto_set), np.array(non_pareto_set), is_efficient


data= np.load('ACBO-uniform(GripperProblemBO)k=2.0d=20maxf=20000fin.npy')
data1= np.load('BILEVEL-DIRECT(GripperProblemBO)k=2.0d=20fmax=1000fin.npy')
pareto, npareto, ar = pareto_max(data)
graph_pareto_init(pareto, data, 'ACBO')
pareto1, _, ar1 = pareto_max(data1)
graph_pareto_init(pareto1, data1,'Bilvel')