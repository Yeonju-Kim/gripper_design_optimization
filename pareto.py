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

def coverage_metric(costs, comp):
    is_efficient = []
    for i, c in enumerate(costs):
        for j, d in enumerate(comp):
            if np.all(c >= d):
                is_efficient.append(True)
                break
        if len(is_efficient) is not i+1:
            is_efficient.append(False)

    assert len(costs)==len(is_efficient)
    print(is_efficient)
    return sum(is_efficient)/float(len(costs))
def graph_pareto_init(p, d, arg=None):
    plt.figure()
    # ss = np.argsort(p[:, 0])
    # if arg is not None:
    #     for i in range(p.shape[0]):
    #         idx = np.argwhere(ss == i)[0][0]
    #         # pdb.set_trace()
    #         plt.text(-1/p[i, 0] + 0.5, p[i, 1]+0.01, str(idx+1), size=10)
    init_idx = 64
    plt.scatter(-1/d[:init_idx, 0], d[:init_idx, 1], c='cyan', s= 10, label ='Initial designs ')
    plt.scatter(-1/d[init_idx:, 0], d[init_idx:, 1], c= 'blue', s = 10, label = 'New designs')
    plt.scatter(-1/p[:, 0], p[:, 1] , c= 'r', s= 10, label = 'Pareto fronts')
    plt.xlim(-240, -80)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim(0, 3.5)
    plt.legend(loc='lower left', fontsize=15)
    plt.xlabel('-Mass', fontsize=18)
    plt.ylabel('Elapsed Time', fontsize=18)
    plt.savefig("Bilevel.pdf",bbox_inches='tight',pad_inches=0)

    # plt.show()
def graph_pareto_init_100overmass(p, d):
    plt.figure()
    for i in range(d.shape[0]):
        plt.text(100*d[i, 0], d[i, 1], str(i), size=8)
    init_idx = 64
    plt.scatter(100*d[:init_idx, 0], d[:init_idx, 1], c='cyan', s= 15, label ='Initial designs ')
    plt.scatter(100*d[init_idx:, 0], d[init_idx:, 1], c= 'blue', s = 15, label = 'New designs')
    plt.scatter(100*p[:, 0], p[:, 1] , c= 'r', s= 5, label = 'Pareto fronts')
    plt.xlim(0.2, 1.2)
    plt.ylim(0, 1.8)
    plt.legend()
    plt.xlabel('100/Mass Metric')
    plt.ylabel('Elapsed Time Metric')

def graph_pareto_comp(p, d):
    # plt.figure()
    # for i in range(d.shape[0]):
    #     plt.text(-100/d[i, 0], d[i, 1], str(i), size=8)
    # init_idx = 64
    # plt.scatter(-100/d[:init_idx, 0], d[:init_idx, 1], c='cyan', s= 15, label ='Initial designs ')
    plt.scatter(-100/p[:, 0], p[:, 1] , c= 'r', s= 5, label = 'pareto front1')

    plt.scatter(-100/d[:, 0], d[:, 1], c= 'blue', s = 15, label = 'pareto front2')

    plt.legend()
    plt.xlabel('-Mass Metric')
    plt.ylabel('Elapsed Time Metric')
    plt.show()


def graph_pareto(p, d):
    pdb.set_trace()
    plt.figure()
    for i in range(d.shape[0]):
        plt.text(-100/d[i, 0], d[i, 1], str(i), size=8)
    init_idx = 64
    kappa_change_idx = 84
    plt.scatter(-100/d[:init_idx, 0], d[:init_idx, 1], c='cyan', s= 15, label ='Initial designs ')
    plt.scatter(-100/d[init_idx:kappa_change_idx, 0], d[init_idx:kappa_change_idx, 1], c= 'blue', s = 15, label = 'kappa = 10.')
    plt.scatter(-100/d[kappa_change_idx:, 0], d[kappa_change_idx:, 1], c= 'green', s = 15, label = 'kappa = 2.')
    plt.scatter(-100/p[:, 0], p[:, 1] , c= 'r', s= 5, label = 'Pareto fronts')

    plt.legend()
    plt.xlabel('-Mass Metric')
    plt.ylabel('Elapsed Time Metric')
    plt.show()


# import numpy as np
import numpy.matlib as mtlib
# import pdb

def hypeIndicatorSampledMax(points,  bounds=[0.,0.], k = 4, nrOfSamples = 100000):
    (nrP, dim) = points.shape
    F = np.zeros(nrP)
    alpha = np.zeros(nrP + 1)

    for i in range(1, k + 1):
        for j in range(1, i):
            if alpha[i] == 0.:
                alpha[i] = 1.
            alpha[i] *= (k - j) / (nrP - j)
        if len(range(1, i)) == 0:
            alpha[i] = 1.
        alpha[i] = alpha[i] / i
    # pdb.set_trace()
    if len(bounds) == 1:
        bounds = mtlib.repmat(bounds, 1, dim)

    BoxU = np.max(points, axis=0)
    S = np.array(mtlib.rand(nrOfSamples, dim) * mtlib.diag(BoxU - bounds) \
        + mtlib.ones((nrOfSamples, dim)) * mtlib.diag(bounds))
    dominated = np.zeros((nrOfSamples, 1))

    for j in range(nrP):
        B = np.array(mtlib.repmat(points[j, :], nrOfSamples, 1) - S)
        ind = np.where(np.sum(B >= 0, axis=1) == dim)[0]
        dominated[ind] = dominated[ind] + 1

    for j in range(nrP):
        B = np.array(mtlib.repmat(points[j, :], nrOfSamples, 1) - S)
        ind = np.where(np.sum(B >= 0, axis=1) == dim)[0]
        x = dominated[ind]
        for nn in range(len(x)):
            F[j] += alpha[int(x[nn, 0])]
    F = F * np.prod(BoxU - bounds) / nrOfSamples

    return F[0]


def hypervolume2d(data):
    data[:,0 ] *= 100.
    pareto, _, _ = pareto_max(data)

    x = np.sort(pareto[:,0])
    y = np.flip(np.sort(pareto[:,1]))
    print(x,y)

    hv = 0
    for i in range(len(pareto)):
        if i ==0 :
            hv += x[i]*y[i]
        else:
            hv += (x[i]-x[i-1])*y[i]
    return hv



if __name__ == '__main__':
    # data = np.load('scores.npy')
    #TODO: FInal gripper results
    # data  = np.load('ACBO-uniform(GripperProblemBO)k=2.0d=20maxf=20000fin.npy')
    # data1= np.load('../data_gripper/BILEVEL-DIRECT(GripperProblemBO)k=2.0d=20fmax=1000fin.npy')
    data= np.load('BILEVEL-DIRECT(GripperProblemBO)k=2.0d=20fmax=1000.npy')
    # data1 = np.load('newBILEVEL-uniform(GripperProblemBO)k=2.0d=20fmax=1000.npy')
    # data1= np.load('../data_gripper/BILEVEL-DIRECT(GripperProblemBO)k=2.0d=20fmax=1000fin.npy')
    # data1=np.load('../data_gripper/BILEVEL-DIRECT(GripperProblemBO)k=2.0d=50fmax=100002.npy')
    # data2=np.load('../data_gripper/BILEVEL-DIRECT(GripperProblemBO)k=2.0d=50fmax=100003.npy')
    # data2 = np.load('BILEVEL-DIRECT(GripperProblemBO)k=2.0d=10fmax=1000fin.npy')
    # data3 = np.load('BILEVEL-uniform(GripperProblemBO)k=2.0d=10fmax=1000fin.npy')
    # data4 = np.load('../data_gripper/BILEVEL-uniform(GripperProblemBO)k=10.0d=100fmax=1000.npy')
    pareto, npareto, ar = pareto_max(data)
    graph_pareto_init(pareto, data, np.where(ar)[0])
    # pareto1, _, ar1 = pareto_max(data1)
    # graph_pareto_init(pareto1, data1,np.where(ar1)[0])
    # pareto2, _, _ = pareto_max(data2)
    # graph_pareto_init(pareto2, data2)
    # pareto3,_, _ = pareto_max(data3)
    # graph_pareto_init(pareto3, data3)
    # pareto4,_, _ = pareto_max(data4)
    # graph_pareto_init(pareto4, data4)
    plt.show()


    pdb.set_trace()
    # data = np.load('../gripper_design_optimization/score_bilevel.npy')
    # data2 = np.load('../gripper_design_optimization/score_ACBO.npy')
    #
    # pareto, npareto, is_eff = pareto_max(data)
    # pareto2, _, is_eff2 = pareto_max(data2)
    # comp_pareto, _, result = pareto_max(np.vstack((pareto, pareto2)))
    # # print(result[:len(pareto)].sum()/float(len(pareto)),
    # #       result[len(pareto):].sum()/float(len(pareto2)))
    # print(coverage_metric(paretoll[0], paretoll[1]), coverage_metric(paretoll[1], paretoll[0]))
    # print(coverage_metric(paretoll[0], paretoll[2]), coverage_metric(paretoll[2], paretoll[0]))
    # print(coverage_metric(paretoll[1], paretoll[2]), coverage_metric(paretoll[2], paretoll[1]))
    # graph_pareto_comp(pareto, pareto2)

    # pdb.set_trace()
    from hv import HyperVolume
    referencePoint = [2. ,2.]
    data[:,0 ] *= 100.
    pareto, npareto, ar = pareto_max(data)
    data1[:,0 ] *= 100.
    pareto1, npareto, ar = pareto_max(data1)

    print(pareto.tolist())
    hv = HyperVolume(referencePoint)

    # volume = hv.compute([1,1])
    # pdb.set_trace()
    # print(hv.compute(pareto.tolist()), hv.compute(pareto1.tolist()))
    print (hypervolume2d(pareto), hypervolume2d(pareto1), hypervolume2d(pareto2),)
    #, hv.compute(paretoll[2].tolist()))
    # # # print(ar)
    # # print(len(data))
    # # graph_pareto_init(pareto, data)
    #