import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
sc = np.load('normalized_score.npy')
design = np.load('design.npy')
from TRINA_BO import RobotArmProblemBO
import pickle
def graph_3d(data, original =None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    ax.scatter3D(data[:, 0], data[:, 1], data[:,2], c= 'r', s = 20 )
    if original is not None:
        ax.scatter3D(original[:, 0], original[:,1], original[:, 2] , c= 'black' , s= 3)
    ax.set_xlabel('pan')
    ax.set_ylabel('tilt')
    ax.set_zlabel('sum of normalized scores')
    plt.show()


# design[:,2] += 45
# design[:,0] += 2*0.174625
# sc_sum = sc.sum(1).reshape(-1, 1)
# data = np.hstack((design[:, 1:], sc_sum))
# arg = np.argwhere(sc.sum(1)>3.6)
# pdb.set_trace()
# graph_3d(data[arg].reshape(-1,3) , data)
# graph_3d()


if __name__ == '__main__':
    # scores = np.load('sc_0.187267.npy')
    # policies = np.load('po_0.187267.npy')
    # score_max = []
    # policies_max = []
    # for i in range (len(scores)):
    #     score_per_design = scores[i]
    #     arg_max = scores[i].argmax(0)
    #     policies_per_design = []
    #     for e in range(5):
    #         policies_per_design.append(policies[i][arg_max[e]][e])
    #
    #     score_max.append(score_per_design.max(0))
    #     policies_max.append(policies_per_design)
    #
    # pdb.set_trace()

    policy_space = [('c0', None), ('c1', None), ('c2', None), ('c3', None), ('c4',None), ('c5', None)]
    problemBO = RobotArmProblemBO(policy_space)
    num_objects = 5
    designs = np.load('design_bigger3.6_PF.npy')
    num_configs= 20
    # max_scores = np.load('score_max_robotarm2.npy')
    # max_policies = np.load('policies_max_robotarm2.npy')
    designs = np.array([[0.18, 72, 22]]) # TODO: find the best configuration
    # problemBO.plot_solution(designs[0].tolist() + np.array(policies_max[0]).T.tolist())
    # pdb.set_trace()
    scores = [[None for j in range(num_configs)] for i in range(len(designs))]
    policies = [[[None for i in range(num_objects)] for l in range(num_configs)] for j in range(len(designs))]
    #index: policies[design_id][]

    # configs
    for d_id in range(len(designs)):
        for obj_id in range(num_objects):
            config_candidates = problemBO.get_config_candidates(design = designs[d_id],
                                                                object_id = obj_id,
                                                                max_samples = num_configs)

            for c_id in range(num_configs):
                policies[d_id][c_id][obj_id] = config_candidates[c_id]

    # pdb.set_trace()
    # Evaluate
    for d_id in range(len(designs)):
        for c_id in range(num_configs):
            design_policy = designs[d_id].tolist() + np.array(policies[d_id][c_id]).T.tolist()
            # pdb.set_trace()

            scoresOI, scoresOD = problemBO.eval([design_policy], mode='MAX_POLICY', visualize=False)
            scores[d_id][c_id] = np.array(scoresOD).reshape(-1)
    pdb.set_trace()

    pickle.dump(scores, open('scores_robotarm_10', 'wb'))
    pdb.set_trace()



