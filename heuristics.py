import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pareto
def delta_metric(data):
    data_sorted = np.sort(data, axis=0)
    distance= data_sorted[1:] - data_sorted[:-1]
    assert np.all(distance >= 0)
    abs = np.abs(distance - np.mean(distance, axis=0))
    # print (abs)
    delta = np.mean(abs, axis = 0) / np.mean(distance, axis= 0)
    metric = np.max(delta)
    return delta, metric

def extreme_points(data):
    n_dims = np.shape(data)[1]
    arg_extreme = []
    for dim in range(n_dims):
        arg_sorted =np.argsort(data[:, dim])
        if arg_sorted[0] not in arg_extreme:
            arg_extreme.append(arg_sorted[0])
        if arg_sorted[-1] not in arg_extreme:
            arg_extreme.append(arg_sorted[-1])
    return np.asarray(arg_extreme)

def smallest_points(data):
    n_dims = np.shape(data)[1]
    arg_extreme = []
    for dim in range(n_dims):
        argmin = np.argmin(data[:, dim])# OR argmax
        if argmin not in arg_extreme:
            arg_extreme.append(argmin)
    return np.asarray(arg_extreme)

def furthest_points(data):
    max_dist = 0
    arg_extreme = []

    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            dist = np.linalg.norm(data[i]- data[j])
            if dist > max_dist :
                arg_extreme = [i, j]
                max_dist = dist
    return np.asarray(arg_extreme)

def farthest_first(data, k):
    selected = furthest_points(data)
    # selected = smallest_points(data)
    bool_selected = np.zeros(data.shape[0]).astype(bool)
    bool_selected[selected] = True
    subset = data[selected]
    arg_subset = selected.tolist()
    # pdb.set_trace()

    #TODO: farthest first
    while len(subset) < k:
        min_dist = None
        for pt in subset:
            dist = np.linalg.norm(pt-data, axis = 1)
            if min_dist is not None:
                min_dist = np.min(np.vstack((dist, min_dist)), axis = 0)
            else:
                min_dist = dist
        sub_idx = np.argmax(min_dist[(1-bool_selected).astype(bool)])
        arg_selected = np.arange(data.shape[0])[(1-bool_selected).astype(bool)][sub_idx]
        bool_selected[arg_selected] = True
        subset = np.vstack((subset, data[arg_selected]))
        arg_subset.append(arg_selected)

    if data.shape[1] == 2:
        graph(subset, data)
    elif data.shape[1] == 3:
        graph_3d(subset, data)

    _, m = delta_metric(subset)
    print(subset)
    print(arg_subset)
    print('original and changed ', delta_metric(data)[1], m)
    return arg_subset

def neiborhood_heuristic(data, k):
    arg_min = furthest_points(data)
    r = []
    for i in range(data.shape[0]):
        for j in range(i+1, data.shape[0]):
            r.append(np.linalg.norm(data[i]- data[j]))
    r.sort()

    L = 0
    R = len(r)-1
    result = []

    while L <= R:
        m = int(np.floor((L+R)/2))
        result = arg_min
        data_uncovered = np.ones(data.shape[0]).astype(bool)
        for center in data[result]:
            distance_from_center = np.linalg.norm(data - center, axis=1)
            data_uncovered = np.logical_and(data_uncovered, distance_from_center > r[m])
        while np.any(data_uncovered):
            dist_ = np.zeros(data.shape[0])
            for i in range(len(result)):
                dist_ += np.linalg.norm(data - data[result[i]], axis= 1)
            arg_ = np.argwhere(data_uncovered)
            next_idx = arg_[np.argmax(dist_[data_uncovered])]

            result = np.hstack((result, next_idx[0]))
            center = data[next_idx[0]]
            distance_from_center = np.linalg.norm(data - center, axis=1)
            data_uncovered = np.logical_and(data_uncovered, distance_from_center > r[m])
            if not np.any(data_uncovered):
                break
            # if data.shape[1] == 2:
            #     fig, ax = plt.subplots()
            #     for i in result:
            #         circle = plt.Circle((data[i,0], data[i,1]), r[m], color='cyan', fill=False)
            #         ax.add_artist(circle)
            #     plt.scatter(data[:, 0], data[:, 1], c= 'r', s =5)
            #
            #     covered = (1 - data_uncovered).astype(bool)
            #     plt.scatter(data[covered, 0], data[covered, 1], c='cyan', s=7)
            #     plt.scatter(data[result, 0], data[result, 1],c= 'b')
            #     plt.show()
            # elif data.shape[1] == 3:
            #     graph_3d(data[result], data)

        if len(result) > k :
            L = m+1
        elif len(result) < k:
            R = m-1
        else:
            _, metric = delta_metric(data[result])
            print('successful', metric)
            break
    if data.shape[1] == 2:
        fig, ax = plt.subplots()
        for i in result:
            circle = plt.Circle((data[i, 0], data[i, 1]), r[m], color='cyan', fill=False)
            ax.add_artist(circle)
        covered = (1 - data_uncovered).astype(bool)
        plt.scatter(data[:, 0], data[:, 1], c='r', s=5)
        plt.scatter(data[covered, 0], data[covered, 1], c='cyan', s=7)
        plt.scatter(data[result, 0], data[result, 1], c='b')
        plt.show()
    elif data.shape[1] == 3:
        graph_3d(data[result], data)
    return result

def is_pareto_simple_max(data):
    costs = data
    pareto_set = []
    non_pareto_set = []

    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
            is_efficient[i] = True  # And keep self

    for i in range(costs.shape[0]):
        if is_efficient[i]:
            pareto_set.append(costs[i])
        else:
            non_pareto_set.append(costs[i])
    pareto_set = np.asarray(pareto_set)
    non_pareto_set = np.asarray(non_pareto_set)

    return pareto_set, non_pareto_set

def graph(data, original):
    plt.figure(figsize=(5,5))
    plt.scatter(data[:, 0], data[:,1], c='b')
    plt.scatter(original[:,0], original[:,1 ], c= 'r', s=5)
    plt.show()

def graph_3d(data, original):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    ax.scatter3D(data[:, 0], data[:, 1], data[:,2], c= 'r', s = 20 )
    ax.scatter3D(original[:, 0], original[:,1], original[:, 2] , c= 'gray' , s= 2)
    # plt.show()

def sphere_coordinates(n):
    theta = np.random.uniform(size = (n, 1))* np.pi/2
    phi = np.random.uniform(size = (n, 1 ))*np.pi/2
    return np.hstack((np.sin (phi )*np.cos(theta), np.sin(phi)* np.sin(theta) , np.cos(phi)))

def sphere_coordinates_(n):
    theta = np.random.uniform(size = (n, 1))* np.pi/2
    phi = np.sin(np.random.normal( size = (n, 1 )))*np.pi/2
    return np.hstack((np.sin (phi)*np.cos(theta), np.sin(phi)* np.sin(theta) , np.cos(phi)) )

if __name__ == '__main__':
    # data = np.load('p2_2000_1000.npy')
    # angle = np.random.normal(size = (200, 1))*np.pi/2
    # original = np.hstack((np.cos(angle), np.sin(angle)))

    # original = sphere_coordinates_(200)

    # original =np.random.normal(size= (10000,2))
    # data, _ = is_pareto_simple_max(original)
    # print(data.shape)
    data = np.load('with_obj_trina_local_solve_init.dat')



    pareto, npareto, ar = pareto.pareto_max(data)
    print(np.where(ar))



    # for i in range(len(data)):
    i = 5
    arg_subset = farthest_first(pareto,i )
    arg = np.where(ar)[0][arg_subset]
    print(arg)
    print(data[arg])
    pdb.set_trace()

    result = neiborhood_heuristic(pareto, i)
    pdb.set_trace()
    plt.show()
