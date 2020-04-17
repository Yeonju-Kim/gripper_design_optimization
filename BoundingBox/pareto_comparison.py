import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


class Observations:
    def __init__(self, num_objectives, exps_th=10000, noise_th=2, text_in_graph = False
                 ,show_step=False, minimizing= False):
        self.num_objectives = num_objectives
        self.lower_bounds = np.empty((0, num_objectives))
        self.upper_bounds = np.empty((0, num_objectives))
        self.num_exps = np.empty((0))
        self.exps_threshold = exps_th
        self.noise_threshold = noise_th
        self.show_steps_ = show_step
        self.is_min = minimizing
        self.exact = None
        self.cur_pareto_idx = None
        self.not_overlapped = None
        self.is_pareto = None
        self.text_in_graph = text_in_graph

    def add_observation(self, lower_bounds, upper_bounds, num_experiments):
        '''
        compare the overlapping area among non-dominated elements
        '''
        self.lower_bounds = np.vstack((self.lower_bounds, lower_bounds))
        self.upper_bounds = np.vstack((self.upper_bounds, upper_bounds))
        self.num_exps = np.hstack((self.num_exps, num_experiments))
        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise RuntimeError("The shape of lower bounds array is not same with upper bounds.")

        self.is_pareto_BB()
        self.is_overlap_in_dominant()
        is_overlapped_in_dominant_set = (1 - self.not_overlapped).astype(bool)
        if self.show_steps_:
            self.draw_plot_pareto()

        if True in is_overlapped_in_dominant_set \
                and np.min(self.num_exps[is_overlapped_in_dominant_set]) < self.exps_threshold:
            subset_idx = np.argmin(self.num_exps[is_overlapped_in_dominant_set])
            parent_idx = np.arange(self.num_exps.shape[0])[is_overlapped_in_dominant_set][subset_idx]
            return parent_idx
        else:
            return -1

    def update_observation(self, lower_bound, upper_bound, more_exps, parent_idx):
        # TODO : Define how to update the bounds!
        self.lower_bounds[parent_idx] = lower_bound
        self.upper_bounds[parent_idx] = upper_bound
        self.num_exps[parent_idx] = more_exps

        self.is_pareto_BB()
        self.is_overlap_in_dominant()
        if self.show_steps_:
            self.draw_plot_pareto()
        is_overlapped_in_dominant_set = (1 - self.not_overlapped).astype(bool)

        if True in np.logical_and(is_overlapped_in_dominant_set, self.is_pareto) \
                and np.min(self.num_exps[is_overlapped_in_dominant_set]) < self.exps_threshold:
            subset_idx = np.argmin(self.num_exps[is_overlapped_in_dominant_set])
            parent_idx = np.arange(self.num_exps.shape[0])[is_overlapped_in_dominant_set][subset_idx]
            return parent_idx
        else:
            return -1

    def reduction_CI_test(self, idx):
        '''
        reduction of Confidence Interval of Bounding Boxes by more experiments
        '''
        parent_idx = idx
        err_magnitude = self.upper_bounds[parent_idx] - self.lower_bounds[parent_idx]
        self.lower_bounds[parent_idx] += np.random.uniform(-1, 1) * 3
        self.upper_bounds[parent_idx] = err_magnitude * 0.8 + self.lower_bounds[parent_idx]
        self.num_exps[parent_idx] *= 2

    def is_overlap(self):
        num_candidates = self.lower_bounds.shape[0]
        num_functions = self.lower_bounds.shape[1]
        not_overlapped = np.ones(num_candidates, dtype=bool)
        for i in range(num_candidates):
            # if not_overlapped[i]:
            l_ = self.lower_bounds[i]
            u_ = self.upper_bounds[i]
            for idx in range(num_candidates):
                if idx != i:
                    # if not_overlapped[idx]:
                    for func_idx in range(num_functions):
                        l_idx = self.lower_bounds[idx]
                        u_idx = self.upper_bounds[idx]
                        if l_[func_idx] < l_idx[func_idx] < u_[func_idx] or \
                                l_[func_idx] < u_idx[func_idx] < u_[func_idx] or \
                                (l_idx[func_idx] <= l_[func_idx] and u_idx[func_idx] >= u_[func_idx]):
                            not_overlapped[idx] = False
                            break
        self.not_overlapped = not_overlapped
        if not_overlapped[not_overlapped].shape[0] < num_candidates:
            return False
        else:
            return True

    def is_overlap_in_dominant(self):
        num_candidates = self.lower_bounds[self.is_pareto].shape[0]
        num_functions = self.lower_bounds[self.is_pareto].shape[1]
        not_overlapped = np.ones(num_candidates, dtype=bool)
        lower_bounds_ = self.lower_bounds[self.is_pareto]
        upper_bounds_ = self.upper_bounds[self.is_pareto]
        for i in range(num_candidates):
            # if not_overlapped[i]:
            l_ = lower_bounds_[i]
            u_ = upper_bounds_[i]
            for idx in range(num_candidates):
                if idx != i:
                    # if not_overlapped[idx]:
                    for func_idx in range(num_functions):
                        l_idx = lower_bounds_[idx]
                        u_idx = upper_bounds_[idx]
                        if l_[func_idx] < l_idx[func_idx] < u_[func_idx] or \
                                l_[func_idx] < u_idx[func_idx] < u_[func_idx] or \
                                (l_idx[func_idx] <= l_[func_idx] and u_idx[func_idx] >= u_[func_idx]):
                            not_overlapped[idx] = False
                            break
                # not_overlapped[i] = True
        global_not_overlapped = np.ones(self.lower_bounds.shape[0], dtype=bool)
        global_not_overlapped[self.is_pareto] = not_overlapped
        self.not_overlapped = global_not_overlapped

        if not_overlapped[not_overlapped].shape[0] < num_candidates:
            return False
        else:
            return True

    def is_pareto_BB(self):
        num_candidates = self.lower_bounds.shape[0]
        is_efficient = np.ones(num_candidates, dtype=bool)

        if self.is_min:
            for i in range(num_candidates):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(self.lower_bounds[is_efficient] < self.upper_bounds[i],
                                                        axis=1)  # Keep any point with a lower cost
                    is_efficient[i] = True  # And keep self
        else: #maximizing
            for i in range(num_candidates):
                if is_efficient[i]:
                    is_efficient[is_efficient] = np.any(self.upper_bounds[is_efficient] > self.lower_bounds[i],
                                                        axis=1)  # Keep any point with a lower cost
                    is_efficient[i] = True  # And keep self

        self.is_pareto = is_efficient
        return

    def draw_plot(self):
        fig, ax = plt.subplots()
        errorboxes_not_overlapped = []
        for u, l in zip(self.upper_bounds[self.not_overlapped], self.lower_bounds[self.not_overlapped]):
            rect = Rectangle((l[0], l[1]), u[0] - l[0], u[1] - l[1])
            errorboxes_not_overlapped.append(rect)

        errorboxes_overlapped = []
        overlap = (1 - self.not_overlapped).astype(bool)
        for u, l in zip(self.upper_bounds[overlap], self.lower_bounds[overlap]):
            rect = Rectangle((l[0], l[1]), u[0] - l[0], u[1] - l[1])
            errorboxes_overlapped.append(rect)

        # Create patch collection with specified colour/alpha
        pc_not_overlap = PatchCollection(errorboxes_not_overlapped, facecolor='b', alpha=0.3,
                                         edgecolor='None')
        pc_overlap = PatchCollection(errorboxes_overlapped, facecolor='g', alpha=0.3,
                                     edgecolor='None')
        plt.xlim(0, 100)
        plt.ylim(0, 100)

        data_x = 0.5 * (self.upper_bounds[:, 0] + self.lower_bounds[:, 0])
        data_y = 0.5 * (self.upper_bounds[:, 1] + self.lower_bounds[:, 1])

        plt.scatter(data_x, data_y, lw=1, s=0.8)
        ax.add_collection(pc_not_overlap)
        ax.add_collection(pc_overlap)
        plt.show()

    def draw_plot_pareto(self, lim=80, add_exact_graph = False, sampled_point = None):
        fig, ax = plt.subplots()

        bb_pareto_not_overlapped = []
        pareto_not_overlapped_mask = np.logical_and(self.is_pareto, self.not_overlapped)
        for u, l in zip(self.upper_bounds[pareto_not_overlapped_mask], self.lower_bounds[pareto_not_overlapped_mask]):
            rect = Rectangle((l[0], l[1]), u[0] - l[0], u[1] - l[1])
            bb_pareto_not_overlapped.append(rect)

        bb_pareto_overlapped = []
        pareto_overlapped_mask = np.logical_and(self.is_pareto, (1 - self.not_overlapped).astype(bool))
        for u, l in zip(self.upper_bounds[pareto_overlapped_mask], self.lower_bounds[pareto_overlapped_mask]):
            rect = Rectangle((l[0], l[1]), u[0] - l[0], u[1] - l[1])
            bb_pareto_overlapped.append(rect)

        bb_non_pareto = []
        non_pareto = (1 - self.is_pareto).astype(bool)
        for u, l in zip(self.upper_bounds[non_pareto], self.lower_bounds[non_pareto]):
            rect = Rectangle((l[0], l[1]), u[0] - l[0], u[1] - l[1])
            bb_non_pareto.append(rect)

        # Create patch collection with specified colour/alpha
        pc_pareto = PatchCollection(bb_pareto_not_overlapped, facecolor='None', alpha=1,
                                    edgecolor='r')
        pc_pareto_overlapped = PatchCollection(bb_pareto_overlapped, facecolor='None', alpha=0.7,
                                               edgecolor='g')
        # pc_pareto_overlapped_edge = PatchCollection(bb_pareto_overlapped, facecolor='None', alpha=1,
        #                                             edgecolor='r')
        pc_non_pareto = PatchCollection(bb_non_pareto, facecolor='None', alpha=0.3,
                                        edgecolor='b', label='dominated')
        if not np.iterable(lim):
            lim = [[0, lim], [0, lim]]
        plt.xlim(lim[0][0], lim[0][1])
        plt.ylim(lim[1][0], lim[1][1])
        # plt.axis('equal')
        data_x = 0.5 * (self.upper_bounds[:, 0] + self.lower_bounds[:, 0])
        data_y = 0.5 * (self.upper_bounds[:, 1] + self.lower_bounds[:, 1])
        # Add collection to axes
        if self.text_in_graph:
            for i in range(self.num_exps.shape[0]):
                ax.text(data_x[i], data_y[i], str(i) + ',' +str(self.num_exps[i]),size=6)

        if add_exact_graph:
            plt.scatter(self.exact[0], self.exact[1], c='g', s=0.5)

        ax.add_collection(pc_pareto)
        ax.add_collection(pc_non_pareto)
        ax.add_collection(pc_pareto_overlapped)
        plt.scatter(data_x[self.is_pareto], data_y[self.is_pareto], lw=1, s=1, c='r')
        plt.scatter(data_x[(1 - self.is_pareto).astype(bool)], data_y[(1 - self.is_pareto).astype(bool)], lw=1, s=0.8,
                    c='b')

        if sampled_point is not None:
            mean_point = sampled_point[0]
            plt.scatter(mean_point[0], mean_point[1], c ='r', s = 1)
            plt.scatter(sampled_point[1][0], sampled_point[1][1], c= 'b', s= 0.6)
        plt.show()
