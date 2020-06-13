from Simulation.gl_vis import *
from BoundingBox.pareto_comparison import Observations
import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic)
import pdb
from pyDOE import *
from scipy.stats import uniform
import matplotlib.pyplot as plt
start_time = time.time()

class optimize_design:
    def __init__(self, init_with_lhs, num_init_samples,  obj_space_lim, world_file_name, object_file_name, iter_per_obj, num_objectives, exps_th, gamma, num_designs, bounds, show_step):
        self.num_init_samples = num_init_samples
        self.init_with_lhs = init_with_lhs
        self.obj_space_lim = obj_space_lim

        self.world_file_name = world_file_name
        self.object_list = glob.glob(object_file_name)
        self.num_objects = len(self.object_list)
        self.iter_per_object = iter_per_obj
        self.num_objectives = num_objectives
        self.num_designs = num_designs

        self.bounds = bounds
        self.dim_design_space = bounds.shape[0]

        self.train_features = np.empty((0, bounds.shape[0]))
        self.train_labels = np.empty((0, bounds.shape[0]))

        self.process_bb = Observations(num_objectives, exps_th=exps_th, show_step=show_step, text_in_graph=True)
        self.gamma = gamma

        self.design_parameter = []
        self.num_trials = []
        self.num_success = []
        self.grasp_quality = []
        self.mass = []
        self._initialize_gp(num_objectives)

    def _initialize_gp(self, num_objectives):
        self.gp = []
        for i in range(num_objectives):
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=25, alpha = 0.0001)
            self.gp.append(gp)

    def do_experiment(self, design_idx):
        d = self.design_parameter[design_idx]
        if len(d) is not 2:
            raise RuntimeError("design parameter's dimension is wrong")
        length = [1.]*9
        width = [1.]*9
        radius = [0.07, 0.07, 0.07]

        curvature = d[0]
        theta = d[1]*30.
        link_angle = [60, 90 - theta, 30 + theta]
        link_tilted_angle = [0, -(30 - theta), 30 - theta]

        num_success, grasp_quality, mass \
            = grasp_test(self.world_file_name, length, width, link_angle, radius, link_tilted_angle, curvature=curvature,
                         object_files=self.object_list, max_iter=self.iter_per_object)
        print("num_success : ", num_success)
        print("grasp_quality: ", grasp_quality)
        print("mass : ", mass)
        return np.asarray(num_success), grasp_quality, mass

    def is_pareto(self, pareto_set, point):
        costs = np.vstack((point, pareto_set))
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)
                is_efficient[i] = True
        return is_efficient[0]

    def is_pareto_simple_max(self):
        costs = self.train_labels
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

    def post_processing_per_design(self, idx):
        grasp_quality = self.grasp_quality[idx]
        l_bounds_result = np.zeros((1, 2))
        u_bounds_result = np.zeros((1, 2))
        num_obj_success = 0
        #TODO: for self.grasp_quality[idx][obj_idx][0] : average value- gamma*std , average + gamma*std
        # for self.grasp_quality[idx][obj_idx][1]: max , max+ noise

        for obj_idx in range(self.num_objects):
            num_exp = grasp_quality[obj_idx].shape[1]
            if num_exp > 0:
                print(grasp_quality[obj_idx])
                noise = np.std(grasp_quality[obj_idx][1].astype(float)) * self.gamma/np.sqrt(num_exp)
                if num_exp == 1: noise =1.
                max_radius = np.max(grasp_quality[obj_idx][1].astype(float))
                l_bounds_result += np.array([0, max_radius])
                u_bounds_result += np.array([0, max_radius+noise*2])
                num_obj_success += 1
        if num_obj_success == 0:
            l_bounds_result[0, 0] = -self.mass[idx]
            return l_bounds_result, l_bounds_result + np.array([0, 1.])
        else:
            l_bounds_result = l_bounds_result / num_obj_success
            l_bounds_result[0,0] = -self.mass[idx]
            u_bounds_result = u_bounds_result / num_obj_success
            u_bounds_result[0,0] = -self.mass[idx]
            print("bound " ,l_bounds_result, u_bounds_result )
            return l_bounds_result, u_bounds_result

    def acquisition_MC_random(self, pareto_set):
        # TODO: Randomly sampled candidates & return argmax value of acquistion function
        exploration = 0.9
        n_samples = 1000
        mc_samples = 100
        tolerance = 0.95
        acquisition_func = np.zeros(n_samples)
        sampled_points = []
        argmax_idx = 0
        x_tries = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(n_samples, self.dim_design_space))

        f = []
        for i in range(self.num_objectives):
            f += [self.gp[i].sample_y(x_tries, mc_samples)]
        f = np.asarray(f)
        print(f.shape)

        prob = np.random.uniform()
        if prob > exploration:
            print("exploration")
            # TODO: Add heuristic exploration term using std values
            _, std = self.gp[1].predict(x_tries, return_std=True)
            std_max = np.max(std)
            extracted = np.extract(std > std_max * tolerance, std)
            subset_idx = np.random.randint(0, extracted.shape[0])
            argmax_idx = np.extract(std > std_max * tolerance, np.arange(std.shape[0]))[subset_idx]
        else:
            n_pareto_vals = []
            for idx in range(n_samples):
                # y_samples = np.vstack((f_1[idx], f_2[idx]))
                y_samples = np.asarray(f[:, idx, :])
                n_pareto = 0
                for y_idx in range(mc_samples):
                    if self.is_pareto(pareto_set, y_samples[:, y_idx]):
                        n_pareto += 1

                acquisition_func[idx] = n_pareto
                n_pareto_vals.append([idx, n_pareto])
                sampled_points.append(y_samples)

            n_pareto_vals = np.asarray(n_pareto_vals)
            n_pareto_max = np.max(n_pareto_vals[:, 1])
            top_values = []

            for i in range(n_samples):
                if n_pareto_vals[i, 1] > tolerance * n_pareto_max:
                    top_values.append(n_pareto_vals[i, 0])
            argmax_idx = np.random.choice(top_values)

            ## Graph
            # ss = [[self.gp[0].predict([x_tries[argmax_idx]], return_std=True)[0],
            #        self.gp[1].predict([x_tries[argmax_idx]], return_std=True)[0]]]
            # ss += [sampled_points[argmax_idx]]
            # self.process_bb.draw_plot_pareto(lim =self.obj_space_lim, sampled_point= ss)
        x_max = x_tries[argmax_idx]
        return x_max

    def latin_cube_init(self):
        samples = lhs(self.dim_design_space,samples= self.num_init_samples,criterion='center')
        train_ = np.zeros(samples.shape)

        for i in range(self.dim_design_space):
            loc_ = self.bounds[i, 0]
            scale_ = self.bounds[i, 1]- self.bounds[i,0]
            train_[:, i] = uniform.ppf(samples[:, i], loc=loc_, scale= scale_)

        self.design_parameter += train_.tolist()
        self.train_features = np.vstack((self.train_features, train_))

        parent_idx = -1
        for idx in range(self.num_init_samples):
            num_success_, grasp_quality_, mass_ = self.do_experiment(idx)
            self.num_trials += [self.iter_per_object]
            self.num_success += [num_success_]
            self.grasp_quality += [grasp_quality_]
            self.mass += [mass_]
            print(mass_)
            # TODO : post processing grasp_quality values
            new_label = self.post_processing_per_design(idx)
            print(new_label)
            parent_idx = self.process_bb.add_observation(new_label[0], new_label[1], np.average(num_success_))

        while parent_idx != -1:
            num_success_, grasp_quality_, _ = self.do_experiment(parent_idx)
            # update data
            self.num_success[parent_idx] += num_success_
            for obj_idx in range(self.num_objects):
                self.grasp_quality[parent_idx][obj_idx] = np.hstack((self.grasp_quality[parent_idx][obj_idx],
                                                                     grasp_quality_[obj_idx]))

            self.num_trials[parent_idx] += self.iter_per_object
            new_label = self.post_processing_per_design(parent_idx)
            parent_idx = self.process_bb.update_observation(new_label[0], new_label[1],
                                                             np.average(self.num_success[parent_idx]),
                                                             parent_idx=parent_idx)

        """update gaussian process fitting & predict & new candidates"""
        self.train_labels = (self.process_bb.upper_bounds + self.process_bb.lower_bounds) * 0.5
        noise = ((self.process_bb.upper_bounds - self.process_bb.lower_bounds) * 0.5 / self.gamma)
        for obj in range(self.num_objectives):
            if obj == 1:
                self.gp[obj].alpha = noise[:, obj]
            self.gp[obj].fit(self.train_features, self.train_labels[:, obj])

        #Graph
        # x_set = []
        # for i in range(self.dim_design_space):
        #     x_i = np.linspace(self.bounds[i, 0], self.bounds[i, 1], 20)
        #     x_set.append(x_i)
        # x_set = np.asarray(x_set)
        # x_ndim = np.meshgrid(*x_set)
        # x_ndim_flatten = []
        # for i in range(self.dim_design_space):
        #     x_ndim_flatten.append(x_ndim[i].flatten())
        # x_ = np.array(x_ndim_flatten).T
        #
        # fontsize_title = 10
        # plt.figure(figsize=(5, 6))
        # for func_idx in range(self.num_objectives):
        #     function_name = "function " + str(func_idx)
        #     plt.subplot(2, 2, func_idx + 1)
        #     plt.title(function_name, fontsize=fontsize_title)
        #     f_i = self.gp[func_idx].sample_y(x_)
        #     plt.pcolormesh(x_ndim[0], x_ndim[1], f_i.reshape((20, 20)))
        #     plt.scatter(np.asarray(self.design_parameter)[:, 0], np.asarray(self.design_parameter)[:, 1], c='g',s=3)
        #     plt.colorbar()
        #     _, std = self.gp[func_idx].predict(x_, return_std=True)
        #     plt.subplot(2, 2, 3 + func_idx)
        #     plt.pcolormesh(x_ndim[0], x_ndim[1], std.reshape((20, 20)))
        #     plt.scatter(np.asarray(self.design_parameter)[:, 0], np.asarray(self.design_parameter)[:, 1], c='g',s=3)
        #     plt.title(function_name + " std", fontsize=fontsize_title)
        #     plt.colorbar()
        # self.process_bb.draw_plot_pareto(lim = self.obj_space_lim, add_exact_graph=False)


    def run(self):
        x_set = []
        for i in range(self.dim_design_space):
            x_i = np.linspace(self.bounds[i, 0], self.bounds[i, 1], 20)
            x_set.append(x_i)
        x_set = np.asarray(x_set)

        x_ndim = np.meshgrid(*x_set)

        x_ndim_flatten = []
        for i in range(self.dim_design_space):
            x_ndim_flatten.append(x_ndim[i].flatten())
        x_ = np.array(x_ndim_flatten).T

        argmax_idx = np.random.randint(1, x_set.shape[1], x_set.shape[0])

        x_max = []
        if self.init_with_lhs:
            self.latin_cube_init()
            pareto_set, _ = self.is_pareto_simple_max()
            x_max = self.acquisition_MC_random(pareto_set)
            print("start bayesian opt with first candidate : ",x_max)
        else:
            for i in range(self.dim_design_space):
                x_max += [x_set[i, argmax_idx[i]]]

        for idx in range(self.num_designs):
            print(" design: ",idx , ": ",  x_max)
            self.train_features = np.vstack((self.train_features, x_max))
            self.design_parameter += [x_max]
            num_success_, grasp_quality_, mass_ = self.do_experiment(-1)

            self.num_trials += [self.iter_per_object]
            self.num_success += [num_success_]
            self.grasp_quality += [grasp_quality_]
            self.mass += [mass_]
            print(mass_)
            # TODO : post processing grasp_quality values
            new_label = self.post_processing_per_design(idx)
            print(new_label)
            parent_idx = self.process_bb.add_observation(new_label[0], new_label[1], np.average(num_success_))
            while parent_idx != -1:
                num_success_, grasp_quality_, _ = self.do_experiment(parent_idx)
                self.num_success[parent_idx] += num_success_
                for obj_idx in range(self.num_objects):
                    self.grasp_quality[parent_idx][obj_idx] = np.hstack((self.grasp_quality[parent_idx][obj_idx],
                                                                         grasp_quality_[obj_idx]))

                self.num_trials[parent_idx] += self.iter_per_object
                new_label = self.post_processing_per_design(parent_idx)
                print(new_label)
                parent_idx = self.process_bb.update_observation(new_label[0], new_label[1],
                                                                 np.average(self.num_success[parent_idx]),
                                                                 parent_idx=parent_idx)

            """update gaussian process fitting & predict & new candidates"""
            self.train_labels = (self.process_bb.upper_bounds + self.process_bb.lower_bounds) * 0.5
            noise = ((self.process_bb.upper_bounds - self.process_bb.lower_bounds)*0.5/self.gamma)
            for obj in range(self.num_objectives):
                if obj == 1:
                    self.gp[obj].alpha = noise[:, obj]
                self.gp[obj].fit(self.train_features, self.train_labels[:, obj])
            x_prev = x_max
            if idx < self.num_designs-1:
                pareto_set, _ = self.is_pareto_simple_max()
                x_max = self.acquisition_MC_random(pareto_set)
                print("design parameters: ", len(self.design_parameter))

            ## Draw plot for acquisition function
            # fontsize_title = 10
            # plt.figure(figsize=(5, 6))
            # for func_idx in range(self.num_objectives):
            #     function_name = "function " + str(func_idx)
            #     plt.subplot(2, 2, func_idx + 1)
            #     plt.title(function_name, fontsize=fontsize_title)
            #     f_i = self.gp[func_idx].sample_y(x_)
            #     plt.pcolormesh(x_ndim[0], x_ndim[1], f_i.reshape((20, 20)))
            #     plt.scatter(np.asarray(self.design_parameter)[:, 0], np.asarray(self.design_parameter)[:, 1], c='g',
            #                 s=3)
            #     plt.scatter(x_prev[0], x_prev[1], c= 'b')
            #     plt.scatter(x_max[0], x_max[1], c='r')
            #     plt.colorbar()
            #     _, std = self.gp[func_idx].predict(x_, return_std=True)
            #     plt.subplot(2, 2, 3 + func_idx)
            #     plt.pcolormesh(x_ndim[0], x_ndim[1], std.reshape((20, 20)))
            #     plt.scatter(np.asarray(self.design_parameter)[:, 0], np.asarray(self.design_parameter)[:, 1], c='g',
            #                 s=3)
            #     plt.scatter(x_prev[0], x_prev[1], c= 'b')
            #     plt.scatter(x_max[0], x_max[1], c='r')
            #     plt.title(function_name + " std", fontsize=fontsize_title)
            #     plt.colorbar()
            # self.process_bb.draw_plot_pareto([[-7., 0], [0, 1]])

        fontsize_title = 10
        plt.figure(figsize=(5, 6))
        for func_idx in range(self.num_objectives):
            function_name = "function " + str(func_idx)
            plt.subplot(2, 2, func_idx + 1)
            plt.title(function_name, fontsize=fontsize_title)
            f_i = self.gp[func_idx].sample_y(x_)
            plt.pcolormesh(x_ndim[0], x_ndim[1], f_i.reshape((20, 20)))
            plt.scatter(np.asarray(self.design_parameter)[:, 0], np.asarray(self.design_parameter)[:, 1], c='g',s=3)
            plt.colorbar()
            _, std = self.gp[func_idx].predict(x_, return_std=True)
            plt.subplot(2, 2, 3 + func_idx)
            plt.pcolormesh(x_ndim[0], x_ndim[1], std.reshape((20, 20)))
            plt.scatter(np.asarray(self.design_parameter)[:, 0], np.asarray(self.design_parameter)[:, 1], c='g',s=3)
            plt.title(function_name + " std", fontsize=fontsize_title)
            plt.colorbar()

        self.process_bb.is_pareto_BB()
        self.process_bb.is_overlap_in_dominant()
        self.process_bb.draw_plot_pareto(lim=self.obj_space_lim)

    def get_result(self):
        pareto_d = np.asarray(self.design_parameter)[self.process_bb.is_pareto]
        print("pareto design: ", pareto_d)


if __name__ == '__main__':
    opt_design = optimize_design(init_with_lhs = False, num_init_samples =10,
                                 world_file_name='Simulation/box_robot_floating.xml',
                                 object_file_name='../ObjectNet3D/CAD/off/cup/[0-9][0-9].off',
                                 iter_per_obj=20, num_objectives=2,
                                 exps_th=0, gamma=1.96, num_designs=30,
                                 bounds=np.asarray([[-1.0, 1], [0, 1.]]),
                                 obj_space_lim = [[-7., 0], [0, 1]],
                                 show_step=False)
    opt_design.run()
    opt_design.get_result()
    print("---- %s seconds --- " % (time.time() - start_time))
    plt.show()
