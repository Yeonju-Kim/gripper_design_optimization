from compile_objects import auto_download
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic)
from design_space import Domain
import numpy as np
import DIRECT

class SingleObjectiveBO:
    def __init__(self,domain,kappa=10.):
        self.domain=domain
        self.kappa=kappa
        
    def init(self,num_grid):
        coordinates=[np.linspace(vminVal,vmaxVal,num_grid) for vminVal,vmaxVal in range(self.domain.vmin,self.domain.vmax)]
        self.points=np.array([dimi.flatten() for dimi in np.meshgrid(coordinates)]).T
        self.scores=[metrics[0] for metrics in self.domain.eval(points)]
        
        #fit initial GP
        self.gp=GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=25, alpha=0.0001)
        self.gp.fit(self.points,self.scores)
        
    def iterate(self):
        def obj(x,user_data):
            return self.acquisition(x),0
        point,acquisition_val,ierror=DIRECT.solve(obj,self.domain.vmin,self.domain.vmax)
        score=self.domain.eval([point])[0][0]
        self.points.append(point)
        self.scores.append(score)
        
    def acquisition(self,x):
        #GP-UCB
        mu,sigma=self.GP.predict(x,return_std=True)
        return mu+sigma*self.kappa
    
    def run(self, num_grid=5, num_iter=100):
        self.init(num_grid)
        for i in range(num_iter):
            print("Single-Objective BO Iter=%d!"%i)
            self.iterate()
            
if __name__=='__main__':
    auto_download()
    
    domain=Domain(design_space='finger_length:0.2,0.5|finger_curvature:-2,2',metrics='ElapsedMetric')
    BO=SingleObjectiveBO(domain)
    BO.run()