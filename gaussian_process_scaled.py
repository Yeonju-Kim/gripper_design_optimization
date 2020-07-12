from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic)
import numpy as np

class GaussianProcessScaled:
    def __init__(self,**args):
        self.gp=GaussianProcessRegressor(**args)
    
    def fit(self,x,y):
        self.vmin=np.array(y).min()
        self.vmax=np.array(y).max()
        self.base=self.vmin
        if self.vmax==self.vmin:
            self.slope=1.
        else: self.slope=1./(self.vmax-self.vmin)
        self.gp.fit(x,[(yi-self.base)*self.slope for yi in y])
        
    def predict(self,xss,return_std=False):
        ret=self.gp.predict(xss,return_std=return_std)
        if return_std:
            return (np.array([m/self.slope+self.base for m in ret[0]]),np.array([s/self.slope for s in ret[1]]))
        else: return np.array([m/self.slope+self.base for m in ret])