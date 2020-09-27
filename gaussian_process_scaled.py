from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic)
import numpy as np

class GaussianProcessScaled:
    def __init__(self,**args):
        self.args=args
        self.gp=None
    
    def fit(self,x,y):
        assert len(x)==len(y)
        x=np.array(x)
        if len(x.shape)==1:
            x=np.expand_dims(x,axis=1)
            
        y=np.array(y)
        if len(y.shape)==1:
            y=np.expand_dims(y,axis=1)
        
        self.vminx=x.min(axis=0)
        self.vmaxx=x.max(axis=0)
        i=0
        for a,b in zip(self.vminx,self.vmaxx):
            if a==b:
                self.vminx[i]=0
                self.vmaxx[i]=1
            i+=1
        self.slopex=np.divide(1.,np.subtract(self.vmaxx,self.vminx))
        
        self.vminy=y.min(axis=0)
        self.vmaxy=y.max(axis=0)
        i=0
        for a,b in zip(self.vminy,self.vmaxy):
            if a==b:
                self.vminy[i]=0
                self.vmaxy[i]=1
            i+=1
        self.slopey=np.divide(1.,np.subtract(self.vmaxy,self.vminy))
        
        x=np.multiply(np.subtract(x,self.vminx),self.slopex)
        y=np.multiply(np.subtract(y,self.vminy),self.slopey)
        if self.gp is None or len(self.gp)!=y.shape[1]:
            self.gp=[GaussianProcessRegressor(**self.args) for n in range(y.shape[1])]
        for i,gp in enumerate(self.gp):
            gp.fit(x,y[:,i])
        
    def predict(self,x,return_std=False):
        x=np.multiply(np.subtract(np.array(x),self.vminx),self.slopex)
        ret=[gp.predict(x,return_std=return_std) for gp in self.gp]
        if return_std:
            mean=np.vstack(tuple(r[0] for r in ret)).T
            mean=np.add(np.divide(mean,self.slopey),self.vminy)
            if mean.shape[1]==1:
                mean=mean[:,0]
            
            std=np.vstack(tuple(r[1] for r in ret)).T
            std=np.divide(std,self.slopey)
            if std.shape[1]==1:
                std=std[:,0]
            return mean,std
        else: 
            mean=np.vstack(tuple(r for r in ret)).T
            mean=np.add(np.divide(mean,self.slopey),self.vminy)
            if mean.shape[1]==1:
                mean=mean[:,0]
            return mean
        
if __name__=='__main__':
    kappa=10.
    nu=None
    length_scale=1.
    if nu is not None:
        kernel=Matern(nu=nu,length_scale=length_scale)
    else: kernel=RBF(length_scale=length_scale)
    gp=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
    
    x=[]
    y=[]
    ys=[]
    N,M=2,3
    D,K=100,7
    import random
    for i in range(D):
        x.append([random.uniform(-1.,1.) for i in range(N)])
        y.append([random.uniform(-1.,1.) for i in range(M)])
        ys.append(random.uniform(-1.,1.))
    
    gp.fit(x,y)
    gp.predict([[random.uniform(-1.,1.) for i in range(N)] for i in range(K)],True)
    gp.predict([[random.uniform(-1.,1.) for i in range(N)] for i in range(K)],False)
    
    gp.fit(x,ys)
    gp.predict([[random.uniform(-1.,1.) for i in range(N)] for i in range(K)],True)
    gp.predict([[random.uniform(-1.,1.) for i in range(N)] for i in range(K)],False)