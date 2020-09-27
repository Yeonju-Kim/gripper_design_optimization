from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic)
from gaussian_process_scaled import GaussianProcessScaled
import DIRECT,pickle,os,copy
from problem_BO import *
import numpy as np

class SingleObjectiveBOGPUCB:
    def __init__(self,problemBO,kappa=10.,nu=None,length_scale=1.):
        if nu is not None:
            kernel=Matern(nu=nu,length_scale=length_scale)
        else: kernel=RBF(length_scale=length_scale)
        self.gp=GaussianProcessScaled(kernel=kernel,n_restarts_optimizer=25,alpha=0.0001)
        self.problemBO=problemBO
        self.kappa=kappa
        if len(self.problemBO.metrics)>1:
            raise RuntimeError('SingleObjectiveBO passed with multiple metrics!')
        
    def init(self,num_grid):
        coordinates=[np.linspace(vminVal,vmaxVal,num_grid) for vminVal,vmaxVal in zip(self.problemBO.vmin,self.problemBO.vmax)]
        if os.path.exists('init.dat'):
            self.load('init.dat')
        else:
            self.points=np.array([dimi.flatten() for dimi in np.meshgrid(*coordinates)]).T.tolist()
            self.scores=[metrics[0] for metrics in self.problemBO.eval(self.points)]
            #self.save('init.dat')
        self.gp.fit(self.points,self.scores)
        
    def iterate(self):
        def obj(x,user_data):
            return -self.acquisition(x),0
        point,acquisition_val,ierror=DIRECT.solve(obj,self.problemBO.vmin,self.problemBO.vmax,logfilename='../direct.txt',algmethod=1)
        point=point.tolist()
        score=self.problemBO.eval([point])[0][0]
        self.points.append(point)
        self.scores.append(score)
        self.gp.fit(self.points,self.scores)
        
    def acquisition(self,x,user_data=None):
        #GP-UCB
        mu,sigma=self.gp.predict([x],return_std=True)
        return mu[0]+sigma[0]*self.kappa
    
    def run(self, num_grid=5, num_iter=100):
        self.init(num_grid)
        for i in range(num_iter):
            print("Single-Objective BO Iter=%d!"%i)
            self.iterate()
          
    def load(self,filename):
        self.points,self.scores=pickle.load(open(filename,'rb'))
        self.gp.fit(self.points,self.scores)
            
    def save(self,filename):
        pickle.dump((self.points,self.scores),open(filename,'wb'))
            
    def name(self):
        return 'SBO-GP-UCB('+self.problemBO.name()+')'
            
    def get_best(self):
        kappa_tmp=self.kappa
        self.kappa=0.
        def obj(x,user_data):
            return -self.acquisition(x),0
        point,acquisition_val,ierror=DIRECT.solve(obj,self.problemBO.vmin,self.problemBO.vmax)
        self.kappa=kappa_tmp
        return point.tolist()

    def plot_iteration(self,plt,accumulate=False,eps=0.1):
        scores=np.array(self.scores)
        if accumulate:
            yss=np.maximum.accumulate(scores)
        else: yss=scores
        
        #plt
        plt.plot(yss,'o-',label='Sampled points')
        plt.title('Single-Objective '+self.name()+('-accumulate' if accumulate else ''))
        plt.legend(loc='lower right')
        plt.xlabel('Iteration')
        plt.ylabel('Metric')
        
        #range rescaled
        vmin=yss.min()
        vmax=yss.max()
        vrng=vmax-vmin
        plt.ylim(vmin-vrng*eps,vmax+vrng*eps)
        return plt

    def plot_func_1D(self,plt,eps,res,repeat):
        assert len(self.problemBO.vmin)==1
        fig,ax=plt.subplots()
        ln_pt,=plt.plot([],[],'o',label='Sample')
        ln_mean,=plt.plot([],[],'b-',label='Prediction')
        ln_sigma,=plt.fill([],[],alpha=.5,fc='b',ec='None',label='.95 confidence interval')
        ax.set_title('Single-Objective '+self.name())
        
        gps=[]
        def init():
            #range
            ax.set_xlim(self.problemBO.vmin[0],self.problemBO.vmax[0])
            vmin=np.array(self.scores).min()
            vmax=np.array(self.scores).max()
            vrng=vmax-vmin
            ax.set_ylim(vmin-vrng*eps,vmax+vrng*eps)
            return ln_pt,ln_mean,ln_sigma
        
        def update(frame):
            if len(gps)==frame:
                gps.append(copy.deepcopy(self.gp))
                xdata=[self.points[i][0] for i in range(frame+1)]
                ydata=[self.scores[i] for i in range(frame+1)]
                gps[-1].fit([[i] for i in xdata],ydata)
            #sampled points
            xdata=[self.points[i][0] for i in range(frame+1)]
            ydata=[self.scores[i] for i in range(frame+1)]
            ln_pt.set_data(xdata,ydata)
            
            #predicted mean of GP
            xdata=np.linspace(self.problemBO.vmin[0],self.problemBO.vmax[0],res).tolist()
            ydata,sdata=gps[frame].predict([[i] for i in xdata],return_std=True)
            ln_mean.set_data(xdata,ydata)
            
            #predicted variance of GP
            xdata=np.array(xdata)
            xdata=np.concatenate([xdata,xdata[::-1]])
            sdata=np.concatenate([ydata-1.9600*sdata,(ydata+1.9600*sdata)[::-1]])
            xsdata=np.stack([xdata,sdata],axis=1)
            ln_sigma.set_xy(xsdata)
            return ln_pt,ln_mean,ln_sigma
        
        from matplotlib.animation import FuncAnimation
        ani=FuncAnimation(fig,update,frames=[i for i in range(len(self.points))],init_func=init,blit=True,repeat=repeat)
        plt.legend(loc='lower right')
        return plt,ani

    def plot_func_2D(self,plt,eps,res,repeat):
        assert len(self.problemBO.vmin)==2
        coordinates=[np.linspace(vminVal,vmaxVal,res) for vminVal,vmaxVal in zip(self.problemBO.vmin,self.problemBO.vmax)]
        xsmesh,ysmesh=np.meshgrid(*coordinates)
        ptsmesh=np.array([dimi.flatten() for dimi in [xsmesh,ysmesh]]).T.tolist()
        zsmesh=self.gp.predict(ptsmesh).reshape(xsmesh.shape)
        
        from mpl_toolkits import mplot3d
        from matplotlib import cm
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ln_pt=ax.scatter(xs=[],ys=[],zs=[],label='Sample')
        ln_mean=ax.plot_wireframe(X=xsmesh,Y=ysmesh,Z=zsmesh,label='Prediction',color='red')
        ax.set_title('Single-Objective '+self.name())
        
        gps=[]
        def init():
            #range
            ax.set_xlim3d(self.problemBO.vmin[0],self.problemBO.vmax[0])
            ax.set_ylim3d(self.problemBO.vmin[1],self.problemBO.vmax[1])
            vmin=np.array(self.scores).min()
            vmax=np.array(self.scores).max()
            vrng=vmax-vmin
            ax.set_zlim3d(vmin-vrng*eps,vmax+vrng*eps)
            return ln_pt,ln_mean
        
        def update(frame):
            if len(gps)==frame:
                gps.append(copy.deepcopy(self.gp))
                xdata=[self.points[i][0] for i in range(frame+1)]
                ydata=[self.points[i][1] for i in range(frame+1)]
                zdata=[self.scores[i] for i in range(frame+1)]
                gps[-1].fit([[x,y] for x,y in zip(xdata,ydata)],zdata)
            #sampled points
            xdata=[self.points[i][0] for i in range(frame+1)]
            ydata=[self.points[i][1] for i in range(frame+1)]
            zdata=[self.scores[i] for i in range(frame+1)]
            ln_pt._offsets3d=(xdata,ydata,zdata)
            
            #predicted mean of GP
            xsmesh=ln_mean._segments3d[:,:,0]
            ysmesh=ln_mean._segments3d[:,:,1]
            ptsmesh=np.array([dimi.flatten() for dimi in [xsmesh,ysmesh]]).T.tolist()
            zsmesh=gps[frame].predict(ptsmesh).reshape(xsmesh.shape)
            ln_mean._segments3d=np.stack([xsmesh,ysmesh,zsmesh],axis=2)
            return ln_pt,ln_mean

        from matplotlib.animation import FuncAnimation
        ani=FuncAnimation(fig,update,frames=[i for i in range(len(self.points))],init_func=init,blit=False,repeat=repeat)
        plt.legend(loc='lower right')
        return plt,ani

    def plot_func(self,plt,eps=0.1,res=256,repeat=False):
        if len(self.problemBO.vmin)==1:
            return self.plot_func_1D(plt,eps,res,repeat)
        elif len(self.problemBO.vmin)==2:
            return self.plot_func_2D(plt,eps,res,repeat)
        else: raise RuntimeError('Plotting domain dimension > 2 is not supported!')

def debug_toy_problem(BO,num_iter=10,repeat=False):
    path='../'+BO.name()+'.dat'
    if not os.path.exists(path):
        BO.run(num_iter=num_iter)
        BO.save(path)
    else:
        BO.load(path)
    #BO.get_best()
    
    import matplotlib.pyplot as plt   
    BO.plot_iteration(plt,accumulate=True)
    BO.plot_iteration(plt,accumulate=False)
    plt.show()
    plt.close()
    
    _,ani=BO.plot_func(plt,repeat)
    plt.show()
    plt.close()

if __name__=='__main__':
    problem=Test1D1MProblemBO()
    BO=SingleObjectiveBOGPUCB(problem)
    debug_toy_problem(BO)
    
    problem=Test2D1MProblemBO()
    BO=SingleObjectiveBOGPUCB(problem)
    debug_toy_problem(BO)