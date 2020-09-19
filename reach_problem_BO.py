from problem_BO import ProblemBO
from metric import *
import numpy as np
import math,copy,os,shutil,time,multiprocessing

class Circle:
    def __init__(self,ctr,rad):
        self.ctr=np.array(ctr)
        self.rad=rad

    def dist(self,a,b):
        len=np.linalg.norm(b-a)
        dir=(b-a)/len
        dist=np.dot(dir,self.ctr-a)
        dist=max(min(dist,len),0)
        dist=np.linalg.norm(self.ctr-(a+dir*dist))
        return max(dist-self.rad,0)

class Reacher:
    def __init__(self,size,angle,target,obstacles):
        self.size=size
        self.angle=angle
        self.target=target
        self.obstacles=obstacles
        
    def ee(self):
        p0=np.array([0.,0.])
        p1=self.rot_2D(self.angle[0])@np.array([0,self.size[0]])
        p2=self.rot_2D(self.angle[0])@(self.rot_2D(self.angle[1])@np.array([0,self.size[1]])+np.array([0,self.size[0]]))
        return p0,p1,p2

    def dist_to_target(self):
        return np.linalg.norm(self.ee()[2]-self.target)

    def dist_from_obstacle(self):
        ret=10000.
        p0,p1,p2=self.ee()
        for obs in self.obstacles:
            ret=min(min(ret,obs.dist(p0,p1)),obs.dist(p1,p2))
        return ret

    @staticmethod
    def rot_2D(a):
        return np.array([[ math.cos(a),-math.sin(a)],
                         [ math.sin(a), math.cos(a)]])

class ReacherSizeMetric(Metric):
    def __init__(self):
        Metric.__init__(self,OBJECT_DEPENDENT=False)
    
    def compute(self,reacher):
        return reacher.size[0]+reacher.size[1]

class ReachQualityMetric(Metric):
    def __init__(self):
        Metric.__init__(self,OBJECT_DEPENDENT=True)
    
    def compute(self,reacher):
        ret=math.exp(-reacher.dist_to_target())
        if reacher.dist_from_obstacle()==0.:
            ret-=1
        return ret

class ReachProblemBO(ProblemBO):
    def __init__(self,*,objects,obstacles,policy_space):
        self.objects=objects
        self.obstacles=obstacles
        
        #policy
        self.policy_names=['angle0' ,'angle1']
        self.policy_vmin= [0        ,-math.pi]
        self.policy_vmax= [math.pi  , math.pi]
        self.policy_init= [math.pi/2,0       ]
        
        #metrics
        self.metrics=[ReacherSizeMetric(),ReachQualityMetric()]
        
        design_space=[('len0',(0.1,1.0)),('len1',(0.1,1.0))]
        ProblemBO.__init__(self,design_space,policy_space)
        
    def compute_metrics(self,points,parallel=True,remove_tmp=True,visualize=False):
        reacher_metrics=[]
        for p in points:
            r=Reacher(np.array(p[:2]),None,None,None)
            reacher_metrics.append([m.compute(r) if not m.OBJECT_DEPENDENT else 0. for m in self.metrics])
            
        all_metrics=[]
        for pt in points:    #point
            policy_metrics=[]
            for policy in self.policies:    #policy
                #init policy
                for id,v in zip(self.vpolicyid,pt):
                    if id>=0:
                        policy[id]=v
                for k,v in self.mimic:
                    policy[k[0]]=[p*v for p in pt[k[1]]] if isinstance(pt[k[1]],list) else pt[k[1]]*v
            
                object_metrics=[]
                for o in self.objects:  #object
                    policyo=[p[id] if isinstance(p,list) else p for p in policy]
                    r=Reacher(np.array(pt[:2]),np.array(policyo),o,self.obstacles)
                    object_metrics.append([m.compute(r) for m in self.metrics])
                policy_metrics.append(object_metrics)
            all_metrics.append(policy_metrics)        
        return reacher_metrics,all_metrics
    
    def name(self):
        return 'ReachProblemBO'

    def visualize(self):
        SZ,sz=500,2.
        import pygame
        pygame.init()
        screen=pygame.display.set_mode([SZ,SZ])
        def to_screen(pt):
            if isinstance(pt,float) or isinstance(pt,int):
                return int(pt*SZ/sz)
            else: 
                return (int((pt[0]+sz/2)*SZ/sz),SZ-int(pt[1]*SZ/sz))
        
        id=0
        pt=[v for v in self.vmin]
        running=True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        pt[id]=min(pt[id]+0.01,self.vmax[id])
                    elif event.key == pygame.K_DOWN:
                        pt[id]=max(pt[id]-0.01,self.vmin[id])
                    elif event.key == pygame.K_LEFT:
                        id=(id+len(pt)-1)%len(pt)
                    else:
                        id=(id+1)%len(pt)
            #display
            screen.fill((255,255,255))
            for c in self.obstacles:
                pygame.draw.circle(screen,(0,0,255),to_screen(c.ctr),to_screen(c.rad))
            for o in self.objects:
                pygame.draw.circle(screen,(255,0,0),to_screen(o),to_screen(0.02))
                
            #display info
            pygame.font.init()
            font = pygame.font.SysFont('Comic Sans MS', 30)
            ptInfo="["
            for ip,p in enumerate(pt):
                if ip==id:
                    ptInfo+="("+str(p)+")"
                else: ptInfo+=str(p)
                ptInfo+="," if ip<len(pt)-1 else "]"
            textsurface = font.render(ptInfo, False, (0,0,0))
            screen.blit(textsurface,(10,0))
            pygame.display.flip()
        pygame.quit()

if __name__=='__main__':
    objects=[(-0.5,1.0),(0.0,1.0),(0.5,1.0)]
    obstacles=[Circle((-0.35,0.5),0.2),Circle((0.35,0.5),0.2)]
    reach=ReachProblemBO(objects=objects,obstacles=obstacles,policy_space=[('angle0',5),('angle1',5)])
    print(reach.eval([(0.1,0.2),(0.3,0.4)]))
    print(reach.eval([(0.1,0.2),(0.3,0.4)],avgObject=False))
    reach.visualize()