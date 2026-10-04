"""Observable deviation triggers, not counterfactual failure labels."""
import torch


class FailureDetector:
    NAMES={0:'none',1:'contact_loss',2:'angle_regression',3:'ee_offset',4:'stagnation',5:'joint_limit_risk'}

    def __init__(self,n,device,bounds=None):
        self.bounds=bounds
        self.age=torch.zeros(n,device=device,dtype=torch.long)
        self.peak=torch.zeros(n,device=device)
        self.previous=self.peak.clone()
        self.stall=self.age.clone();self.contact_run=self.age.clone()
        self.lost=self.age.clone();self.regress=self.age.clone()
        self.contacted=torch.zeros(n,device=device,dtype=torch.bool)

    def reset(self,ids):
        for value in [self.age,self.peak,self.previous,self.stall,self.contact_run,self.lost,self.regress,self.contacted]:value[ids]=0

    @torch.no_grad()
    def update(self,robot,gt):
        self.age+=1;angle=gt[:,0];both=(gt[:,9:11]>.5).all(-1)
        self.contact_run=torch.where(both,self.contact_run+1,0)
        self.contacted|=self.contact_run>=4
        self.lost=torch.where(self.contacted&~both,self.lost+1,0)
        self.peak=torch.maximum(self.peak,angle)
        self.regress=torch.where((self.peak>.12)&(self.peak-angle>.06),self.regress+1,0)
        improved=angle>self.previous+.001
        self.previous=torch.maximum(self.previous,angle)
        self.stall=torch.where(improved,0,self.stall+1)
        distance=torch.linalg.vector_norm(robot[:,18:21]-gt[:,2:5],dim=-1)
        code=torch.zeros_like(self.age)
        eligible=(self.age>=90)&(angle<1.)
        closed=robot[:,7:9].sum(-1)<.045
        code[eligible&(self.age>=210)&(self.stall>=100)&(closed|(distance<.055))]=4
        code[eligible&(self.age>=170)&(distance>.14)]=3
        code[eligible&(self.regress>=8)]=2
        code[eligible&(self.lost>=10)]=1
        # Maximum takeover latency leaves >=360 of the original 600 ticks.
        code[(self.age>=240)&(angle<.08)&(code==0)]=4
        if self.bounds is not None:
            q=robot[:,:7];velocity=robot[:,9:16]
            margin=torch.full_like(q,.20);margin[:,5]=.40
            risky=(((q-self.bounds[:,:,0])<margin)&(velocity<-.10))|(((self.bounds[:,:,1]-q)<margin)&(velocity>.10))
            # Wrist limits caused the observed branch failure. The elbow's
            # normal nominal approach runs near its lower limit; do not
            # turn that ordinary posture into an early recovery trigger.
            risky[:,:4]=False;risky[:,6]=False
            code[(self.age>=8)&risky.any(-1)&~both&(angle<.25)&(code==0)]=5
        return code,distance
