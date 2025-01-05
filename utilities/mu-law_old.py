import numpy as np
class MuLaw():
    def __init__(self,
                 mu:float,
                 M:float):
        self.mu=mu
        self.M=M
        self.regularization=np.log(mu*M+1.0)
    
    def __call__(self,x:float, *args, **kwds):
        return  np.sign(x)*np.log(np.abs(x)*self.mu+1.0)/self.regularization
