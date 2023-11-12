import torch
import random
import numpy as np

cat= torch.cat

def decat(out, cat_order, dev= 'cpu'):
    assert sum(cat_order) == out.shape[-1], "Cat orders don't match"
    
    og_shape= list(out.shape[:-1])
    tot_out= []
    tot= 0
    flatout= out.flatten(0, -2)
    for i in cat_order:
        o= tot_out.append(flatout[:,tot:i+tot].view(og_shape + [-1]).to(dev))
        tot += i

    return tot_out

class Explorer():
    def __init__(self, out_size, epsilon_range= (0.05, 0.9), decay_rate= 0.999):
        self.epsilon= epsilon_range[1]
        self.epsilon_min= epsilon_range[0]
        self.decay_rate= decay_rate
        self.out_size= out_size
        
        self.weights= torch.ones([out_size])
        self.weights[-1] = 5
        
    def step(self):
        self.epsilon*= self.decay_rate
        
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        
    
    def explore(self, inds, q=None):
        out= -1
        explore_bool= float(torch.rand([1])) < self.epsilon
        
        if explore_bool:
            out= int(np.random.choice(list(inds), [1]))
            
        return explore_bool, int(out)
    
class Rewarder():
    def __init__(self, batch_size, decat):
        self.decat= decat
        self.NORM= float(3**5)
    
    def reward(self, action, x, y):
        
        r= torch.zeros_like(action).float()
        
        valid_actions, salaries, proj= decat(x, self.decat)
        
        sal_sel= salaries[action].squeeze(-1)
        r_sel= y[action].squeeze(-1)
        
        sal_tot= torch.zeros(sal_sel.shape)
        for n in range(sal_sel.shape[1]):
            sal_tot[:,n] = sal_sel[:,:n+1].sum(dim=-1)
        
        sal_good= (sal_tot < 1)
        sal_bad= (~sal_good)
        r = r_sel
        
        r[sal_bad] = -3
        
        return r