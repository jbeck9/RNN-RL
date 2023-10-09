import torch
import random
from collections import deque

import numpy as np

def decat(out, cat_order, device= 'cpu'):
    assert sum(cat_order) == out.shape[-1], "Cat orders don't match"
    
    og_shape= list(out.shape[:-1])
    tot_out= []
    tot= 0
    flatout= out.flatten(0, -2)
    for i in cat_order:
        tot_out.append(flatout[:,tot:i+tot].view(og_shape + [-1]).to(device))
        tot += i

    return tot_out

class ReplayBuffer():
    def __init__(self, max_size=50, cat_order= None):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []
        
        self.cat_order= cat_order
        
    def decat(self, out):
        tot_out= []
        tot= 0
        for i in self.cat_order:
            tot_out.append(out[:,tot:i+tot])
            tot += i
            
        # return Transition(*tot_out)
        return tot_out

    def pop(self, data):
        to_return = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i])
                    self.data[i] = element
                else:
                    to_return.append(element)
        out= torch.cat(to_return)
        
        if self.cat_order is not None:
            return self.decat(out)
        else:
            return out
        
class ReplayMemory(object):
    def __init__(self, capacity, decat= None):
        self.memory = deque([], maxlen=capacity)
        self.cat_order= decat

    def push(self, dat):
        """Save a transition"""
        self.memory.extend(dat)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
def norm(x, a=0, b=1):
    return (((b-a) * (x - x.min())) / (x.max() - x.min())) + a


def data_gen(batch_size=64, inst_size= 100, class_size= 5):
    
    cl= torch.rand([batch_size, inst_size, class_size])
    cl= (cl > 0.5).float()
    
    sal= (torch.rand([batch_size, inst_size, 1]) + 0.1) * 0.4
    
    proj= torch.rand([batch_size, inst_size, 1])
    
    fpts= torch.normal(proj, 0.3)
    
    fpts= norm(fpts)
    proj= norm(proj)
    
    # print(float(torch.corrcoef(torch.cat([fpts, proj], dim=-1).flatten(0,1).T)[1,0]))
    
    x= torch.cat([cl, sal, proj], dim=-1), fpts
    
    return x


def sample_eval(x, sal, rew, nsamples= 30000):
    inds= np.array(list(range(len(x))))
    n_space= x.shape[-1]
    
    ind= 0.1 * np.ones([nsamples, n_space])
    for ci in range(n_space):
        mask= x[:,ci].bool().numpy()
        ind[:, ci] = np.random.choice(inds[mask], [nsamples])
    
    ind= np.unique(np.sort(ind, axis=1), axis=0)
    ind= ind[np.array([len(np.unique(b)) == n_space for b in ind]).astype(bool)]
    
    ind= ind[sal[ind].sum(dim=1)[:,0] < 1]
    return rew[ind].sum(dim=1)[:,0].max()
    
    
    
    # x_choices= x[ind]
    # valid_mask_1= np.array([np.diag(b) for b in x_choices])
    
if __name__ == "__main__":
    in_cat= [6, 1, 1, 1]
    x,y= data_gen(batch_size= 1, inst_size= 30, class_size= 6)
    x, last, sal, proj= decat(x, in_cat)
    
    sample_eval(x[0,:,:-1], sal[0], y[0])