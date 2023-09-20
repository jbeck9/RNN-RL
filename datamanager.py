import torch
import random
from collections import deque

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
    
    indicator= torch.zeros([batch_size, inst_size, 1])
    indicator[:,-1,:] = 1
    
    cl= torch.rand([batch_size, inst_size, class_size])
    cl= (cl > 0.5).float()
    cl[:,:,-1] = 1.
    
    sal= (torch.rand([batch_size, inst_size, 1]) + 0.1) * 0.5
    
    proj= torch.rand([batch_size, inst_size, 1])
    
    fpts= torch.normal(proj, 0.3)
    
    fpts= norm(fpts)
    proj= norm(proj)
    
    # print(float(torch.corrcoef(torch.cat([fpts, proj], dim=-1).flatten(0,1).T)[1,0]))
    
    x= torch.cat([cl, indicator, sal, proj], dim=-1), fpts
    
    return x