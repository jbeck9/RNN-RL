import torch
import random

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
    def __init__(self, out_size, init_epsilon):
        self.epsilon= init_epsilon
        self.out_size= out_size
        
        self.weights= torch.ones([out_size])
        self.weights[-1] = 2
        
    def ind_to_action(self, ind):
        out= torch.zeros([self.out_size])
        out[ind] = 1
        
        return out
    
    def exp_strat_1(self, valid):
        mask= valid.bool()
        
        i= torch.Tensor(range(self.out_size))[mask].long()
        w= self.weights[mask]
        
        ind= random.choices(list(i), weights=w)
        
        return self.ind_to_action(ind).unsqueeze(0)
        
    
    def explore(self, valid):
        out= []
        explore_mask= torch.rand([valid.shape[0]]) < self.epsilon
        
        for n in range(len(valid)):
            if explore_mask[n]:
                out.append(self.exp_strat_1(valid[n]))
            else:
                out.append(torch.zeros([self.out_size]).unsqueeze(0))
            
        return cat(out, dim=0)[explore_mask], explore_mask
    
class Rewarder():
    def __init__(self, batch_size, decat):
        self.decat= decat
        self.batch_size= batch_size
        
        self.reset()
        
    def reset(self):
        self.r_cum= torch.zeros([self.batch_size])
        self.c_cum= torch.zeros([self.batch_size])
    
    def rew_strat_1(self, action_hist, valid, last):
        r= -1
        terminate= False
        
        stack_sum= action_hist.sum(dim=0)[:-1]
        action= action_hist[0]

        #Wrong class
        if (action * valid).sum(dim=-1) < 0.5:
            r = -1
            terminate= True
            return torch.Tensor([r]), torch.Tensor([terminate])
        
        #Multiple selections of same class
        if torch.any(stack_sum> 1.5):
            r = -1
            terminate = True
            return torch.Tensor([r]), torch.Tensor([terminate])
        
        
        if action[:-1].sum() == 1:
            terminate= False
            r = 1
            if torch.all(stack_sum == 1):
                r= 5
                terminate= True
        
        return torch.Tensor([r]), torch.Tensor([terminate])
    
    def reward(self, action, action_hist, X_hist, fpts):
        
        action_hist[:,0] = action
        valid_actions, last, salaries, proj= decat(X_hist, self.decat)
        
        # print(valid_actions)
        
        r_out= []
        t_out= []
        for n in range(len(action)):
            if action[n, :-1].any():
                self.r_cum[n] += fpts[n,0]
                self.c_cum[n] += salaries[n,0]
            
            r, t= self.rew_strat_1(action_hist[n], valid_actions[n], last[n])
            r_out.append(r)
            t_out.append(t)
            
        return cat(r_out), cat(t_out).bool()