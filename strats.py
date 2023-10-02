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
    
    def rew_strat_1(self, action_hist, valid, last, reward, sal):
        r= 0.01
        terminate= False
        
        stack_sum= action_hist.sum(dim=0)[:-1]
        action= action_hist[-1]

        #Wrong class
        if (action * valid).sum(dim=-1) < 0.5:
            r = -1
            terminate= True
        
        #Multiple selections of same class
        elif torch.any(stack_sum> 1.5):
            r = -1
            terminate = True
            
        elif sal.sum() > 1:
            r= -1
            terminate= True
        
        
        elif action[:-1].sum() == 1:
            terminate= False
            r = 0.4*reward[-1]
            if torch.all(stack_sum == 1):
                terminate= True
                r= 0.02*(reward.sum()**4)
                # print(float(r))
        
        if not terminate and last:
            terminate = True
            r= -1
        
        return torch.Tensor([r]), torch.Tensor([terminate])
    
    def reward(self, action, action_hist, x, fpts):
        
        action_hist[:,-1] = action
        valid_actions, last, salaries, proj= decat(x, self.decat)
        
        sel_ind= action_hist[:,:,:-1].bool().any(dim=-1)
        
        # print(valid_actions)
        
        r_out= []
        t_out= []
        for n in range(len(action)):
            # sel_ind= action_hist[:-1]
            
            sel_r= fpts[n, sel_ind[n]]
            sel_sal= salaries[n, sel_ind[n]]
            r, t= self.rew_strat_1(action_hist[n], valid_actions[n,-1], last[n, -1], sel_r, sel_sal)
            r_out.append(r)
            t_out.append(t)
            
        return cat(r_out), cat(t_out).bool()