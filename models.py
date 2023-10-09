import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

import datamanager
import strats
import math

import numpy as np
import mdn

cat= torch.cat

def get_device(x):
    device= x.get_device()
    if device < 0:
        device= 'cpu'
    return device

def get_mask(T, max_T= None):
    if max_T is None:
        out= torch.zeros([len(T), int(max(T))]).long()
    else:
        out= torch.zeros([len(T), max_T]).long()
    
    for n in range(len(T)):
        ind= int(T[n])
        out[n, :ind] = True
        
    return out.bool()

def input_padded(x, model, mask, pad_val=-2):
    if mask is None:
        return model(x)
    
    sq_x= x[mask]
    sq_out= model(sq_x)
    
    base_shape= list(x.shape)
    base_shape[-1] = sq_out.shape[-1]
    base= pad_val * torch.ones(base_shape)
    base= base.to(sq_out.device)
    
    base[mask] = sq_out
    return base
    
def decat(out, cat_order, device= 'cpu'):
    assert sum(cat_order) == out.shape[-1], "Cat orders don't match"
    
    og_shape= list(out.shape[:-1])
    tot_out= []
    tot= 0
    flatout= out.flatten(0, -2)
    for i in cat_order:
        o= tot_out.append(flatout[:,tot:i+tot].view(og_shape + [-1]).to(device))
        tot += i

    return tot_out
    
def qpack(x, a, r):
    return [(x[n],a[n], r[n]) for n in range(len(x))]

def qunpack(q):
    x= []
    a= []
    r= []
    for n in q:
        x.append(n[0].unsqueeze(0))
        a.append(n[1].unsqueeze(0))
        r.append(n[2].unsqueeze(0))
        
    return cat(x, dim=0),cat(a, dim=0), cat(r, dim=0)

def get_class_mask(valid):
    out= torch.zeros_like(valid)
    
    for n in range(valid.shape[-1]):
        out[:,:,n] = valid[:,:,n]
    
    return out.bool()

def doubleq(cls):
    class DoubleQ(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
            
            self.policy= cls(*args, **kwargs)
            self.target= cls(*args, **kwargs)
            
            self.target.load_state_dict(self.policy.state_dict())
            
            self.tau = 0.002
            
        def target_copy(self):
            target_net_state_dict = self.target.state_dict()
            policy_net_state_dict = self.policy.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            
            self.target.load_state_dict(target_net_state_dict)
            
        def save(self, path= 'model.pt'):
            torch.save(self.policy.state_dict(), path)
        
        def load_model(self, path= 'model.pt'):
            self.policy.load_state_dict(torch.load(path))
            self.target.load_state_dict(self.policy.state_dict())
            
        
    return DoubleQ


class TEncoder(nn.Module):
    def __init__(self, input_size, nhidden, nhead=16, layers=2):
        super(TEncoder, self).__init__()
        
        self.layers= layers
        self.nhidden= nhidden
        self.layers= layers
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, batch_first= True)
        
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        net= [nn.Linear(input_size, nhidden), nn.LeakyReLU(0.1)]
        for _ in range(1):
            net.extend([nn.Linear(nhidden, nhidden), nn.LeakyReLU(0.1), nn.Dropout(0.1)])
        net.extend([nn.Linear(nhidden, nhidden)])
        self.in_linear= nn.Sequential(*net)
        
        
    def forward(self, x, mask):
        device= get_device(x)
        
        x= input_padded(x, self.in_linear, mask)
        return self.model(x, src_key_padding_mask= ~mask.to(device))

@doubleq
class RnnQnet(nn.Module):
    def __init__(self, nclasses, enc_size, seq_len, nhidden, layers, in_cat= None, padval= -2, lin_layers= 2, nmixes= 4):
        super().__init__()
        
        self.layers= layers
        self.nhidden= nhidden
        self.enc_size= enc_size
        self.nclasses= nclasses
        self.input_size= nclasses + 2
        self.padval= padval
        self.seqlen= seq_len
        self.flatmix_shape= nmixes * 3
        
        self.in_cat= in_cat
        
        self.inf_cat= [1, 1]
        
        
        self.hidden= None
        
        self.enc= TEncoder(self.input_size, enc_size)
        
        self.S_rnn = nn.GRU(
            input_size= nhidden,
            hidden_size= nhidden, 
            num_layers=layers, 
            batch_first=True
        )
        
        self.emb= mdn.MdnLinear(nhidden + nhidden, nhidden, nmixes)
        
    def inf(self, x, fpts, T, rewarder, explorer):
        explorer.epsilon= 0
        self.eval()
        
        cl, sal, proj= decat(x, self.in_cat)
        
        cl_mask= get_class_mask(x[:,:,:n_classes])
        mask= get_mask(T, max_T= seq_len).unsqueeze(-1).tile([1,1,n_classes]) * cl_mask
        with torch.no_grad():
            a, r= model.policy(x,fpts,rewarder.reward, explorer.explore, mask, nsacks= 1, sacks_indep= True)
            a= a.long()
            print(r)
            print((r < 0).any(dim=-1).count_nonzero())
        r_out= torch.gather(fpts.squeeze(-1), 1, a.flatten(1)).view(a.shape)
        
        return r_out, a
        
    def forward(self, x,y,r_fn, e_fn , mask, device= 'cuda', actions= None, nsacks= 1, sacks_indep= True):
        x= x.to(device)
        bs= x.shape[0]
        ind= torch.Tensor(range(self.seqlen))
        if actions is None:
            inf= True
            actions= self.padval * torch.ones([bs,nsacks,self.nclasses]).long()
            rewards= self.padval * torch.ones([bs,nsacks,self.nclasses])
        
        else:
            inf= False
            Q_a= self.padval * torch.ones([bs, nsacks,self.nclasses, self.flatmix_shape]).to(device)
            Q_amax= self.padval * torch.ones([bs, nsacks, self.nclasses, 1]).to(device)
        
        if not sacks_indep:
            hidden= torch.zeros([self.layers, bs, self.nhidden]).to(device)
            
        for k in range(nsacks):
            if sacks_indep:
                hidden= torch.zeros([self.layers, bs, self.nhidden]).to(device)
            sel_mask= torch.ones([bs, self.seqlen]).bool()
            for p in range(self.nclasses):
                enc_mask= mask[:,:,p] * sel_mask
                z= self.enc(x, enc_mask)
                
                i= torch.cat([z, hidden[-1].unsqueeze(1).tile([1, x.shape[1], 1])], dim=-1)
                Q= input_padded(i, self.emb, enc_mask)
                
                mem_i= torch.zeros([bs, self.nhidden]).to(device)
                for n in range(bs):
                    bm= enc_mask[n]
                    binds= ind[bm]
                    if inf:
                        if len(binds) > 0:
                            q= Q[n, bm]
                            exp_bool, aexp= e_fn(binds, q)
                            if exp_bool:
                                a= aexp
                            else:
                                # q_sample= mdn.GaussianMix(q).sample()
                                q_sample= mdn.GaussianMix(q).expectation()
                                a= int(binds[q_sample.argmax()])
                            sel_mask[n, a] = False
                        actions[n,k,p] = a
                    else:
                        a= actions[n,k,p]
                        if a >= 0:
                            Q_a[n,k,p]= Q[n, a]
                            Q_amax[n,k,p]= mdn.GaussianMix(Q[n,bm]).expectation().max()
                            sel_mask[n, a] = False
                            
                    mem_i[n] = z[n, a]
                    
                hmask= actions[:,k,p] >= 0
                _, hidden[:,hmask] = self.S_rnn(mem_i.unsqueeze(1)[hmask], hidden[:,hmask])
        
        if inf:
            for n in range(bs):
                rewards[n]= r_fn(actions[n], x[n], y[n])
            
            return actions, rewards
        
        else:
            out_batch_mask= (actions >= 0).all(dim=-1)
            terminate= torch.zeros_like(actions).bool().to(device)
            
            # terminate[:,-1,-1] = True
            # Q_n= Q_amax.flatten(1)
            # Q_n[:,0:-1] = Q_n[:,1:]
            # Q_n= Q_n.view(Q_amax.shape)
            
            terminate[:,:,-1] = True
            Q_n= Q_amax
            Q_n[:,:,0:-1]= Q_n[:,:,1:]
            
            # mdn.GaussianMix(Q_a[0,0].unsqueeze(0)).plot_prob_dist()
            
            return Q_a, Q_n, terminate, out_batch_mask
        
def qLoss(out, lf, gamma= 1):
    q_state= out[0]
    next_q= out[1]
    reward= out[2]
    not_terminal= out[3]
    
    target= reward + (gamma * next_q * not_terminal)
    
    mix= mdn.GaussianMix(q_state)
    l = -mix.log_prob(target.detach()).mean()
    
    # return lf(mix.mean[:,0].unsqueeze(1), target)
    return l

if __name__ == '__main__':
    n_classes= 5
    seq_len= 30
    nhidden= 256
    batch_size= 128
    
    lf= nn.HuberLoss()
    # lf= nn.MSELoss()
    buffer= datamanager.ReplayMemory(batch_size * 200)
    
    T= seq_len * torch.ones([batch_size]).long()
    in_cat= [n_classes, 1, 1]
    
    model= RnnQnet(n_classes, nhidden, seq_len, nhidden, 2, in_cat= in_cat).cuda()
    model.load_model()
    
    loss_cat= [model.policy.flatmix_shape, 1, 1, 1]
    
    explorer= strats.Explorer(n_classes, (0.05, 0.9), 0.999)
    rewarder= strats.Rewarder(batch_size, model.policy.in_cat)
    
    model_op= optim.Adam(model.policy.parameters(), lr=0.0001)
    
    try:
        pass
        for _ in range(50000):
            model_op.zero_grad()
            
            x,fpts= datamanager.data_gen(batch_size= batch_size, inst_size= seq_len, class_size= n_classes)
            
            cl_mask= get_class_mask(x[:,:,:n_classes])
            mask= get_mask(T, max_T= seq_len).unsqueeze(-1).tile([1,1,n_classes]) * cl_mask
            with torch.no_grad():
                a, r= model.policy(x,fpts,rewarder.reward, explorer.explore, mask)
                
            buffer.push(qpack(x, a, r))
            x_b, a_b, r_b= qunpack(buffer.sample(batch_size))
            # print(r_b[0])
            
            cl_mask= get_class_mask(x_b[:,:,:n_classes])
            mask= get_mask(T, max_T= seq_len).unsqueeze(-1).tile([1,1,n_classes]) * cl_mask
            
            Qs, Qn, terminate, out_mask= model.policy(x_b, r_b,rewarder.reward, explorer.explore, mask, actions= a_b.long().cuda())
            with torch.no_grad():
                _, Qn, _, _= model.target(x_b, r_b,rewarder.reward, explorer.explore, mask, actions= a_b.long())
            qcat= decat(cat([Qs, Qn, r_b.cuda().unsqueeze(-1), ~terminate.unsqueeze(-1)], dim=-1)[out_mask].flatten(0,1), loss_cat)
            
            loss= qLoss(qcat, lf)
            print(float(loss), float(r_b.mean()), explorer.epsilon)
            loss.backward()
            
            model_op.step()
            explorer.step()
            model.target_copy()
    finally:
        x,fpts= datamanager.data_gen(batch_size= batch_size, inst_size= seq_len, class_size= n_classes)
        T= seq_len * torch.ones([batch_size]).long()
        
        cl, sal, proj= decat(x, in_cat)
        
        r_out, a= model.policy.inf(x, fpts, T, rewarder, explorer)
        r_max= r_out.sum(dim=-1).max(dim=1)[0]
        
        scores= []
        for n in range(x.shape[0]):
            rand_rew= float(datamanager.sample_eval(cl[n], sal[n], fpts[n]))
            print(f"MOD: {float(r_max[n])}, RAND: {rand_rew}")
            scores.append(float(r_max[n]) / rand_rew)
        
        scores= np.array(scores)
        print(scores.mean())
        
        
    
    