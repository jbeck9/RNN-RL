import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy

import datamanager
import strats
import math

NLL_CONST= math.log(math.sqrt(2 * math.pi))

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

class ProbLinear(nn.Module):
    def __init__(self, input_size, nhidden, output_size, epsilon= 0.01):
        super(ProbLinear, self).__init__()
        
        self.out_size= output_size
        self.epsilon= epsilon
        
        # net= [nn.Linear(input_size, nhidden), nn.ReLU()]
        # for _ in range(1):
        #     net.extend([nn.Linear(nhidden, nhidden), nn.ReLU()])
        # self.lin1= nn.Sequential(*net)
        
        net= [nn.Linear(input_size, nhidden), nn.ReLU()]
        for _ in range(1):
            net.extend([nn.Linear(nhidden, nhidden), nn.ReLU()])
        net.extend([nn.Linear(nhidden, output_size)])
        self.lin_mean= nn.Sequential(*net)
        
        net= [nn.Linear(input_size, nhidden), nn.ReLU()]
        for _ in range(1):
            net.extend([nn.Linear(nhidden, nhidden), nn.ReLU()])
        net.extend([nn.Linear(nhidden, output_size), nn.Softplus()])
        self.lin_var= nn.Sequential(*net)
        
    def forward(self, x):
        # out= self.lin1(x)
        
        mean= self.lin_mean(x).unsqueeze(1)
        var= self.lin_var(x).unsqueeze(1) + self.epsilon
        
        return cat([mean, var], dim=1).view(x.shape[0], self.out_size * 2)


class TEncoder(nn.Module):
    def __init__(self, input_size, nhidden, nhead=16, layers=2):
        super(TEncoder, self).__init__()
        
        self.layers= layers
        self.nhidden= nhidden
        self.layers= layers
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=nhidden, nhead=nhead, batch_first= True)
        
        self.model = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.in_linear= nn.Sequential(nn.Linear(input_size, nhidden), 
                                      nn.ReLU(),
                                      nn.Linear(nhidden, nhidden))
        
        
    def forward(self, x, mask):
        device= get_device(x)
        
        x= input_padded(x, self.in_linear, mask)
        return self.model(x, src_key_padding_mask= ~mask.to(device))

@doubleq
class RnnQnet(nn.Module):
    def __init__(self, nclasses, enc_size, seq_len, nhidden, layers, in_cat= None, padval= -2, lin_layers= 2):
        super().__init__()
        
        self.layers= layers
        self.nhidden= nhidden
        self.enc_size= enc_size
        self.nclasses= nclasses
        self.input_size= nclasses + 2
        self.padval= padval
        self.seqlen= seq_len
        
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
        
        # net= [nn.Linear(nhidden, nhidden), nn.LeakyReLU(0.2)]
        # for _ in range(1):
        #     net.extend([nn.Linear(nhidden, nhidden), nn.LeakyReLU(0.2)])
        # net.extend([nn.Linear(nhidden, nhidden), ProbLinear(nhidden, nhidden, 1)])
        self.emb= ProbLinear(nhidden + nhidden, nhidden, 1)
        
    def inf(self, x, fpts, T, rewarder, explorer):
        explorer.epsilon= 0
        self.eval()
        
        cl, sal, proj= decat(x, self.in_cat)
        
        cl_mask= get_class_mask(x[:,:,:n_classes])
        mask= get_mask(T, max_T= seq_len).unsqueeze(-1).tile([1,1,n_classes]) * cl_mask
        with torch.no_grad():
            a, r= model.policy(x,fpts,rewarder.reward, explorer.explore, mask)
            a= a.long()
        
        return r[:,-1]
        
    def forward(self, x,y,r_fn, e_fn , mask, device= 'cuda', actions= None):
        x= x.to(device)
        bs= x.shape[0]
        ind= torch.Tensor(range(self.seqlen))
        hidden= torch.zeros([self.layers, bs, self.nhidden]).to(device)
        if actions is None:
            inf= True
            actions= self.padval * torch.ones([bs, self.nclasses]).long()
            rewards= self.padval * torch.ones([bs, self.nclasses])
        
        else:
            inf= False
            Q_a= self.padval * torch.ones([bs, self.nclasses, 2]).to(device)
            Q_amax= self.padval * torch.ones([bs, self.nclasses, 1]).to(device)
        
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
                    a= actions[n, p]
                    if len(binds) > 0:
                        q= Q[n, bm]
                        exp_bool, aexp= e_fn(binds, q)
                        if exp_bool:
                            a= aexp
                        else:
                            if self.training:
                                q_sample= torch.distributions.Normal(q[:,0], q[:,1]).sample()
                                a= int(binds[q_sample.argmax()])
                            else:
                                a= int(binds[q[:,0].argmax()])
                        sel_mask[n, a] = False
                    actions[n, p] = a
                else:
                    a= actions[n, p]
                    if a >= 0:
                        sel_mask[n, a] = False
                        amax= int(binds[Q[n, bm][:,0].argmax()])
                        Q_a[n,p]= Q[n, a]
                        Q_amax[n,p]= Q[n, amax][0]
                        
                mem_i[n] = z[n, a]
            _, hidden = self.S_rnn(mem_i.unsqueeze(1), hidden)
        
        if inf:
            for n in range(bs):
                rewards[n]= r_fn(actions[n], x[n], y[n])
            
            return actions, rewards
        
        else:
            out_batch_mask= (actions >= 0).all(dim=-1)
            terminate= torch.zeros_like(actions).bool().to(device)
            terminate[:,-1] = True
            Q_amax[:,0:-1] = Q_amax[:,1:]
            
            return Q_a, Q_amax, terminate, out_batch_mask
        
def qLoss(out, lf, gamma= 1):
    q_state= out[0]
    next_q= out[1]
    reward= out[2]
    not_terminal= out[3]
    
    q_mean= q_state[:,0].unsqueeze(-1)
    q_var= q_state[:,1].unsqueeze(-1)
    
    # print(next_q)
    
    target= reward + (gamma * next_q * not_terminal)
    
    l= ((q_mean - target) ** 2) / (2 * q_var) + torch.log(q_var)# + NLL_CONST
    
    # print(q_mean)
    # print(reward)
    return l.mean()
    # return lf(q_mean, target)

if __name__ == '__main__':
    n_classes= 5
    seq_len= 30
    nhidden= 256
    batch_size= 128
    
    lf= nn.HuberLoss()
    # lf= nn.MSELoss()
    buffer= datamanager.ReplayMemory(batch_size * 100)
    
    T= seq_len * torch.ones([batch_size]).long()
    in_cat= [n_classes, 1, 1]
    
    loss_cat= [2, 1, 1, 1]
    
    model= RnnQnet(n_classes, nhidden, seq_len, nhidden, 2, in_cat= in_cat).cuda()
    model.load_model()
    
    explorer= strats.Explorer(n_classes, (0.02, 0.9), 0.999)
    rewarder= strats.Rewarder(batch_size, model.policy.in_cat)
    
    model_op= optim.Adam(model.policy.parameters(), lr=0.0001)
    
    try:
        for _ in range(10000):
            model_op.zero_grad()
            
            x,fpts= datamanager.data_gen(batch_size= batch_size, inst_size= seq_len, class_size= n_classes)
            
            cl_mask= get_class_mask(x[:,:,:n_classes])
            mask= get_mask(T, max_T= seq_len).unsqueeze(-1).tile([1,1,n_classes]) * cl_mask
            with torch.no_grad():
                a, r= model.policy(x,fpts,rewarder.reward, explorer.explore, mask)
                
            buffer.push(qpack(x, a, r))
            x_b, a_b, r_b= qunpack(buffer.sample(batch_size))
            
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
        
        r_out= model.policy.inf(x, fpts, T, rewarder, explorer)
        
        for n in range(x.shape[0]):
            rand_rew= float(datamanager.sample_eval(cl[n], sal[n], fpts[n]))
            print(f"MOD: {float(r_out[n])}, RAND: {rand_rew}")
        
        
    
    