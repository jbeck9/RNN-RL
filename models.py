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
    
def get_total_valid(current, hist):
    hist= hist.sum(dim=1) > 0
    hist[:,-1] = False
    return current * ~hist

def vargmax_sel(x, valid):
    out= torch.zeros(x.shape)
    for n in range(len(x)):
        m= valid[n].bool()
        ind= torch.Tensor(range(x.shape[-1]))[m][x[n][m].argmax()].long()
        
        out[n, ind] = 1
        
    return out

def argmax_sel(x):
    out= torch.zeros(x.shape)
    for n in range(len(x)):
        ind= x[n].argmax()
        
        out[n, ind] = 1
        
    return out

def qpack(x, o, T):
    return [(x[n],o[n], T[n]) for n in range(len(T))]

def get_terminate(T, length):
    
    out= torch.zeros([T.shape[0], length])
    
    for n in range(len(T)):
        out[n, T[n]-1] = 1
        
    return out.bool()
    

def qunpack(q):
    x= []
    o= []
    T= []
    
    for n in q:
        x.append(n[0].unsqueeze(0))
        o.append(n[1].unsqueeze(0))
        T.append(n[2].unsqueeze(0))
        
    return cat(x, dim=0),cat(o, dim=0), cat(T, dim=0)

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
    def __init__(self, input_size, nhidden, output_size, epsilon= 0.02):
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
        self.layers= layers
        self.enc_size= enc_size
        self.nclasses= nclasses
        self.input_size= nclasses + 3
        self.padval= padval
        self.seqlen= seq_len
        
        self.in_cat= in_cat
        
        self.out_cat= [self.nclasses, 1]
        
        
        self.hidden= None
        
        self.enc= TEncoder(self.input_size, enc_size)
        
        self.S_rnn = nn.GRU(
            input_size= nhidden,
            hidden_size= nhidden, 
            num_layers=layers, 
            batch_first=True
        )
        
        net= [nn.Linear(self.nclasses + nhidden, nhidden), nn.LeakyReLU(0.2)]
        for _ in range(1):
            net.extend([nn.Linear(nhidden, nhidden), nn.LeakyReLU(0.2)])
        net.extend([nn.Linear(nhidden, nhidden)])
        self.emb= nn.Sequential(*net)
        
        # self.attn = nn.Sequential(nn.Linear(nhidden + nhidden, nhidden),
        #                           nn.ReLU(),
        #                           nn.Linear(nhidden, seq_len))
        
        # self.attn_combine = nn.Sequential(nn.Linear(nhidden + enc_size, nhidden),
        #                           nn.ReLU(),
        #                           nn.Linear(nhidden, nhidden))
        
        
        
        self.out_linear= ProbLinear(nhidden, nhidden, nclasses)
    
    def reset_hidden(self):
        self.hidden= None
    
    def update_hidden(self, inp, z, mask):
        
        inp= inp[mask]
        emb_inp= self.emb(inp)
        hidden= self.hidden[:,mask]
        z= z[mask]
        
        attn_weights = F.softmax(self.attn(cat([emb_inp, hidden[-1]], dim=-1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),z)
        out = torch.cat((emb_inp, attn_applied[:,0]), 1)
        out = self.attn_combine(out).unsqueeze(1)
        _, hidden = self.S_rnn(out, hidden)
        
        self.hidden[:,mask] = hidden
        
    def forward(self, x, a, T_in, device= 'cuda'):
        self.reset_hidden()
        
        x= x.to(device)
        a= a.to(device)
        pad_mask= get_mask(T_in, max_T= self.seqlen)
        iter_mask= get_mask(T_in - 1, max_T= self.seqlen)
        
        terminate= get_terminate(T_in, self.seqlen).unsqueeze(-1).to(device)
        bs= x.shape[0]
        
        Qs= self.padval * torch.ones([bs, self.seqlen, self.nclasses * 2]).to(device)
        Qn= self.padval * torch.ones([bs, self.seqlen, self.nclasses]).to(device)
        
        z= self.enc(x, torch.ones_like(pad_mask).bool())
        
        if self.hidden is None:
            self.hidden= torch.zeros([self.layers, bs, self.nhidden]).to(device)
        
        mask= torch.ones([bs]).bool().to(device)
        
        i= torch.cat([torch.zeros([bs, self.nclasses]).to(device), z[:,0]], dim=-1).unsqueeze(1)
        _, self.hidden = self.S_rnn(self.emb(i), self.hidden)
        qs= self.out_linear(self.hidden[-1][mask])
        Qs[mask,0] = qs
        for n in range(self.seqlen - 1):
            mask= iter_mask[:,n]
            
            if torch.all(~mask):
                break
            
            i= torch.cat([a[:,n].to(device)[mask], z[mask, n + 1]], dim=-1).unsqueeze(1)
            _, self.hidden[:,mask] = self.S_rnn(self.emb(i), self.hidden[:,mask])
            q= self.out_linear(self.hidden[-1][mask])
            
            Qn_dist= q.view(mask.sum(), 2, -1)
            # Qn[mask, n]= torch.distributions.Normal(Qn_dist[:,0], torch.sqrt(Qn_dist[:,1])).sample()
            
            Qn[mask,n] = Qn_dist[:,0]
            Qs[mask, n+1] = q
        
        return Qs, Qn, terminate, pad_mask
            
        
    def inf(self, x,y,r_fn, e_fn ,T_in , all_valid_actions= None, device= 'cuda'):
        self.reset_hidden()
        x= x.to(device)
        
        pad_mask= get_mask(T_in, max_T= x.shape[1])
        
        bs= x.shape[0]
            
        if self.hidden is None:
            self.hidden= torch.zeros([self.layers, bs, self.nhidden]).to(device)
        
        history= torch.zeros([bs,self.seqlen, self.nclasses])
        rf_out= self.padval * torch.ones([bs, self.seqlen, sum(self.out_cat)])
        
        action= torch.zeros([bs, self.nclasses])
        mask= torch.ones([bs]).bool()
        T_out= copy.deepcopy(T_in)
        
        with torch.no_grad():
            z= self.enc(x, pad_mask)
            i= torch.cat([action.to(device), z[mask,0]], dim=-1).unsqueeze(1)
            _, self.hidden[:,mask] = self.S_rnn(self.emb(i), self.hidden)
        
        for n in range(self.seqlen):
            if all_valid_actions is None:
                valid_actions= torch.ones([bs, self.nclasses])
            else:
                valid_actions= all_valid_actions[:,n]
            
            total_permit= get_total_valid(valid_actions, history[:, :n+1])
            # total_permit= torch.ones_like(total_permit)
            explore_choice, explore_mask= e_fn(total_permit[mask])
            
            
            action= self.padval * torch.ones_like(action)
            sub_action= action[mask]
            with torch.no_grad():
                pred_out= self.out_linear(self.hidden[-1][mask]).view(-1, 2, action.shape[-1])
                pred= pred_out[:,0]
                pred_var= pred_out[:,1]
                if self.training:
                    pred= torch.distributions.Normal(pred, torch.sqrt(pred_var)).sample()
                # print(pred[0])
                # print(pred_var[0])
                act_out= vargmax_sel(pred[~explore_mask], torch.ones_like(total_permit)[mask][~explore_mask])
                sub_action[~explore_mask] = act_out
                
                sub_action[explore_mask] = argmax_sel(pred_var[explore_mask])
            
            # if n == 10:
            #     print(pred[0])
            #     print(pred_var[0])
            sub_action[explore_mask] = explore_choice
            action[mask] = sub_action
            
            reward= self.padval * torch.ones([bs])
            terminate= torch.zeros([bs]).bool()
            reward_out, terminate_out= r_fn(action[mask], history[:,:n+1][mask], x[:,:n+1][mask], y[:,:n+1][mask])
            
            reward[mask]= reward_out
            terminate[mask] = terminate_out
            
            T_out[terminate] = n+1
            
            mask= T_out > n+1
            rf_out[:,n] = cat([action.to(device),reward.unsqueeze(1).to(device)], dim=-1)
            
            if torch.all(~mask):
                rtot= rf_out[:,:,6]
                print(rtot.max(dim=1)[0].mean())
                # print(r_tot[0][r_tot[0] > -2].sum())
                return rf_out.detach(), T_out
            else:
                history[:, n] = action
                i= torch.cat([action.to(device)[mask], z[mask, n + 1]], dim=-1).unsqueeze(1)
                with torch.no_grad():
                    _, self.hidden[:,mask] = self.S_rnn(self.emb(i), self.hidden[:,mask])
        
def qLoss(out, lf, gamma= 1):
    q_state= out[0]
    action= out[1]
    next_q= out[2]
    reward= out[3]
    not_terminal= out[4]
    
    q_state= q_state.view(q_state.shape[0], 2, -1)
    
    action_amax= torch.argmax(action, dim=-1).unsqueeze(1)
    q_sel_m= torch.gather(q_state[:,0], -1, action_amax)
    q_sel_v= torch.gather(q_state[:,1], -1, action_amax)
    
    # q_sel= torch.distributions.Normal(q_sel_m, 0.05 + torch.sqrt(q_sel_v))
    
    next_q_sel= next_q.max(dim=-1)[0].unsqueeze(1).detach()
    
    target= reward + (gamma * next_q_sel * not_terminal)
    
    l= ((q_sel_m - target) ** 2) / (2 * q_sel_v) + torch.log(q_sel_v)# + NLL_CONST
    
    return l.mean()
    # return lf(q_sel_m, target)

if __name__ == '__main__':
    n_classes= 6
    seq_len= 30
    nhidden= 256
    batch_size= 128
    
    lf= nn.HuberLoss()
    # lf= nn.MSELoss()
    buffer= datamanager.ReplayMemory(batch_size * 400)
    
    T= seq_len * torch.ones([batch_size]).long()
    in_cat= [n_classes, 1, 1, 1]
    
    loss_cat= [n_classes*2, n_classes, n_classes, 1, 1]
    
    model= RnnQnet(n_classes, nhidden, seq_len, nhidden, 1, in_cat= in_cat).cuda()
    model.load_model()
    
    explorer= strats.Explorer(n_classes, (0.01, 0.3), 0.999)
    rewarder= strats.Rewarder(batch_size, model.policy.in_cat)
    
    model_op= optim.Adam(model.policy.parameters(), lr=0.0001)
    
    try:
        for _ in range(100000):
            model_op.zero_grad()
            
            x,fpts= datamanager.data_gen(batch_size= batch_size, inst_size= seq_len, class_size= n_classes)
            infout, T_out= model.policy.inf(x,fpts,rewarder.reward, explorer.explore, T, all_valid_actions= x[:,:,:n_classes])
            
            buffer.push(qpack(x, infout, T_out))
            
            X_b, O_b, T_b= qunpack(buffer.sample(batch_size))
            a_b, r_b= decat(O_b, model.policy.out_cat, device= 'cuda')
            Qs, Qn, terminate, pad_mask= model.policy(X_b, a_b, T_b)
            with torch.no_grad():
                _, Qn, _, _= model.target(X_b, a_b, T_b)
            qcat= decat(cat([Qs, a_b, Qn, r_b, ~terminate], dim=-1)[pad_mask], loss_cat)
            
            loss= qLoss(qcat, lf)
            print(float(loss), explorer.epsilon)
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            model_op.step()
            explorer.step()
            
            model.target_copy()
    finally:
        model= model.to('cpu')
        model.eval()
        with torch.no_grad():
            explorer.epsilon= 0
            
            x_in,y= datamanager.data_gen(batch_size= batch_size, inst_size= seq_len, class_size= n_classes)
            x, last, sal, proj= decat(x_in, in_cat)

            model.policy.reset_hidden()
            infout, T_out= model.policy.inf(x_in,y,rewarder.reward, explorer.explore, T, all_valid_actions= x_in[:,:,:n_classes], device= 'cpu')
            a, r= decat(infout, model.policy.out_cat)
            
            modind= a[:,:,:-1].sum(dim=-1) == 1
            
            for n in range(batch_size):
                mod_rew= float(y[n,modind[n]].sum())
                rand_rew= float(datamanager.sample_eval(x[n,:,:-1], sal[n], y[n]))
                
                print(f"RAND: {rand_rew}, MODEL: {mod_rew}")
        
        
    
    