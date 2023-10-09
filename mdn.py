import torch
import torch.nn as nn

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import copy
import math

LL_CONST= math.log(math.sqrt(2 * math.pi))
cat= torch.cat

def choice(w, nsamples):
    w_tot= torch.zeros(w.shape)
    for n in range(w.shape[1]):
        w_tot[:,n] = w[:,:n+1].sum(dim=-1)
        
    w_tot= w_tot.unsqueeze(1).tile([1, nsamples, 1])
    r= torch.rand([w_tot.shape[0],w_tot.shape[1], 1])
    return (r < w_tot).long().argmax(dim=-1)

class MdnLinear(nn.Module):
    def __init__(self, input_size, nhidden, nmix, epsilon= 0.001):
        super(MdnLinear, self).__init__()
        
        self.nmix= nmix
        self.epsilon= epsilon
        
        net= [nn.Linear(input_size, nhidden), nn.LeakyReLU(0.1)]
        for _ in range(1):
            net.extend([nn.Linear(nhidden, nhidden), nn.LeakyReLU(0.1), nn.Dropout(0.1)])
        net.extend([nn.Linear(nhidden, nmix), nn.Softmax(dim=-1)])
        self.lin_pi= nn.Sequential(*net)
        
        net= [nn.Linear(input_size, nhidden), nn.LeakyReLU(0.1)]
        for _ in range(1):
            net.extend([nn.Linear(nhidden, nhidden), nn.LeakyReLU(0.1), nn.Dropout(0.1)])
        net.extend([nn.Linear(nhidden, nmix)])
        self.lin_mean= nn.Sequential(*net)
        
        net= [nn.Linear(input_size, nhidden), nn.LeakyReLU(0.1)]
        for _ in range(1):
            net.extend([nn.Linear(nhidden, nhidden), nn.LeakyReLU(0.1), nn.Dropout(0.1)])
        net.extend([nn.Linear(nhidden, nmix), nn.Softplus()])
        self.lin_var= nn.Sequential(*net)
        
    def forward(self, x):
        # out= self.lin1(x)
        
        pi= self.lin_pi(x).unsqueeze(1)
        mean= self.lin_mean(x).unsqueeze(1)
        var= self.lin_var(x).unsqueeze(1) + self.epsilon
        
        return cat([pi, mean, var], dim=1).permute(0,2,1).flatten(1)

class GaussianMix():
    def __init__(self, inp):
        self.ogshape= list(inp.shape)
        self.bs= inp.flatten(0, -2).shape[0]
        
        i= inp.view(self.bs, -1, 3)
        self.N= i.shape[1]
        
        self.pi= i[:,:,0]
        self.mean= i[:,:,1]
        self.var= i[:,:,2]
        
        self.device= inp.device
        
    def reshape_output(self, out, feature_size):
        outshape= copy.deepcopy(self.ogshape)
        outshape[-1] = feature_size
        return out.view(outshape)
        
    def sample(self, nsamples= 1):
        std= self.var.sqrt()
        
        ind= choice(self.pi, nsamples).to(self.device)
        
        m= torch.gather(self.mean, 1, ind)
        s= torch.gather(std, 1, ind)
        out= torch.normal(m, s)
        return self.reshape_output(out, nsamples)
    
    def cdf(self, target):
        std= self.var.sqrt()
        cdf= (self.pi * (0.5 * (1 + torch.erf((target - self.mean) * std.reciprocal() / math.sqrt(2))))).sum(dim=1)
            
        return self.reshape_output(cdf, 1)
    
    def expectation(self):
        out= (self.pi * self.mean).sum(dim=1)
        return self.reshape_output(out, 1)
    
    def log_prob(self, target):
        
        pi= self.pi
        mean= self.mean
        var= self.var
        
        if target.dim() > 2:
            pi= pi.unsqueeze(-1).tile([1,1,target.shape[-1]])
            mean= mean.unsqueeze(-1).tile([1,1,target.shape[-1]])
            var= var.unsqueeze(-1).tile([1,1,target.shape[-1]])

        log_probs = torch.log(pi) -((target - mean) ** 2) / (2 * var) - torch.log(var) - LL_CONST
        return torch.logsumexp(log_probs, dim=1)
    
    def plot_sample_dist(self, nsamples):
        s= self.sample(nsamples).detach().cpu()
        
        plt.clf()
        plt.xlabel("Value")
        for n,x in enumerate(s.detach().cpu()):
            sns.kdeplot(x, label= n)
        plt.legend()
        plt.pause(0.01)
        
    def plot_prob_dist(self, fval= 0.1):
        r= torch.arange(-10,10, 0.01).unsqueeze(0).unsqueeze(0).tile([self.bs,self.N, 1])
        p= torch.exp(self.log_prob(r.to(self.mean.device)).detach()).cpu()
        
        plt.clf()
        plt.xlabel("Value")
        for n,x in enumerate(p):
            plt.plot(r[0,0][x>fval], x[x>fval], label= n)
        plt.legend()
        plt.pause(0.01)


if __name__ == '__main__':
    i= torch.Tensor([[ 1.6364e-01,  2.7257e+00,  2.2509e-03,  1.5924e-01,  3.1759e+00,
              2.2264e-03,  3.3602e-01,  2.9156e+00,  1.0050e-03,  3.4109e-01,
              3.0282e+00,  1.0000e-03],
            [ 1.5913e-01,  1.9884e+00,  1.7839e-03,  1.5443e-01,  2.3958e+00,
              1.8924e-03,  3.4034e-01,  2.1611e+00,  1.0029e-03,  3.4609e-01,
              2.2662e+00,  1.0000e-03],
            [ 1.4006e-01,  1.5008e+00,  2.1541e-03,  1.8421e-01,  1.8993e+00,
              1.7729e-03,  3.2540e-01,  1.6578e+00,  1.0102e-03,  3.5033e-01,
              1.7661e+00,  1.0001e-03],
            [ 1.3393e-01,  8.7731e-01,  2.0029e-03,  1.9184e-01,  1.3353e+00,
              1.7276e-03,  3.1228e-01,  1.0800e+00,  1.0162e-03,  3.6195e-01,
              1.1948e+00,  1.0000e-03],
            [ 1.1248e-02, -2.0128e+00,  1.0010e-03,  3.0186e-01,  5.2295e-01,
              1.5919e-03,  2.4884e-01,  2.2605e-01,  2.1466e-03,  4.3805e-01,
              3.8152e-01,  1.0005e-03]])
              
    
    gm= GaussianMix(i)
    
    # out= gm.log_prob(torch.Tensor([3]).unsqueeze(0))
    # print(torch.exp(out))
    # out= gm.sample(1000)
    out= gm.plot_prob_dist()
    
    
    
    # model= MdnLinear(6, 256, 4)
    # i= torch.rand([256, 6])
    # out= GaussianMix(model(i))
    
    # print(out.pi)

