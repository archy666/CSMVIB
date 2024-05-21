import torch
import random
import numpy as np


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)  # Python built-in random number generator
    np.random.seed(seed)  # Numpy's random number generator
    torch.manual_seed(seed)  # PyTorch's random number generator
    torch.cuda.manual_seed(seed)  # Random number generator for GPU when used
    torch.backends.cudnn.deterministic = True  # Ensure reproducibility, but may decrease speed
    torch.backends.cudnn.benchmark = False  # Turn off automatic algorithm search for stability in experiments


def distmat(X):
    """ distance matrix
    """

    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D) 
    return D

def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def GaussianMatrix(X,Y,sigma):
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    size1 = X.size()
    size2 = Y.size()
    G = (X*X).sum(-1)
    H = (Y*Y).sum(-1)
    Q = G.unsqueeze(-1).repeat(1,size2[0])
    R = H.unsqueeze(-1).T.repeat(size1[0],1)

    H = Q + R - 2*X@(Y.T)
    H = torch.clamp(torch.exp(-H/2/sigma**2),min=0)
    
    
    return H

def CS_QMI(x,y,sigma = None):
    """
    x: NxD
    y: NxD
    Kx: NxN
    ky: NxN
    """
    
    N = x.shape[0]
    #print(N)
    if not sigma:
        sigma_x = 10*sigma_estimation(x,x)
        sigma_y = 10*sigma_estimation(y,y)
       
        Kx = GaussianMatrix(x,x,sigma_x)
        Ky = GaussianMatrix(y,y,sigma_y)
    
    else:
        Kx = GaussianMatrix(x,x,sigma)
        Ky = GaussianMatrix(y,y,sigma)
    
    #first term
    self_term1 = torch.trace(Kx@Ky.T)/(N**2)
    
    #second term  
    self_term2 = (torch.sum(Kx)*torch.sum(Ky))/(N**4)
    
    #third term
    term_a = torch.ones(1,N).to(x.device)
    term_b = torch.ones(N,1).to(x.device)
    cross_term = (term_a@Kx.T@Ky@term_b)/(N**3)
    CS_QMI = -2*torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)
    
    return CS_QMI

def CS_QMI_normalized(x,y,sigma):

    QMI = CS_QMI(x, y, sigma)
    var1 = torch.sqrt(CS_QMI(x, x, sigma))
    var2 = torch.sqrt(CS_QMI(y, y, sigma))
    
    return QMI/(var1*var2)

def CS_Div(x,y1,y2,sigma): # conditional cs divergence Eq.18
    K = GaussianMatrix(x,x,sigma)
    L1 = GaussianMatrix(y1,y1,sigma)
    L2 = GaussianMatrix(y2,y2,sigma)
    L21 = GaussianMatrix(y2,y1,sigma)

    H1 = K*L1
    self_term1 = (H1.sum(-1)/(K**2).sum(-1)).sum(0)
    
    H2 = K*L2
    self_term2 = (H2.sum(-1)/(K**2).sum(-1)).sum(0)
    
    H3 = K*L21 
    cross_term = (H3.sum(-1)/(K**2).sum(-1)).sum(0)
    
    return -2*torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)