import torch
from scipy.optimize import minimize_scalar
import numpy as np

def get_linear_params(model):
    para_params = [p for p in model.paraphrase_linear.parameters()] + [p for p in model.paraphrase_linear_interact.parameters()]
    sim_params = [p for p in model.similarity_linear.parameters()]
    return para_params + sim_params

def get_1d_grads(params):
    grads = []
    for param in params:
        grads.append(torch.zeros_like(param.data).view(-1) if param.grad is None else param.grad.view(-1))
    grads = torch.cat(grads)
    return grads

def apply_1d_grad(grad, params):
    index = 0
    for param in params:
        total_size = param.nelement()
        param.grad = grad[index:total_size].view(param.shape)
        index += total_size

def cagrad(g1, g2, c=0.5):
    # mostly taken from https://github.com/Cranial-XIX/CAGrad/blob/main/toy.py
    g0 = (g1 + g2) / 2
    
    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()
    
    g0_norm = 0.5 * np.sqrt(g11+g22+2*g12+1e-4)
    coef = c * g0_norm

    def obj(x):
        return coef * np.sqrt(x**2*(g11+g22-2*g12)+2*x*(g12-g22)+g22+1e-4) + \
                0.5*x*(g11+g22-2*g12)+(0.5+x)*(g12-g22)+g22
    
    res = minimize_scalar(obj, bounds=(0,1), method='bounded')
    x = res.x

    gw = x * g1 + (1-x) * g2
    gw_norm = np.sqrt(x**2*g11+(1-x)**2*g22+2*x*(1-x)*g12+1e-4)

    lmbda = coef / (gw_norm+1e-4)
    g = g0 + lmbda * gw
    return g / (1+c)