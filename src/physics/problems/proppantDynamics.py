import torch

from phys import *
from utils import derivative_batched, smooth_clamp

class proppantDynamics_dless(Physics):
    """
    Undimensionalised statement for proppant dynamics in opened fracture
    
    variables:
        1. c - proppant concentration, c in [0, 1)
        2. p - domain pressure
    
    PDE:
        dc/dt + w^2 / mu dp/dx dc/dx + alpha^2 w^2 / mu (dp/dy - (1 + rc) G) dc/dy = 0 
        d/dx (w^3 / mu dp/dx) + d/dy (w^3/mu (dp/dy - (1 + rc) G)) = 0
        mu = (1 - c)^beta
    
    boundary conditions:
        1. inlet: c = c_in(t), u = u_in / dp/dn = -mu/w^2 (assume it be horizontal boundary)
        2. outlet: ???
        3. slip: u = 0 -> dp/dn = alpha (1 + rc) G e_y
    initial condition: c = 0
    
    parameters:
        w: geometry of fracture, w in (0, 1]
        alpha: aspect ratio, alpha = L/H
        r = c_max (rho_proppant - rho_fluid) / rho_fluid
        G = rho_fluid g H p_0
    
        There L and H are horizontal and vertical sizes of the domain, 
        c_max - maximum proppant concentration,
        rho_proppant and rho_fluid are corresponding densities,
        g - acceleration of gravity,
        p_0 - dimensionless coefficient for pressure
    """
    
    def __init__(self, device="cpu", dim=2, has_time=True):
        super().__init__(device, dim, has_time)
    
    def setParameters(self, params: dict, boundaries: dict, initial: dict = None):
        required = {"alpha", "beta", "dzeta", "kappa", "w", "mu", "r", "G", "c_k", "t_k"}
        missing  = required - params.keys()
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        self.par        = params
        self.boundaries = {
            name: bound["bc"]
            for name, bound in boundaries.items()
            if "bc" in bound
        }
        self.initial    = initial or {
            "c": lambda x, y: torch.zeros_like(x)
        }
        self.transforms = {
            "c": lambda c: smooth_clamp(c, lo=0.0, hi=1.0, eps=1e-16),
        }

    def residualPDE(self, net, coords):
        
        t, y, x = coords[:,0:1], coords[:,1:2], coords[:,2:3]
        
        pred = net(coords, self.transform)
    
        c   = pred[:,0:1]
        p_x = pred[:,1:2]
        p_y = pred[:,2:3]
        
        width  = self.par["w"](x, y)
        mu = self.par["mu"](c)
        gravity = (1 + self.par["r"] * c) * self.par["G"]

        mobility = width**2 / mu

        ux = -mobility * p_x
        uy = -mobility * (p_y - gravity) * self.par["alpha"]

        c_x = derivative_batched(c, x) * ux
        c_y = derivative_batched(c, y) * uy * self.par["alpha"]
        c_t = derivative_batched(c, t)

        ux_x = derivative_batched(width * ux, x)
        uy_y = derivative_batched(width * uy, y) * self.par["alpha"]
        
        convection = c_t + c_x + c_y
        
        poisson = ux_x + uy_y
        correlation = derivative_batched(p_x, y) - derivative_batched(p_y, x)
        
        return {
            "convection": convection,
            "poisson": poisson,
            "correlation": correlation
        }