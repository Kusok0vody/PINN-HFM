import torch
import time
import json
import io
import os

from torch.autograd import Variable
from texttable import Texttable
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import programs.conditions as cnd
import programs.misc as misc
import programs.objects as obj 
from programs.NN import *



class NumericalDataset(Dataset):
    def __init__(self, Nx, Ny, Nt, T, path, dtype):
        super().__init__()
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        
        self.dtype  = dtype
        self.path   = path
        
        self.NtNew = int(Nt * 0.6)
        
        x = torch.linspace(0, 1, Nx, dtype=torch.float32)
        y = torch.linspace(0, 1, Ny, dtype=torch.float32)
        t = torch.linspace(0, T*0.6, self.NtNew, dtype=torch.float32)

        coords = torch.stack(torch.meshgrid(t, y, x, indexing='ij')).reshape(3, -1)
        self.t = coords[0]
        self.y = coords[1]
        self.x = coords[2]
        
        self.data = {"p": [], "c": [], "u_x": [], "u_y": []}
        self.loadToMemory()

    def read_data(self, valType, path):
        path_ = os.path.join(path, valType + ".bin")        
        with open(path_, "rb") as f:
            res =  np.frombuffer(f.read(), dtype=self.dtype).reshape(self.Nt, self.Ny, self.Nx)[:self.NtNew].reshape(-1)
        return torch.as_tensor(res, dtype=torch.float32)

    def loadToMemory(self):
        # types = ("p", "c", "u_x", "u_y")
        # types = ("p", "c")
        self.data["p"] = self.read_data("p", self.path)
        self.data["c"] = self.read_data("c", self.path) / 0.65
            

    def clear(self):
        self.data = {"p": [], "c": [], "u_x": [], "u_y": []}
    
    def __len__(self):
        return self.NtNew * self.Nx * self.Ny
    
    def __getitem__(self, idx):
        return {
            "t": self.t[idx],
            "x": self.x[idx],
            "y": self.y[idx],
            "c": self.data["c"][idx],
        }

class CoordsDataset(Dataset):
    def __init__(self, size, ranges, mode):
        self.size = size
        self.mode = mode

        if self.mode == "L":
            self.t, self.y, self.x = self.generateLinearPoints(ranges)
        elif self.mode == "R":
            self.t, self.y, self.x = self.generateRandomPoints(ranges)
        elif self.mode == "E":
            self.t = torch.Tensor([])
            self.y = torch.Tensor([])
            self.x = torch.Tensor([])
       
    def generateLinearPoints(self, ranges):
        grids = [torch.linspace(r[0], r[1], self.size) if r[0] - r[1] !=0 else torch.Tensor([r[0]]) for r in ranges]
        return torch.stack(torch.meshgrid(*grids, indexing='ij')).reshape(len(ranges), -1)

    def generateRandomPoints(self, ranges):
        size = 1
        for r in ranges:
             size *= self.size if (r[0] != r[1]) else 1
        grids = [torch.Tensor(size).uniform_(r[0], r[1]) if r[0] - r[1] !=0 else r[0]*torch.ones(size) for r in ranges]
        return torch.stack(grids).reshape(len(ranges), -1)
    
    def __add__(self, other):
        if isinstance(other, CoordsDataset):
            self.t = torch.cat((self.t, other.t))
            self.y = torch.cat((self.y, other.y))
            self.x = torch.cat((self.x, other.x))
            if self.mode != other.mode: self.mode = "M"
        elif isinstance(other, dict):
            self.t = torch.cat((self.t, other["t"]))
            self.y = torch.cat((self.y, other["y"]))
            self.x = torch.cat((self.x, other["x"]))
        
        return self
            
    def __iadd__(self, other):
        self.t = torch.cat((self.t, other.t))
        self.y = torch.cat((self.y, other.y))
        self.x = torch.cat((self.x, other.x))
        
        if self.mode != other.mode:
            self.mode = "M"
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "t": self.t[idx],
            "y": self.y[idx],
            "x": self.x[idx],
        }


class Poisson_Convection:
    def __init__(self,
                 data : dict,
                 device : str = 0,
                 net : Net = 0,
                ):

        # CPU/GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device==0 else device
        # self.device='cpu'
        # torch.set_default_device(self.device)

        # Derivative debug
        torch.autograd.set_detect_anomaly(False)
        
        min_data = {'alpha'  : float,
                    'zeta'   : float,
                    'kappa'  : float,
                    'times'  : list,
                    't_max'  : float,
                    'c_cond' : list,
                    'N_PDE'  : int,
                    'N_IC'   : int,
                    'N_BC'   : int,
        } 
        
        for i in min_data.keys():
            if not(i in data):
                raise ValueError(f'Value \"{i}\" not found')
            if type(min_data[i])==int and not(type(data[i])==int):
                raise ValueError(f'Value {data[i]} in {i} must be integer')

        self.updateData(data)

        # Make a model
        if net==0:
            self.model = Net(input_size  = self.NN_params.get('input_size'), # type: ignore
                             neurons_arr = self.NN_params.get('neurons_arr'),
                             output_size = self.NN_params.get('output_size'),
                             depth       = self.NN_params.get('depth'),
                             act         = self.NN_params.get('act'),
                            ).to(self.device)
        else:
            self.model = net.to(self.device)
        
        # Make first arrays of IC, BC
        self.makeDistributedPoints()
    
    def updateData(self, data:dict):
        # Training parameters
        self.criterion   = torch.nn.MSELoss()
        self.Adam_epochs = data.setdefault('Adam_epochs', 1000)
        self.lr          = data.setdefault('lr', 1e-5)
        self.epoch       = data.setdefault('epoch', 0)
        self.k           = data.setdefault('k', 10)
        self.max_epoch   = data.setdefault('max_epoch', 50000)
        self.max_iter    = data.setdefault('max_iter',  2)
        self.save_after  = data.setdefault('save_after',  True)
        self.path        = data.setdefault('path',  '')
        self.start       = time.time()
        self.end         = time.time()

        # Weights
        self.weights      = data.setdefault('weights', [1,1,1,1,1,1,1,1])
        
        # Loss arrays
        self.losses = data.setdefault('losses', [])
        self.PDE    = data.setdefault('PDE',    [])
        self.BC     = data.setdefault('BC',     [])
        self.IC     = data.setdefault('IC',     [])
        self.corr   = data.setdefault('corr',   [])
        self.data   = data.setdefault('data',   [])

        # Auxiliary tensors 0 and 1
        self.zeros = torch.Tensor([0])
        self.ones  = torch.Tensor([1])

        # Geometry
        self.alpha = data.get('alpha')
        self.zeta  = data.get('zeta')
        self.kappa = data.get('kappa')
        self.ratio = data.setdefault('ratio', 100)
        self.t_max = data.get('t_max')
        self.T     = self.kappa * self.ratio

        self.beta = data.setdefault('beta', -2.5)

        # Initial condition parameters
        self.N_IC      = data.get('N_IC')
        self.coordsIC = CoordsDataset(self.N_IC, [[0, 0], [0, 0]], "E")
        self.batchSizeIC = data.setdefault("batchSizeIC", 2*self.N_IC**2)

        # Boundary condition parameters
        self.N_BC   = data.get('N_BC')
        self.N_BC2  = self.N_BC**2
        self.c_cond = data.get('c_cond')
        self.times  = data.get('times')
        self.coordsBC = CoordsDataset(self.N_BC, [[0, 0], [0, 0]], "E")
        self.batchSizeBC = data.setdefault("batchSizeBC", 8*self.N_BC**2)

        # PDE parameters
        self.N_PDE = data.get('N_PDE')
        self.coordsPDE = CoordsDataset(self.N_PDE, [[0, 0], [0, 0], [0, 0]], "E")
        self.batchSizePDE = data.setdefault("batchSizePDE", 2*self.N_PDE**3)
        
        # Data parameters
        self.data_path = data.get("data_path")
        self.data_shape = data.get("data_shape", [101, 101, 101])
        self.dataset = NumericalDataset(self.data_shape[0], self.data_shape[1], self.data_shape[2], self.T, self.data_path, np.float64)
        self.batchSizeData = data.setdefault("batchSizeData", 16384)
        self.dataLoader = DataLoader(self.dataset, batch_size=self.batchSizeData, pin_memory=True, num_workers=4)

        # Crack width parameters
        self._w  = data.get('w')
        self._w1 = data.setdefault('w1', 1)
        self._w2 = data.setdefault('w2', 1)    
        self._w3 = data.setdefault('w3', -1)
        self._w4 = data.setdefault('w4', 2)
        self._w_name = data.setdefault('w_func', 'const')
        self.w_func = obj.Width(self._w_name, 1, self._w1, self._w2, self._w3, self._w4)
        
        # Neural network parameters
        self.act_dict  = {'Sin': obj.Sin(), 'Cos': obj.Cos(), 'Ricker':obj.MexicanHat(), 'Morlet':obj.Morlet(), None:obj.Sin()}
        self.NN_params = {}
        self.NN_params['input_size']  = data.get('NN_params', {1:1}).setdefault('input_size',  3)
        self.NN_params['output_size'] = data.get('NN_params', {1:1}).setdefault('output_size', 3)
        self.NN_params['neurons_arr'] = data.get('NN_params', {1:1}).setdefault('neurons_arr', [48,48,48,48,48])
        self.NN_params['depth']       = data.get('NN_params', {1:1}).setdefault('depth', 4)
        self.NN_params['act']         = self.act_dict.get(data.setdefault('NN_params', {1:1}).setdefault('act', None))
        
    def softReLU(self, x, eps = 0.001): return (x + torch.sqrt(x**2 + eps**2)) / 2
    
    def transform(self, pred):
        eps = 0.001
        X = (pred[:, 0] + torch.sqrt(pred[:, 0]**2 + eps**2)) / 2
        new_pred = pred.clone()
        new_pred[:, 0] = (X + 1 - torch.sqrt((X - 1)**2 + eps**2)) / 2
        return new_pred

    def computePDE(self, t, y, x):
                    
        prediction_PDE = self.model([t, y, x], self.transform)
    
        c   = prediction_PDE[:,0]
        p_x = prediction_PDE[:,1]
        p_y = prediction_PDE[:,2]
        
        width  = self.w_func(x, y)
        
        mu = self.viscosity(c)

        c_x = misc.derivative(c, x) * -width**2 * p_x / mu
        c_y = misc.derivative(c, y) * -width**2 * p_y / mu * self.alpha**2
        c_t = misc.derivative(c, t)

        ux_x = misc.derivative(-width**2 * p_x / mu, x)
        uy_y = misc.derivative(-width**2 * p_y / mu, y) * self.alpha**2
        
        convection = c_t + c_x + c_y
        
        poisson = ux_x + uy_y
        correlation = misc.derivative(p_x, y) - misc.derivative(p_y, x)
        
        return convection, poisson, correlation
    
    def lossFunction(self):
    
        self.optimizer.zero_grad()
        
        # Initial condition
        self.IC.append(0.0)
        if self.weights[3] != 0.0:
            t = self.coordsIC.t.to(self.device, non_blocking=True)
            y = self.coordsIC.y.to(self.device, non_blocking=True)
            x = self.coordsIC.x.to(self.device, non_blocking=True)
            c = self.c_IC.to(self.device)
                
            prediction_IC = self.model([t, y, x], self.transform)[:,0]
            loss_IC = self.weights[3] * self.criterion(prediction_IC, c)
            loss_IC.backward()
            self.IC[-1] += float(loss_IC.item())
    
        # Boundary conditions
        self.BC.append(0.0)
        if self.weights[4] != 0 or self.weights[5] != 0 or self.weights[6] != 0:
            loss_BC_total = torch.tensor(0.0, device=self.device)
        
            t = self.coordsBC.t.to(self.device, non_blocking=True)
            y = self.coordsBC.y.to(self.device, non_blocking=True)
            x = self.coordsBC.x.to(self.device, non_blocking=True)
            
            c = self.c.to(self.device, non_blocking=True)
            p = self.p.to(self.device, non_blocking=True)
            u = self.u.to(self.device, non_blocking=True)
  
            prediction_BC = self.model([t, y, x], self.transform)
            
            if self.weights[4] != 0:
                prediction_c = prediction_BC[:,0][self.where_c_in]
                loss_BC_total += self.weights[4] * self.criterion(prediction_c, c[self.where_c_in])

            if self.weights[5] != 0:
                K = self.N_BC2
                N = self.N_BC2
                prediction_px = prediction_BC[:,1]
                prediction_py = prediction_BC[:,2]
                loss_BC_total += self.weights[5] * (self.criterion(prediction_py[0*K:1*K], p[0*K:1*K]) +
                                            self.criterion(prediction_py[1*K:2*K], p[1*K:2*K]) +
                                            self.criterion(prediction_px[2*K:3*K], p[2*K:3*K]) +
                                            self.criterion(prediction_px[3*K:4*K], p[3*K:4*K]) +
                                            
                                            self.criterion(prediction_py[4*K+0*N:4*K+1*N], p[4*K+0*N:4*K+1*N]) +
                                            self.criterion(prediction_py[4*K+1*N:4*K+2*N], p[4*K+1*N:4*K+2*N]) +
                                            self.criterion(prediction_px[4*K+2*N:4*K+3*N], p[4*K+2*N:4*K+3*N]) +
                                            self.criterion(prediction_px[4*K+3*N:4*K+4*N], p[4*K+3*N:4*K+4*N])
                                            )                                       

            if self.weights[6] != 0:
                width  = self.w_func(x, y)
                mu = self.viscosity(prediction_BC[:,0])
                prediction_ux = -width**2 / mu * prediction_BC[:,1] * self.ratio
                prediction_uy = -width**2 / mu * prediction_BC[:,2] * self.ratio
                loss_BC_total += self.weights[5] * (self.criterion(prediction_uy[0*K:1*K], u[0*K:1*K]) +
                                            self.criterion(prediction_uy[1*K:2*K], u[1*K:2*K]) +
                                            self.criterion(prediction_ux[2*K:3*K], u[2*K:3*K]) +
                                            self.criterion(prediction_ux[3*K:4*K], u[3*K:4*K]) +
                                            
                                            self.criterion(prediction_uy[4*K+0*N:4*K+1*N], u[4*K+0*N:4*K+1*N]) +
                                            self.criterion(prediction_uy[4*K+1*N:4*K+2*N], u[4*K+1*N:4*K+2*N]) +
                                            self.criterion(prediction_ux[4*K+2*N:4*K+3*N], u[4*K+2*N:4*K+3*N]) +
                                            self.criterion(prediction_ux[4*K+3*N:4*K+4*N], u[4*K+3*N:4*K+4*N])
                                            )
            
            loss_BC_total.backward()
            self.BC[-1] += float(loss_BC_total.item())
        
        # PDE
        self.PDE.append(0.0)
        self.corr.append(0.0)
        if self.weights[0] != 0.0 or self.weights[1] != 0.0 or self.weights[2] != 0.0:
            loader = DataLoader(self.coordsPDE, batch_size=self.batchSizePDE, pin_memory=True, num_workers=4)
            
            for batch in loader:
                t = batch["t"].to(self.device, non_blocking=True).requires_grad_(True)
                y = batch["y"].to(self.device, non_blocking=True).requires_grad_(True)
                x = batch["x"].to(self.device, non_blocking=True).requires_grad_(True)
                conv, div, corr = self.computePDE(t, y, x)

                loss_PDE = (
                    self.weights[0] * self.criterion(conv, torch.zeros_like(conv)) +
                    self.weights[1] * self.criterion(div,  torch.zeros_like(div))
                )
                self.PDE[-1] += float(loss_PDE.item())
            
                loss_corr = self.weights[2] * self.criterion(corr, torch.zeros_like(corr))
                self.corr[-1] += float(loss_corr.item())
                
                total_loss = loss_PDE + loss_corr
                total_loss.backward()
    
        self.data.append(0.0)
        if self.weights[7] != 0.0:
            loader = DataLoader(self.dataset, batch_size=self.batchSizeData, pin_memory=True, num_workers=4)
            for batch in loader:
                x_data = batch["x"].to(self.device, non_blocking=True)
                y_data = batch["y"].to(self.device, non_blocking=True)
                t_data = batch["t"].to(self.device, non_blocking=True)
                c_true = batch["c"].to(self.device, non_blocking=True)
                
                c_pred = self.model([t_data, y_data, x_data], self.transform)[:,0]
                loss_data = self.weights[7] * self.criterion(c_pred, c_true)
                loss_data.backward()
                self.data[-1] += float(loss_data.item())

        loss = self.IC[-1] + self.BC[-1] + self.PDE[-1] + self.corr[-1] + self.data[-1]
        self.losses.append(loss)

        torch.cuda.empty_cache()

        if self.epoch % self.k == 0:
            self.end = time.time()
            self.print_tab.add_rows([['|',
                                    f'{self.epoch}\t',               '|',
                                    f'{self.PDE[-1]:2.5f}\t',     '|',
                                    f'{self.corr[-1]:2.5f}\t',    '|',
                                    f'{self.IC[-1]:2.5f}\t',      '|',
                                    f'{self.BC[-1]:2.5f}\t',      '|',
                                    f'{self.data[-1]:2.5f}\t',    '|',
                                    f'{self.losses[-1]:2.5f}\t',     '|',
                                    f'{self.end - self.start:1.6f}', '|'
                                    ]])
            print(self.print_tab.draw())
            self.start = time.time()
        self.epoch += 1
        return loss
        
    def train(self):
        self.print_tab = Texttable()
        self.print_tab.set_deco(Texttable.HEADER)
        self.print_tab.set_cols_width([1,10,1,15,1,15,1,15,1,15,1,15,1,15,1,10,1])
        self.print_tab.add_rows([['|','Epoch','|', 'PDE loss','|','p corr loss','|','IC loss','|','BC loss','|','Data loss','|','Summary loss','|','time','|']])
        print(self.print_tab.draw())

        self.model.train()

        for _ in range(self.max_iter):
            self.optimizer = self.model.set_optimizer('NAdam', lr=self.lr)
            while self.epoch < self.Adam_epochs:
                self.optimizer.step(self.lossFunction)
            self.optimizer = self.model.set_optimizer('LBFGS', max_iter=self.max_epoch)
            self.optimizer.step(self.lossFunction)
            if self.save_after:
                self.save(self.path)

    @staticmethod
    def load(path, loadloss=True, device='cpu'):
        with open(path + '.json') as data_file:
            data = json.load(data_file)
        if loadloss:
            with open(path + '_loss.json') as data_file:
                data_loss = json.load(data_file)
            data = {**data, **data_loss}
        
        instance = Poisson_Convection(data, device, torch.load(path+'.pt', map_location=device, weights_only=False))
        return instance
    
    def updateFromFile(self, path, loadloss=True, device='cpu'):
        with open(path+'.json') as data_file:
            data = json.load(data_file)
        if loadloss:
            with open(path+'_loss.json') as data_file:
                data_loss = json.load(data_file)
            data = {**data, **data_loss}
        self.updateData(data)
        self.model = torch.load(path+'.pt', map_location=device, weights_only=False)
        self.makeDistributedPoints()

    def save(self, path='', name='', saveloss=True):
        if path=='':
            current_date = datetime.now()
            folder_name = 'data/'+ current_date.strftime("%Y.%m.%d")
            os.makedirs(folder_name, exist_ok=True)
            path = folder_name + '/' + str(self.epoch)
        else:
            if not os.path.exists(path):
                os.makedirs(path)
        if name=='':
            path += f'/{self.epoch}'
        else:
            path += f'/{name}'
            
        self.NN_params['act'] = [k for k, v in self.act_dict.items() if v == self.NN_params['act']][0]
        data = {'Adam_epochs' : self.Adam_epochs,
                'epoch'       : self.epoch,
                'k'           : self.k,
                'max_epoch'   : self.max_epoch,
                'max_iter'    : self.max_iter,
                'save_after'  : self.save_after,
                'path'        : self.path,

                'weights'     : self.weights,

                'alpha'       : self.alpha,
                'zeta'        : self.zeta,
                'beta'        : self.beta,
                'times'       : self.times,
                'kappa'       : self.kappa,
                't_max'       : self.t_max,
                'ratio'       : self.ratio,

                'N_IC'        : self.N_IC,
                
                'N_BC'        : self.N_BC,
                'c_cond'      : self.c_cond,

                'N_PDE'       : self.N_PDE,
                
                'data_path'   : self.data_path,
                'data_shape'  : self.data_shape,

                'w1'          : self.w1,
                'w2'          : self.w2,
                'w3'          : self.w3,
                'w4'          : self.w4,
                'w_func'      : self.w_name,

                'NN_params'   : self.NN_params
               }
        
        with io.open(path+'.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(data,
                              indent=4, sort_keys=False,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(str(str_))

        if saveloss:
            dataloss = {'losses' : self.losses,
                        'PDE'    : self.PDE,
                        'BC'     : self.BC,
                        'IC'     : self.IC,
                        'corr'   : self.corr,
                        'data'   : self.data
                       }
            with io.open(path+'_loss.json', 'w', encoding='utf8') as outfile:
                str_ = json.dumps(dataloss,
                                  indent=4, sort_keys=False,
                                  separators=(',', ': '), ensure_ascii=False)
                outfile.write(str(str_))
        
        torch.save(self.model, path+'.pt')
        self.NN_params['act'] = self.act_dict.get(self.NN_params['act'])

    def psi(self, y):
        return torch.where((y - 1 / 2).abs().round(decimals=5) <= self.zeta / 2, 1., 0.)

    def viscosity(self, c):
        return (1 - c) ** self.beta

    def boundaryConditions(self):
        with torch.no_grad():
            t = self.coordsBC.t
            y = self.coordsBC.y
            x = self.coordsBC.x
            
            self.c = torch.zeros(len(x))
            self.p = torch.zeros(len(x))
            self.u = torch.zeros(len(x))

            psi        = self.psi(y)
            left_side  = torch.where(x==0, 1, 0)
            right_side = torch.where(x==1, 1, 0)

            times = [0.0] + [self.T*times for times in self.times] + [self.T*self.t_max]
            for i in range(len(times)-1):
                time_start = torch.where(t>=times[i], 1.0, 0.0)
                time_end   = torch.where(t<=times[i+1], 1.0, 0.0)
                self.c = torch.where(
                    time_start + 
                    time_end   +
                    left_side  +
                    psi     == 4,
                    self.c_cond[i], self.c
                )
                
                self.p = torch.where(
                    time_start + 
                    time_end   + 
                    left_side  + 
                    psi     == 4,
                   -self.viscosity(self.c_cond[i]), self.p
                ) 
                
                self.u = torch.where(
                    time_start + 
                    time_end   +
                    left_side  +
                    psi     == 4,
                    1.0, self.u
                )

            
            w_right = (self.psi(y[x==0]) * self.w_func(self.zeros, y[x==0])).sum() / (torch.ones_like(y[x==1]) * self.w_func(self.ones, y[x==1])).sum()
            self.p  = torch.where(right_side==1, -w_right, self.p)
            self.p /=  self.ratio * self.w_func(x, y)**2

            self.u  = torch.where(right_side==1,
                                  w_right, self.u)
    
    def initialConditions(self):
        self.c_IC = self.c_cond[0] * self.psi(self.coordsIC.y) * torch.where(self.coordsIC.x==0, 1, 0)
    
    def makeDistributedPoints(self):
        x_range = [0, 1]
        y_range = [0, 1]
        t_range = [0, self.T*self.t_max]
        
        # -------------------------
        # --- Initial Condition ---
        # -------------------------
        CL_IC = CoordsDataset(self.N_IC, [[0, 0], y_range, x_range], "L")
        CR_IC = CoordsDataset(self.N_IC, [[0, 0], y_range, x_range], "R")
        self.coordsIC = CL_IC + CR_IC
        self.initialConditions()
        
        # ------------------
        # --- PDE Points ---
        # ------------------
        CR_PDE = CoordsDataset(self.N_PDE, [t_range, y_range, x_range], "R") + self.coordsPDE[self.N_PDE**3:]
        
        loader = DataLoader(CR_PDE, batch_size=self.batchSizePDE, pin_memory=True, num_workers=4)
        sampled_indices_pde = []
        for batch in loader:
            t = batch["t"].to(self.device, non_blocking=True).requires_grad_(True)
            y = batch["y"].to(self.device, non_blocking=True).requires_grad_(True)
            x = batch["x"].to(self.device, non_blocking=True).requires_grad_(True)
            conv, div, corr = self.computePDE(t, y, x)
            pde_dist = (
                self.weights[0] * torch.where(conv.abs()>0.01, 1, 0) +
                self.weights[1] * torch.where(div.abs()>0.01,  1, 0) +
                self.weights[2] * torch.where(corr.abs()>0.01, 1, 0)
            )

            sampled_indices_pde += torch.multinomial(pde_dist / pde_dist.sum(), int(len(x) / 2), replacement=True).tolist()
        
        self.coordsPDE = CoordsDataset(self.N_PDE, [t_range, y_range, x_range], "L") + CR_PDE[sampled_indices_pde]

        # ---------------------------
        # --- Boundary Conditions ---
        # ---------------------------
        with torch.no_grad():
            CL_BC = (
                CoordsDataset(self.N_BC, [t_range, [1, 1], x_range], "L") +
                CoordsDataset(self.N_BC, [t_range, [0, 0], x_range], "L") +
                CoordsDataset(self.N_BC, [t_range, y_range, [0, 0]], "L") +
                CoordsDataset(self.N_BC, [t_range, y_range, [1, 1]], "L")
            )
            
            CR_BC = (
                CoordsDataset(self.N_BC, [t_range, [1, 1], x_range], "R") +
                CoordsDataset(self.N_BC, [t_range, [0, 0], x_range], "R") +
                CoordsDataset(self.N_BC, [t_range, y_range, [0, 0]], "R") +
                CoordsDataset(self.N_BC, [t_range, y_range, [1, 1]], "R")
            )
            
            self.coordsBC = CL_BC + CR_BC
            
            self.boundaryConditions()

            self.where_c_in = (self.coordsBC.x==0) & ((self.coordsBC.y - 1 / 2).abs().round(decimals=5) <= self.zeta / 2)

    def __str__(self):
        print_tab = Texttable() 
        print_tab.set_cols_align(["l", "l", "l", "l"])
        print_tab.set_cols_valign(["m", "m", "m", "m"])

        print_tab.set_precision(0)
        print_tab.set_cols_dtype(["t", "i", "t", "e"]) 
        print_tab.add_rows([ 
                ["Adam_epochs", self.Adam_epochs, "lr", self.lr]
                ], header=False) 
        
        print_tab.set_cols_dtype(["t", "i", "t", "a"]) 
        print_tab.add_rows([ 
                ["epoch", self.epoch, 'c_cond', [round(x, 3) for x in self.c_cond]], 
                ["path", self.path,'times', self.times], 
                ["weights", self.weights,'w_name', self.w_name]
                ], header=False) 

        try:
            print_tab.set_precision(3)
            print_tab.set_cols_dtype(["a", "e", "a", "e"]) 
            print_tab.add_rows([ 
                    ['last loss', self.losses[-1], 'min loss', min(self.losses)],
                    ], header=False) 
        except: pass

        print_tab.set_precision(3)
        print_tab.set_cols_dtype(["a", "f", "a", "i"]) 
        print_tab.add_rows([      
                ['alpha', self.alpha, 'N_IC', self.N_IC],
                ['zeta', self.zeta, 'N_PDE', self.N_PDE],
                ['kappa', self.kappa, 'N_BC', self.N_BC],
                ['t_max', self.t_max, 'ratio', self.ratio],
                ['beta', self.beta, '', '']
                ], header=False) 
        
        return print_tab.draw() 

    def eval(self) : self.model.eval()

    def convert(self, t, y, x):
        t = t * self.T
        return t, y, x

    def get_c(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        with torch.no_grad():
            c = self.model([t, y, x], self.transform)[:,0]
            return c.data.cpu().numpy()

    def get_px(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        with torch.no_grad():
            px = self.model([t, y, x], self.transform)[:,1] * self.ratio
            return px.data.cpu().numpy()

    def get_py(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        with torch.no_grad():
            py = self.model([t, y, x], self.transform)[:,2] * self.ratio
            return py.data.cpu().numpy()

    def get_ux(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        with torch.no_grad():
            pred = self.model([t, y, x], self.transform)
            mu = (1 - pred[:,0])**(self.beta)
            ux = -pred[:,1] * self.w_func(x,y)**2 * self.ratio / mu
            return ux.data.cpu().numpy()

    def get_uy(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        with torch.no_grad():
            pred = self.model([t, y, x], self.transform)
            mu = (1 - pred[:,0])**(self.beta)
            uy = -pred[:,2] * self.w_func(x,y)**2 * self.ratio / mu
            return uy.data.cpu().numpy()
    
    def get_mu(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        with torch.no_grad():
            mu = self.viscosity(self.model([t, y, x], self.transform)[:,0])
            return mu.data.cpu().numpy()

    def get_conv(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        pred = self.model([t, y, x], self.transform)
        mu = self.viscosity(pred[:,0])
        с_t = misc.derivative(pred[:,0], t)
        c_x = self.w_func(x,y)**2 / mu * pred[:,1] * misc.derivative(pred[:,0], x)
        c_y = self.w_func(x,y)**2 / mu * pred[:,2] * misc.derivative(pred[:,0], y) * self.alpha**2
        conv =  с_t - c_x - c_y
        return conv.data.cpu().numpy()

    def get_div(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        pred = self.model([t, y, x], self.transform)
        mu = self.viscosity(pred[:,0])
        ux = pred[:,1] * self.w_func(x,y)**2 / mu
        uy = pred[:,2] * self.w_func(x,y)**2 / mu
        div = misc.derivative(ux,x) + misc.derivative(uy,y)
        return div.data.cpu().numpy()
        
    def get_corr(self, t, y, x):
        t, y, x = self.convert(t, y, x)
        pred = self.model([t, y, x], self.transform)
        pxy = misc.derivative(pred[:,1],y)
        pyx = misc.derivative(pred[:,2],x)
        corr = pxy - pyx
        return corr.data.cpu().numpy()

    @property
    def w1(self): return self._w1
    @w1.setter
    def w1(self, w1):
        self._w1 = w1
        self.w_func = obj.Width(self._w_name, 1, self._w1, self._w2, self._w3, self._w4)

    @property
    def w2(self): return self._w2
    @w2.setter
    def w2(self, w2):
        self._w2 = w2
        self.w_func = obj.Width(self._w_name, 1, self._w1, self._w2, self._w3, self._w4)

    @property
    def w3(self): return self._w3
    @w3.setter
    def w3(self, w3):
        self._w3 = w3
        self.w_func = obj.Width(self._w_name, 1, self._w1, self._w2, self._w3, self._w4)

    @property
    def w4(self): return self._w4
    @w4.setter
    def w4(self, w4):
        self._w4 = w4
        self.w_func = obj.Width(self._w_name, 1, self._w1, self._w2, self._w3, self._w4)

    @property
    def w_name(self): return self._w_name
    @w_name.setter
    def w_name(self, w_name):
        self._w_name = w_name
        self.w_func = obj.Width(self._w_name, 1, self._w1, self._w2, self._w3, self._w4)







        ####################################################
        ####################################################
        ###                                              ###
        ###   DEPRECATED, BUT IT'S A PITY TO DELETE IT   ###
        ###                                              ###
        ####################################################
        ####################################################

        #     x_random, y_random = self.generate_random_points(self.N_IC, [x_range, y_range])
            
        #     try: self.x_IC
        #     except AttributeError:
        #         self.x_IC = torch.Tensor([]).to(self.device)
        #         self.y_IC = torch.Tensor([]).to(self.device)
    
        #     x = Variable(torch.cat((x_random, self.x_IC[self.N_IC**2:]))).to(self.device)
        #     y = Variable(torch.cat((y_random, self.y_IC[self.N_IC**2:]))).to(self.device)
        #     t = Variable(torch.zeros_like(x)).to(self.device)
        #     self.Initial_conditions(x,y)

        #     ic_dist = (self.model([x,y,t], self.transform)[:,0]).abs()
        #     if (ic_dist.sum()==0).item(): ic_dist = torch.ones_like(ic_dist)
        #     sampled_indices_ic = torch.multinomial(ic_dist/ic_dist.sum(), self.N_IC**2, replacement=True)
            
        # self.x_IC = Variable(torch.cat((x_linear, torch.index_select(x, -1, sampled_indices_ic))), requires_grad=True).to(self.device)
        # self.x_IC = Variable(torch.cat((x_linear, torch.index_select(x, -1, sampled_indices_ic))), requires_grad=True).to(self.device)
        
        
            # x = torch.Tensor(self.N_BC).to(self.device).uniform_(0, 1)
            # y = torch.Tensor(self.N_BC).to(self.device).uniform_(0, 1)
            # t = torch.Tensor(self.N_BC).to(self.device).uniform_(0, self.T*self.t_max)
            # c_condition_random = cnd.form_boundaries([x, y, t], self.ones, self.zeros)

            # K = self.l(self.N_BC)**2
            # try: self.x_BC
            # except AttributeError:
            #     K = 0
            #     self.x_BC = torch.Tensor([]).to(self.device)
            #     self.y_BC = torch.Tensor([]).to(self.device)
            #     self.t_BC = torch.Tensor([]).to(self.device)
    
            # x = Variable(torch.cat((self.x_BC[4*K:], c_condition_random[:,0])), requires_grad=True)
            # y = Variable(torch.cat((self.y_BC[4*K:], c_condition_random[:,1])), requires_grad=True)
            # t = Variable(torch.cat((self.t_BC[4*K:], c_condition_random[:,2])), requires_grad=True)

            # N = self.N_BC2
            
            # c, p, _ = self.Boundary_conditions(x,y,t)
    
            # bc_dist = self.model([x,y,t], self.transform)

            # bc_dist_top    = torch.cat(((bc_dist[:,2][:K] - p[:K]).abs(), (bc_dist[:,2][4*K:4*K+N] - p[4*K:4*K+N]).abs()))
            # xtop, t1       = self.make_BC_dist(bc_dist_top, torch.cat((x[:K], x[4*K:4*K+N])), torch.cat((t[:K], t[4*K:4*K+N])))
    
            # bc_dist_bottom = torch.cat(((bc_dist[:,2][K:2*K] - p[K:2*K]).abs(), (bc_dist[:,2][4*K+N:4*K+2*N] - p[4*K+N:4*K+2*N]).abs()))
            # xbottom, t2    = self.make_BC_dist(bc_dist_bottom, torch.cat((x[K:2*K], x[4*K+N:4*K+2*N])), torch.cat((t[K:2*K], t[4*K+N:4*K+2*N])))
    
            # bc_dist_left   = torch.cat((
            #                  (bc_dist[:,0] * self.psi(y) - c)[2*K:3*K].abs() + \
            #                  (bc_dist[:,1] - p)[2*K:3*K].abs(),
    
            #                  (bc_dist[:,0] * self.psi(y) - c)[4*K+2*N:4*K+3*N].abs() + \
            #                  (bc_dist[:,1] - p)[4*K+2*N:4*K+3*N].abs()
            #                  ))
    
            # yleft, t3      = self.make_BC_dist(bc_dist_left, torch.cat((y[2*K:3*K], y[4*K+2*N:4*K+3*N])), torch.cat((t[2*K:3*K], t[4*K+2*N:4*K+3*N])))
    
            # bc_dist_right  = torch.cat(((bc_dist[:,1][3*K:4*K] - p[3*K:4*K]).abs(), (bc_dist[:,1][4*K+3*N:] - p[4*K+3*N:]).abs()))
            # yright, t4     = self.make_BC_dist(bc_dist_right, torch.cat((y[3*K:4*K], y[4*K+3*N:])), torch.cat((t[3*K:4*K], t[4*K+3*N:])))
    
            # self.x_BC = Variable(torch.cat((c_condition_linear[:,0], xtop, xbottom, torch.zeros(self.N_BC2), torch.ones(self.N_BC2))), requires_grad=True)
            # self.y_BC = Variable(torch.cat((c_condition_linear[:,1], torch.ones(self.N_BC2), torch.zeros(self.N_BC2), yleft, yright)), requires_grad=True)
            # self.t_BC = Variable(torch.cat((c_condition_linear[:,2], t1, t2, t3, t4)), requires_grad=True)
            
    # def criterion(self, pred, true, time_term=0, t=None):
    #     C = 2
    #     D = 5
    #     term = 1
    #     if self.time_weights:
    #         if time_term==1:
    #             term = C * (1 - t / self.t_max) + 1 
    #         elif time_term==2:
    #             term = C * t / self.t_max + 1 
    #         elif time_term==3:
    #             term = 1 - C * (D * t * (t - self.t_max)**2 - t) / self.t_max
    #     raw_loss = self.crit_func(term * pred, term * true)
    #     return raw_loss

    # def relobralo_func(self, losses):
    #     ratios = torch.exp(losses / (self.rel_T * self.prev_losses))
    #     weights = torch.softmax(ratios, dim=0)
    #     if torch.rand(1).item() < self.rel_rho:
    #         weights = self.prev_losses
    #     updated_weights = self.rel_alpha * self.prev_losses + (1 - self.rel_alpha) * weights
    #     self.prev_losses = updated_weights.detach()
    #     return updated_weights
    
                                            #   self.criterion(prediction_py[4*K+0*N:4*K+1*N], self.p[4*K+0*N:4*K+1*N], 3, self.t_BC[4*K+0*N:4*K+1*N]) +
                                        #   self.criterion(prediction_py[4*K+1*N:4*K+2*N], self.p[4*K+1*N:4*K+2*N], 3, self.t_BC[4*K+1*N:4*K+2*N]) +
                                        #   self.criterion(prediction_px[4*K+2*N:4*K+3*N], self.p[4*K+2*N:4*K+3*N], 3, self.t_BC[4*K+2*N:4*K+3*N]) +
                                        #   self.criterion(prediction_px[4*K+3*N:4*K+4*N], self.p[4*K+3*N:4*K+4*N], 3, self.t_BC[4*K+3*N:4*K+4*N])
                                        #  )
                                                                                                                          
                                        #   self.criterion(prediction_uy[4*K+0*N:4*K+1*N], self.u[4*K+0*N:4*K+1*N]) +
                                        #   self.criterion(prediction_uy[4*K+1*N:4*K+2*N], self.u[4*K+1*N:4*K+2*N]) +
                                        #   self.criterion(prediction_ux[4*K+2*N:4*K+3*N], self.u[4*K+2*N:4*K+3*N]) +
                                        #   self.criterion(prediction_ux[4*K+3*N:4*K+4*N], self.u[4*K+3*N:4*K+4*N])
                                        
                                        
        #                                         self.time_weights = data.setdefault('time_weights', False)
        # self.relobralo    = data.setdefault('relobralo', False)
        # self.rel_alpha    = data.setdefault('rel_alpha', 0.99)
        # self.rel_T        = data.setdefault('rel_T', 0.1)
        # self.rel_rho      = data.setdefault('rel_rho', 0.999)
        # self.prev_losses  = torch.ones(4).to(self.device)
        
                #         'time_weights': self.time_weights,
                # 'relobralo'   : self.relobralo,
                # 'rel_alpha'   : self.rel_alpha,
                # 'rel_T'       : self.rel_T,
                # 'rel_rho'     : self.rel_rho,