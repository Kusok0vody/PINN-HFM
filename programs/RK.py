import numpy as np
import json
import io

import struct
import zlib
import warnings

from tqdm.notebook import tqdm
from scipy.sparse.linalg._dsolve.linsolve import MatrixRankWarning

from programs.Poisson_Solver import PoissonSolver
from programs.WENO5s import WENO5

def elliptic(x,y, chi):
    """Auxiliary function for RK3_solver"""
    r_square = x**2 + 4/(1-chi**2)*(y-1/2)**2
    R_square = 1/(1-chi**2)
    eps = 10**(-2)
    diff = (R_square - r_square)/R_square + eps**2
    # return np.where(0<diff, np.sqrt(np.abs(diff)), eps)
    return np.ones_like(x)

class RK_solver():
        
    def __init__(self, init_params: dict):
        mindata = {'chi'  : float,
                   'cmax' : float,
                   'beta' : float,
                   'mu0'  : float,
                   'rho1' : float,
                   'rho2' : float,
                   'v'    : float,
                   'w0'   : float,
                   'Nt'   : int,
                   'Nx'   : int,
                   'Ny'   : int,
                   'L'    : float,
                   'H'    : float,
                   'c_in_arr'       : list,
                   'c_in_durations' : list,
                   'CFL'            : float,
                   'WENO_type'      : str
                   }
        
        for i in mindata.keys():
            if not(i in init_params):
                raise ValueError(f'Value \"{i}\" not found')
            if type(mindata[i])==int and not(type(init_params[i])==int):
                raise ValueError(f'Value {init_params[i]} in {i} must be integer')
        init_params['c_in_times'] = np.cumsum(init_params['c_in_durations'])
        self.update_data(init_params)


    def update_data(self, data:dict):
        self.Nt = data.get('Nt')
        self.Nx = data.get('Nx')
        self.Ny = data.get('Ny')

        self.mu0  = data.get('mu0')
        self.cmax = data.get('cmax')
        self.beta = data.get('beta')
        self.rho1 = data.get('rho1')
        self.rho2 = data.get('rho2')
        self.g    = data.setdefault('g', 0)
        self.v    = data.get('v')
        self.w0   = data.get('w0')

        self.chi = data.get('chi')
        self.L   = data.get('L')
        self.H   = data.get('H')
        self.dx  = data.setdefault('dx', self.L/self.Nx)
        self.dy  = data.setdefault('dy', self.H/self.Ny)

        self.x0_points = np.linspace(self.dx/2, self.L-self.dx/2, self.Nx)
        self.y0_points = np.linspace(self.dy/2, self.H-self.dy/2, self.Ny)
        self.use_WENO  = data.setdefault('use_WENO', True)
        self.rk_stages = data.setdefault('rk_stages', 3)
        self.lim_type  = data.setdefault('lim_type', 'minmod')
        self.riemann   = data.setdefault('riemann', False)

        self.WENO_type           = data.setdefault('WENO_type', 'Z')
        self.prefer_sparse       = data.setdefault('prefer_sparse', True)
        self.Pressure_Solver     = PoissonSolver(data)
        self.WENO5_reconstructor = WENO5(data)

        self.dT   = data.setdefault('dT', 10**(-2))
        self.CFL  = data.get('CFL')
        self.eps  = data.setdefault('eps', 10**(-10))
        self.step = data.setdefault('step', 0)
        
        self.psi = np.where(np.abs(self.y0_points-self.H/2)<self.chi/2,1,0)

        self.meshPQ_X, self.meshPQ_Y = np.meshgrid(self.x0_points, self.y0_points)
        self.meshVx_X, self.meshVx_Y = np.meshgrid(np.linspace(0, self.L, self.Nx+1), self.y0_points)
        self.meshVy_X, self.meshVy_Y = np.meshgrid(self.x0_points, np.linspace(0, self.H, self.Ny+1))
         
        self.c_in_arr   = data.get('c_in_arr')
        self.c_in_times = data.get('c_in_times')
        
        self.PQ_shape = data.setdefault('PQ_shape', (self.Nt, self.Ny,   self.Nx  ))
        self.Vx_shape = data.setdefault('Vx_shape', (self.Nt, self.Ny,   self.Nx+1))
        self.Vy_shape = data.setdefault('Vy_shape', (self.Nt, self.Ny+1, self.Nx  ))

        self.times = data.setdefault('times', np.zeros(shape=(self.Nt,)))
        self.Q  = np.array(data.get('Q_array' )).reshape(self.PQ_shape) if data.get('Q_array')!=None  else np.zeros(self.PQ_shape)
        self.P  = np.array(data.get('P_array' )).reshape(self.PQ_shape) if data.get('P_array')!=None  else np.zeros(self.PQ_shape)
        self.Vx = np.array(data.get('Vx_array')).reshape(self.Vx_shape) if data.get('Vx_array')!=None else np.zeros(self.Vx_shape)
        self.Vy = np.array(data.get('Vy_array')).reshape(self.Vy_shape) if data.get('Vy_array')!=None else np.zeros(self.Vy_shape)

        self.w     = np.array(data.get('w')).reshape(self.meshPQ_X.shape)  if data.get('w')!=None  else self.w0*elliptic(self.meshPQ_X/self.L, self.meshPQ_Y/self.H, self.chi)
        # self.v_in  = self.v*self.w[:,0]  / self.w0
        # self.v_out = self.v*self.w[:,-1] / self.w0 * np.mean(self.w[:,0]**2) / np.mean(self.w[:,-1]**2)
        
        self.v_in = self.v * self.psi
        self.v_out = self.v * np.sum(self.psi*self.w[:,0]) / np.sum(self.psi*self.w[:,-1]) * self.psi


    def load(self, path):
        with open(path+'.json') as data_file:
            data = json.load(data_file)
        self.update_data(data)

        arrays = []
        with open(path+'.bin', 'rb') as f:
            num_arrays = struct.unpack('I', f.read(4))[0]

            for _ in range(num_arrays):
                num_dims = struct.unpack('I', f.read(4))[0]
                shape = struct.unpack(f'{num_dims}I', f.read(4 * num_dims))

                compressed_size = struct.unpack('I', f.read(4))[0]
                compressed_data = f.read(compressed_size)
                binary_data = zlib.decompress(compressed_data)

                array_flat = np.frombuffer(binary_data, dtype=np.float64)
                array = array_flat.reshape(shape)
                arrays.append(array)
        self.Q, self.P = arrays

        for n in tqdm(range(self.Nt+1), 'Recomputing velocities'):
            Lambda = -self.w**2 / (12 * self.mu(self.Q[n]/self.w))

            Fy = Lambda*self.rho(self.Q[n]/self.w)*self.g
            if self.g==0:
                Fy_center = np.zeros((self.Ny-1, self.Nx))
            else: 
                Fy_center = 2/(1/Fy[:-1] + 1/Fy[1:])
            LambdaY = 2/(1/Lambda[:-1] + 1/Lambda[1:]) #ij -> Lambda i+1/2, j
            LambdaX = 2/(1/Lambda[:,:-1] + 1/Lambda[:,1:]) #ij -> Lambda i, j+1/2

            self.Vx[n][:, 1:-1] = LambdaX * (self.P[n][:,1:] - self.P[n][:,:-1])/self.dx
            self.Vx[n][:, 0] = self.v_in
            self.Vx[n][:, -1] = self.v_out

            self.Vy[n][1:-1] = LambdaY * (self.P[n][1:] - self.P[n][:-1])/self.dy  + Fy_center

        
    def save(self, path):
        data = {'Nt'     : self.step,
                'Nx'     : self.Nx,
                'Ny'     : self.Ny,

                'mu0'    : self.mu0,
                'cmax'   : self.cmax,
                'beta'   : self.beta,
                'rho1'   : self.rho1,
                'rho2'   : self.rho2,
                'g'      : self.g,
                'v'      : self.v,
                'w0'     : self.w0,

                'chi'    : self.chi,
                'L'      : self.L,
                'H'      : self.H,
                'dx'     : self.dx,
                'dy'     : self.dy,
                
                'use_WENO'      : self.use_WENO,
                'rk_stages'     : self.rk_stages,
                'lim_type'      : self.lim_type,
                'riemann'       : self.riemann,
                'WENO_type'     : self.WENO_type,
                'prefer_sparse' : self.prefer_sparse,

                'CFL'    : self.CFL,
                'eps'    : self.eps,
                'dT'     : self.dT,
                'step'   : self.step,

                'PC_shape'   : self.PQ_shape,
                'Vx_shape'   : self.Vx_shape,
                'Vy_shape'   : self.Vy_shape,

                'c_in_arr'   : self.c_in_arr,
                'c_in_times' : self.c_in_times.tolist(),
                'times'      : self.times.tolist(),
                'w'          : self.w.tolist(),
                }

        with io.open(path+'.json', 'w', encoding='utf8') as outfile:
            str_ = json.dumps(data,
                              indent=4, sort_keys=False,
                              separators=(',', ': '), ensure_ascii=False)
            outfile.write(str_)

        with open(path+'.bin', 'wb') as f:
            arrays = [self.Q, self.P]
            f.write(struct.pack('I', len(arrays)))

            for array in arrays:
                shape = array.shape
                f.write(struct.pack('I', len(shape)))
                f.write(struct.pack(f'{len(shape)}I', *shape))

                binary_data = array.tobytes()
                compressed_data = zlib.compress(binary_data)
                f.write(struct.pack('I', len(compressed_data)))
                f.write(compressed_data)
        print(f'Saved at {path}')


    def mu(self, c:np.ndarray) -> np.ndarray: 
        """Computes mu(c)\n
        Parameters
        ----------
        c: np.ndarray
            concentration, 2D array, shape = (Ny, Nx) 
        """
        return self.mu0 * (1 - c/self.cmax)**(-self.beta)


    def rho(self,c:np.ndarray) -> np.ndarray:
        """Computes rho(c)\n
        Parameters
        ----------
        c: np.ndarray
            concentration, 2D array, shape = (Ny, Nx) 
        """
        return self.rho1 + (self.rho2-self.rho1)*c


    def pressure_rhs(self, wFy_center) -> np.ndarray:
        "Auxiliary function for pressure_and_velocities(), computes RHS for linear equation A@p = rhs"
        d_wFy_dy = np.zeros_like(self.w)
        d_wFy_dy[1:-1] = (wFy_center[1:]-wFy_center[:-1])/self.dy
        
        rhs = -d_wFy_dy*self.dx*self.dy

        rhs[:,0] += self.v_in*self.w[:,0]*self.dy
        
        rhs[:, -1] += -self.v_out*self.w[:,-1]*self.dy

        rhs[0] += -wFy_center[0]*self.dx
        rhs[-1] += wFy_center[-1]*self.dx
        return rhs
    

    def pressure_and_velocities(self, c: np.ndarray,
                                p0: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Computes pressure and velocities fields for given concentration\n
        Parameters
        ----------
        c: np.ndarray
            concentration, 2D array, shape = (Ny, Nx)
        p0: np.ndarray
            initial approximation for CG poisson solver, 2D array, shape = (Ny, Nx)    
        """
        Lambda = -self.w**2 / (12 * self.mu(c))
        
        Fy = Lambda*self.rho(c)*self.g
        wFy = Fy*self.w
        if self.g==0:
            Fy_center = np.zeros((self.Ny-1, self.Nx))
            wFy_center = np.zeros((self.Ny-1, self.Nx))
        else: 
            Fy_center = 2/(1/Fy[:-1] + 1/Fy[1:])
            wFy_center = 2/(1/wFy[:-1] + 1/wFy[1:])
        rhs_matrix = self.pressure_rhs(wFy_center)  
        if self.prefer_sparse or self.step==0:           
            warnings.filterwarnings("error")
            try:
                pressure = self.Pressure_Solver.solve_sparse(self.w*Lambda, rhs_matrix)
                
            except MatrixRankWarning:
                print(self.step, '\t')
                print('emergency cg')
                
                pressure = self.Pressure_Solver.solve_CG(self.w*Lambda, rhs_matrix, p0, self.eps) # values at points y_i, x_j
            warnings.resetwarnings()
        else: 
            pressure = self.Pressure_Solver.solve_CG(self.w*Lambda, rhs_matrix, p0, self.eps) # values at points y_i, x_j
        LambdaY = 2/(1/Lambda[:-1] + 1/Lambda[1:]) #ij -> Lambda i+1/2, j
        LambdaX = 2/(1/Lambda[:,:-1] + 1/Lambda[:,1:]) #ij -> Lambda i, j+1/2
        
        vx = np.zeros((self.Ny, self.Nx+1))
        vx[:, 1:-1] = LambdaX * (pressure[:,1:] - pressure[:,:-1])/self.dx
        vx[:, 0] = self.v_in
        vx[:, -1] = self.v_out
        
        
        vy = np.zeros((self.Ny+1, self.Nx))
        vy[1:-1] = LambdaY * (pressure[1:] - pressure[:-1])/self.dy  + Fy_center

        return pressure - np.min(pressure), vx, vy


    def apply_IC(self, c0: np.ndarray):   
        """Computes values for t^0 and updates data arrays\n
        Parameters
        ----------
        c0: np.ndarray
            concentration at initial moment, 2D array, shape = (Ny, Nx)
        """
        self.Q[0] = c0*self.w
        self.P[0], self.Vx[0], self.Vy[0] = self.pressure_and_velocities(c0, np.zeros_like(c0).reshape(-1))


    def lim(self, x,y):
        if self.lim_type=='minmod':
            return np.sign(x)*np.maximum(0.0, np.minimum(abs(x), np.sign(x)*y))
        if self.lim_type=='van':
            eps = 10**(-12)
            return (x*(np.square(y)+eps)+ y*(np.square(x)+eps))/ (np.square(x) + np.square(y)+2*eps)
        

    def boundary_vals(self,q, c_in):
        extQ_x = np.zeros((self.Ny, self.Nx+6))
        extQ_x[:,0] = self.w[:,0]*c_in*self.psi + q[:,0]*(1-self.psi)
        extQ_x[:,1] = self.w[:,0]*c_in*self.psi + q[:,0]*(1-self.psi)
        extQ_x[:,2] = self.w[:,0]*c_in*self.psi + q[:,0]*(1-self.psi)
        extQ_x[:,3:-3] = q
        extQ_x[:,-3] = q[:,-1]
        extQ_x[:,-2] = q[:,-1]
        extQ_x[:,-1] = q[:,-1]

        extQ_y = np.zeros((self.Ny+6, self.Nx))
        extQ_y[0] = q[0]
        extQ_y[1] = q[0]
        extQ_y[2] = q[0]
        extQ_y[3:-3] = q
        extQ_y[-3] = q[-1]
        extQ_y[-2] = q[-1]
        extQ_y[-1] = q[-1]

        if self.use_WENO:
            extQ_left, extQ_right = self.WENO5_reconstructor.reconstruct(extQ_x, 'x')
            extQ_bottom, extQ_top = self.WENO5_reconstructor.reconstruct(extQ_y, 'y')

            Q_left   = extQ_left[:,2:-2]
            Q_right  = extQ_right[:,2:-2]
            Q_bottom = extQ_bottom[2:-2]
            Q_top    = extQ_top[2:-2] 
        else:
            kappa = -1 #-1, 1/3
            kappaM = 0.25*(1.0 - kappa)
            kappaP = 0.25*(1.0 + kappa)            

            dqx = extQ_x[:,1:]-extQ_x[:,:-1]
            dqy = extQ_y[1:]-extQ_y[:-1] 

            dqr = self.lim(dqx[:,1:],  dqx[:,:-1])[:,1:-1]
            dql = self.lim(dqx[:,:-1], dqx[:,1:])[:,1:-1]
            dqt = self.lim(dqy[1:],    dqy[:-1])[1:-1]
            dqb = self.lim(dqy[:-1],   dqy[1:])[1:-1]

            Q_left   = extQ_x[:,2:-2] - kappaM * dqr - kappaP * dql
            Q_right  = extQ_x[:,2:-2] + kappaM * dqr + kappaP * dql
            Q_bottom = extQ_y[2:-2]   - kappaM * dqt - kappaP * dqb
            Q_top    = extQ_y[2:-2]   + kappaM * dqt + kappaP * dqb
                
        return Q_left, Q_right, Q_bottom, Q_top
    

    def Lu(self, Q_left, Q_right, Q_bottom, Q_top, Vx, Vy, dt):
        if self.riemann=='force':
            fx, fy = self.force(Q_left, Q_right, Q_bottom, Q_top, Vx,Vy,dt)
            fxp = -np.minimum(fx[:,1:],0)+np.maximum(fx[:,:-1],0)
            fyp = -np.minimum(fy[1:],0)+np.maximum(fy[:-1],0)
            fxm = -np.minimum(fx[:,:-1],0)+np.maximum(fx[:,1:],0)
            fym = -np.minimum(fy[:-1],0)+np.maximum(fy[1:],0)
            
        else:
            fxp = -Q_left[:,2:]   * np.minimum(Vx[:,1:],  0) + Q_right[:,:-2]  * np.maximum(Vx[:,:-1], 0)
            fxm = -Q_left[:,1:-1] * np.minimum(Vx[:,:-1], 0) + Q_right[:,1:-1] * np.maximum(Vx[:,1:],  0)

            fyp = -Q_bottom[2:]   * np.minimum(Vy[1:],  0) + Q_top[:-2]  * np.maximum(Vy[:-1], 0)
            fym = -Q_bottom[1:-1] * np.minimum(Vy[:-1], 0) + Q_top[1:-1] * np.maximum(Vy[1:],  0)

        return (fxp-fxm)/self.dx + (fyp-fym)/self.dy 
    
    
    def force(self, qL, qR, qB, qT, vx, vy, dt):
        
        uL, uR, uT, uB = qR[:,:-1], qL[:,1:], qT[:-1], qB[1:]

        fL, fR, fB, fT = uL*vx, uR*vx, uB*vy, uT*vy

        uLWx = (uL + uR - dt / self.dx * (fR - fL)) / 2
        fLWx = uLWx * vx
        uLWy = (uB + uT - dt / self.dx * (fT - fB)) / 2
        fLWy = uLWy * vy
        return 0.25 * (fL + 2.0 * fLWx + fR - self.dx / dt * (uR - uL)), 0.25 * (fB + 2.0 * fLWy + fT - self.dy / dt * (uT - uB))
    
    
    def RK(self,
            Q: np.ndarray,
            Q_RK: np.ndarray,
            dt: float,
            Vx: np.ndarray,
            Vy: np.ndarray,
            c_in: float,
            step : int,
          ) -> np.ndarray:
        """
        First step RK
        -------------
        q^n(1) = q^n  + dt*L(q^n)
        L(f) = div(fV), V = (Vx, Vy)\n
        Second step RK
        --------------
        q^n(2) = 3*q^n/4 + (q^n(1) + dt*L(q^n(1)))/4
        L(f) = div(fV), V = (Vx, Vy)\n
        Third step RK
        -------------
        q^n+1 = q^n/3 + (q^n(2) + dt*L(q^n(2)))*2/3
        L(f) = div(fV), V = (Vx, Vy)\n
        Parameters
        ----------
        Q: np.ndarray
            q^n, 2D array, shape = (Ny, Nx)
        Q_RK: np.ndarray
            q^n(), 2D array, shape = (Ny, Nx)
        dt: float
            dt
        Vx: np.ndarray
            Vx, 2D array, shape = (Ny, Nx+1)
        Vy: np.ndarray
            Vy, 2D array, shape = (Ny+1, Nx)
        c_in: float
            Concentration at left boundary
        step : 1, 2, 3
            The step of the RK
        """
        Q_left, Q_right, Q_bottom, Q_top = self.boundary_vals(Q_RK, c_in)
       
        Lu = self.Lu(Q_left, Q_right, Q_bottom, Q_top, Vx, Vy, dt)
        
        if step==1: new_Q = Q * 0 + (Q_RK + dt * Lu)

        else:
            if self.rk_stages==2:
                if step==1: new_Q = Q * 0 + (Q_RK + dt * Lu)

                elif step==2: new_Q = Q / 2 + 1/2 * (Q_RK + dt * Lu)
                    
            elif self.rk_stages==3:
                if step==2: new_Q = 3/4 * Q + 1/4*(Q_RK + dt * Lu)

                elif step==3: new_Q = 1/3 * Q + 2/3*(Q_RK + dt * Lu)
                
        return np.minimum(np.maximum(new_Q,0),(self.cmax-0.01)*self.w)

        
    def make_step(self, Q, P, Vx, Vy, current_time):
        """
        Computes values for t^k+1 and updates data arrays\n
        Parameters
        ----------
        k: int
            index of previous step
        """
        if self.rk_stages==1: return self.rk1_cycle(Q, P, Vx, Vy, current_time)

        if self.rk_stages==2: return self.rk2_cycle(Q, P, Vx, Vy, current_time)        

        if self.rk_stages==3: return self.rk3_cycle(Q, P, Vx, Vy, current_time)
    
        
    def solve(self, tmax=-1):
        if tmax==-1: tmax = self.Nt*self.dT
        q, p, vx, vy, current_time = (self.Q[self.step],
                                      self.P[self.step],
                                      self.Vx[self.step],
                                      self.Vy[self.step],
                                      self.times[self.step],
                                      )
        local_step = 0
        total_steps = min(int((tmax -  current_time) / self.dT), self.Nt)
        progress_bar = tqdm(total=total_steps, desc="Solving", unit="step")

        while current_time <= tmax:
            q, p, vx, vy, current_time = self.make_step(q, p, vx, vy, current_time)
            local_step += 1

            if np.abs(current_time - self.dT * (self.step + 1)) <= 1e-5:
                self.step += 1
                self.times[self.step] = current_time
                self.Q[self.step], self.P[self.step], self.Vx[self.step], self.Vy[self.step] = q, p, vx, vy
                progress_bar.update(1)

                if self.step == self.Nt - 1:
                    print("Nt-limit")
                    break

        progress_bar.close()


    def rk1_cycle(self, Q, P, Vx, Vy, current_time):
        """
        Computes values for t^k+1 and updates data arrays\n
        Parameters
        ----------
        k: int
            index of previous step
        """
        tau = self.CFL * np.min([self.dx / np.max(np.abs(Vx)), self.dy / np.max(np.abs(Vy))])
        dt = np.min([tau, self.dT * (self.step + 1) - current_time])

        c_in_index_now = np.searchsorted(self.c_in_times, current_time)
        c_in = self.c_in_arr[c_in_index_now]

        q_RK1 = self.RK(Q, Q, dt, Vx, Vy, c_in, 1)    
        p_RK1, vx_RK1, vy_RK1 = self.pressure_and_velocities(q_RK1/self.w, P)
        return q_RK1, p_RK1, vx_RK1, vy_RK1, current_time+dt
    

    def rk2_cycle(self, Q, P, Vx, Vy, current_time):
        """
        Computes values for t^k+1 and updates data arrays\n
        Parameters
        ----------
        k: int
            index of previous step
        """
        tau = self.CFL*np.min([self.dx / np.max(np.abs(Vx)), self.dy / np.max(np.abs(Vy))])
        dt = np.min([tau, self.dT * (self.step + 1) - current_time])

        c_in_index_now = np.searchsorted(self.c_in_times, current_time)
        c_in = self.c_in_arr[c_in_index_now]

        q_RK1 = self.RK(Q, Q, dt, Vx, Vy, c_in, 1)    
        p_RK1, vx_RK1, vy_RK1 = self.pressure_and_velocities(q_RK1/self.w, P)
            
        q_RK2 = self.RK(Q, q_RK1, dt, vx_RK1, vy_RK1, c_in, 2)
        p_RK2, vx_RK2, vy_RK2 = self.pressure_and_velocities(q_RK2/self.w, p_RK1)

        return q_RK2, p_RK2, vx_RK2, vy_RK2, current_time+dt
    

    def rk3_cycle(self, Q, P, Vx, Vy, current_time):
        """
        Computes values for t^k+1 and updates data arrays\n
        Parameters
        ----------
        k: int
            index of previous step
        """
        tau = self.CFL * np.min([self.dx / np.max(np.abs(Vx)), self.dy / np.max(np.abs(Vy))])
        dt = np.min([tau, self.dT * (self.step + 1) - current_time])

        c_in_index_now = np.searchsorted(self.c_in_times, current_time)
        c_in = self.c_in_arr[c_in_index_now]

        q_RK1 = self.RK(Q, Q, dt, Vx, Vy, c_in, 1)    
        p_RK1, vx_RK1, vy_RK1 = self.pressure_and_velocities(q_RK1/self.w, P)
            
        q_RK2 = self.RK(Q, q_RK1, dt, vx_RK1, vy_RK1, c_in, 2)
        p_RK2, vx_RK2, vy_RK2 = self.pressure_and_velocities(q_RK2/self.w, p_RK1)
            
        q_RK3 = self.RK(Q, q_RK2, dt, vx_RK2, vy_RK2, c_in, 3)
        p_RK3, vx_RK3, vy_RK3 = self.pressure_and_velocities(q_RK3/self.w, p_RK2) 
        return q_RK3, p_RK3, vx_RK3, vy_RK3, current_time+dt