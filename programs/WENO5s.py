import numpy as np
from scipy.sparse import eye

    
class WENO5():
    def __init__(self, params):
        self.type = params.get('WENO_type')
        

    def compute_IS_K(self, k, f: np.ndarray, projection: str) -> np.ndarray:
        """Computes smoothness indicator n0
         Parameters
        ----------
        f: np.ndarray
            data array, 2D array, shape = (Ny, Nx)
        projection: str
            axis of array used for computation
            """
        if projection == 'x': N = f.shape[1]
        if projection == 'y': N = f.shape[0]
        
        if k==0:
            m1 = eye(N,N, -2) -2*eye(N,N,-1) + eye(N,N)
            m2 = eye(N,N,-2) -4*eye(N,N,-1) + 3*eye(N,N)
        if k==1:
            m1 = eye(N,N, -1) -2*eye(N,N) + eye(N,N,1)
            m2 = eye(N,N,-1) - eye(N,N,+1)
        if k==2:
            m1 = eye(N,N) -2*eye(N,N,1) + eye(N,N, 2)
            m2 = 3*eye(N,N) -4* eye(N,N,1) + eye(N,N,2)
            
        if projection == 'x': return 13/12*np.square(f@m1.T) + 1/4*np.square(f@m2.T)
        if projection == 'y': return 13/12*np.square(m1@f) + 1/4*np.square(m2@f)
        
    
    def eno_K(self, k, f: np.ndarray, projection: str) -> np.ndarray:
        """Computes eno n0 at boundary
         Parameters
        ----------
        f: np.ndarray
            data array, 2D array, shape = (Ny, Nx)
        side: str
            boundary of the cell
            """
        if projection == 'x': N = f.shape[1]
        if projection == 'y': N = f.shape[0]
            
        if k == 0:
            mlb = -1/6*eye(N,N, -2) + 5/6*eye(N,N,-1) + 2/6*eye(N,N)
            mrt = 2/6*eye(N,N, -2) -7/6*eye(N,N,-1) + 11/6*eye(N,N)
        if k == 1:
            mlb = 2/6*eye(N,N, -1)  + 5/6*eye(N,N) - 1/6*eye(N,N,1)
            mrt = -1/6*eye(N,N, -1) + 5/6*eye(N,N) + 2/6*eye(N,N,1)
        if k == 2:
            mlb = 11/6*eye(N,N) - 7/6*eye(N,N,1) + 2/6*eye(N,N,2)
            mrt = 2/6*eye(N,N)  + 5/6*eye(N,N,1) - 1/6*eye(N,N,2)

        if projection == 'x':
            dataL = f@mlb.T
            dataR = f@mrt.T
        if projection == 'y':
            dataL = mlb@f
            dataR = mrt@f
        return dataL, dataR

    
    def weights(self, f: np.ndarray, projection: str) -> np.ndarray:
        """Computes weights
         Parameters
        ----------
        f: np.ndarray
            data array, 2D array, shape = (Ny, Nx)
        projection: str
            axis of array used for computation
            """
        weights_lb = np.zeros((f.shape[0], f.shape[1], 3))
        weights_rt = np.zeros((f.shape[0], f.shape[1], 3))
       
        dl0, dl1, dl2 = 0.3, 0.6, 0.1
        dr0, dr1, dr2 = 0.1, 0.6, 0.3
        eps = 10**(-5)
        IS0 = self.compute_IS_K(0, f, projection)
        IS1 = self.compute_IS_K(1, f, projection)
        IS2 = self.compute_IS_K(2, f, projection)
        if self.type != 'Z':
            
            alphal0 = dl0/((eps + IS0)**2)
            alphal1 = dl1/((eps + IS1)**2)
            alphal2 = dl2/((eps + IS2)**2)
            
            sum_l = alphal0 + alphal1 + alphal2
            weights_lb[:,:,0] = alphal0/sum_l
            weights_lb[:,:,1] = alphal1/sum_l
            weights_lb[:,:,2] = alphal2/sum_l
            
            alphar0 = dr0/((eps + IS0)**2)
            alphar1 = dr1/((eps + IS1)**2)
            alphar2 = dr2/((eps + IS2)**2)
            sum_r = alphar2 + alphar1 + alphar0
            
            weights_rt[:,:,0] = alphar0/sum_r
            weights_rt[:,:,1] = alphar1/sum_r
            weights_rt[:,:,2] = alphar2/sum_r

        if self.type == 'M':
            sigmal0 = weights_lb[:,:,0]*(dl0 + dl0**2 - 3*dl0*weights_lb[:,:,0]+weights_lb[:,:,0]**2)/(dl0**2 + weights_lb[:,:,0]*(1-2*dl0))
            sigmal1 = weights_lb[:,:,1]*(dl1 + dl1**2 - 3*dl1*weights_lb[:,:,1]+weights_lb[:,:,1]**2)/(dl1**2 + weights_lb[:,:,1]*(1-2*dl1))
            sigmal2 = weights_lb[:,:,2]*(dl2 + dl2**2 - 3*dl2*weights_lb[:,:,2]+weights_lb[:,:,2]**2)/(dl2**2 + weights_lb[:,:,2]*(1-2*dl2))
            weights_lb[:,:,0] = sigmal0/(sigmal0 + sigmal1 + sigmal2)
            weights_lb[:,:,1] = sigmal1/(sigmal0 + sigmal1 + sigmal2)
            weights_lb[:,:,2] = sigmal2/(sigmal0 + sigmal1 + sigmal2)
            
            sigmar0 = weights_rt[:,:,0]*(dr0 + dr0**2 - 3*dr0*weights_rt[:,:,0]+weights_rt[:,:,0]**2)/(dr0**2 + weights_rt[:,:,0]*(1-2*dr0))
            sigmar1 = weights_rt[:,:,1]*(dr1 + dr1**2 - 3*dr1*weights_rt[:,:,1]+weights_rt[:,:,1]**2)/(dr1**2 + weights_rt[:,:,1]*(1-2*dr1))
            sigmar2 = weights_rt[:,:,2]*(dr2 + dr2**2 - 3*dr2*weights_rt[:,:,2]+weights_rt[:,:,2]**2)/(dr2**2 + weights_rt[:,:,2]*(1-2*dr2))
            weights_rt[:,:,0] = sigmar0/(sigmar0 + sigmar1 + sigmar2)
            weights_rt[:,:,1] = sigmar1/(sigmar0 + sigmar1 + sigmar2)
            weights_rt[:,:,2] = sigmar2/(sigmar0 + sigmar1 + sigmar2)
            
        if self.type == 'Z':
            tau5 = np.abs(IS0 - IS2)
            epsZ = 10**(-10)
            IS0Z = (IS0 + epsZ)/(IS0 + tau5 + epsZ)
            IS1Z = (IS1 + epsZ)/(IS1 + tau5 + epsZ)
            IS2Z = (IS2 + epsZ)/(IS2 + tau5 + epsZ)
            sigmal0 = dl0/IS0Z 
            sigmal1 = dl1/IS1Z
            sigmal2 = dl2/IS2Z
            weights_lb[:,:,0] = sigmal0/(sigmal0 + sigmal1 + sigmal2)
            weights_lb[:,:,1] = sigmal1/(sigmal0 + sigmal1 + sigmal2)
            weights_lb[:,:,2] = sigmal2/(sigmal0 + sigmal1 + sigmal2)
            sigmar0 = dr0/IS0Z 
            sigmar1 = dr1/IS1Z
            sigmar2 = dr2/IS2Z
            weights_rt[:,:,0] = sigmar0/(sigmar0 + sigmar1 + sigmar2)
            weights_rt[:,:,1] = sigmar1/(sigmar0 + sigmar1 + sigmar2)
            weights_rt[:,:,2] = sigmar2/(sigmar0 + sigmar1 + sigmar2)
            
        return weights_lb, weights_rt
    
    def reconstruct(self, f: np.ndarray, projection: str) -> np.ndarray:
        """Computes values at both boundaries
         Parameters
        ----------
        f: np.ndarray
            data array, 2D array, shape = (Ny, Nx)
        projection: str
            axis of array used for computation
            """
        weights_lb, weights_rt = self.weights(f, projection)
        eno0L, eno0R = self.eno_K(0,f, projection)
        eno1L, eno1R = self.eno_K(1,f, projection)
        eno2L, eno2R = self.eno_K(2,f, projection)
        
        rL = weights_lb[:,:,0]*eno0L+weights_lb[:,:,1]*eno1L+weights_lb[:,:,2]*eno2L
        rR = weights_rt[:,:,0]*eno0R+weights_rt[:,:,1]*eno1R+weights_rt[:,:,2]*eno2R
        
        return rL, rR