import numpy as np
import scipy
import matplotlib.pyplot as plt


def data_and_indexes(tau: float, Lambda: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ny, Nx = Lambda.shape

    LambdaY = np.zeros((Ny, Nx))
    LambdaX = np.zeros((Ny, Nx))

    LambdaY[:-1, :] = 2 / (1 / Lambda[:-1, :] + 1 / Lambda[1:, :])
    LambdaX[:, :-1] = 2 / (1 / Lambda[:, :-1] + 1 / Lambda[:, 1:])

    index_grid = np.arange(Ny * Nx).reshape(Ny, Nx)

    row_top = index_grid[:-1, :].ravel()
    col_top = row_top + Nx
    data_top = LambdaY[:-1, :].ravel() / tau

    row_bottom = index_grid[1:, :].ravel()
    col_bottom = row_bottom - Nx
    data_bottom = LambdaY[:-1, :].ravel() / tau

    row_left = index_grid[:, :-1].ravel()
    col_left = row_left + 1
    data_left = (LambdaX[:, :-1] * tau).ravel()

    row_right = index_grid[:, 1:].ravel()
    col_right = row_right - 1
    data_right = (LambdaX[:, :-1] * tau).ravel()

    row_center = index_grid.ravel()
    col_center = row_center
    m_center = np.zeros(Ny * Nx, dtype=np.float64)
    m_center[:-Nx] -= (LambdaY[:-1, :] / tau).ravel()
    m_center[Nx:] -= (LambdaY[:-1, :] / tau).ravel()
    m_center[:-1] -= (LambdaX[:, :] * tau).ravel()[:-1]
    m_center[1:] -= (LambdaX[:, :] * tau).ravel()[:-1]


    data = np.concatenate((data_top, data_bottom, data_left, data_right, m_center))
    row = np.concatenate((row_top, row_bottom, row_left, row_right, row_center))
    col = np.concatenate((col_top, col_bottom, col_left, col_right, col_center))

    return data, row, col


class PoissonSolver():
    def __init__(self, init_dict: dict):
        self.Nx = init_dict['Nx']
        self.Ny = init_dict['Ny']
        self.L = init_dict['L']
        self.H = init_dict['H']
        self.dx = self.L/self.Nx
        self.dy = self.H/self.Ny
        self.tau = self.dy/self.dx
        self.meshP_X, self.meshP_Y = np.meshgrid(np.linspace(self.dx/2, self.L-self.dx/2, self.Nx), np.linspace(self.dy/2, self.L-self.dy/2, self.Ny))
        self.meshVx_X, self.meshVx_Y = np.meshgrid(np.linspace(0, self.L, self.Nx+1), np.linspace(self.dy/2, self.L-self.dy/2, self.Ny))
        self.meshVy_X, self.meshVy_Y = np.meshgrid(np.linspace(self.dx/2, self.L-self.dx/2, self.Nx), np.linspace(0, self.H, self.Ny+1))
    

    def matrix(self, Lambda: np.ndarray) -> scipy.sparse.csc_matrix:
        """Linearizes div(mobility*grad P) for given mobility\n
        Parameters
        ----------
        Lambda: np.ndarray
            Mobility, 2D array, shape = (Ny, Nx)
        """
        data, row_indexes, col_indexes = data_and_indexes(self.tau, Lambda)
        return scipy.sparse.csc_matrix((data, (row_indexes, col_indexes)), shape=(self.Nx*self.Ny, self.Nx*self.Ny))
    
    def solve_sparse(self,
                     Lambda: np.ndarray,
                     rhs_matrix: np.ndarray) -> np.ndarray:
        """Computes pressure field for given mobility and sources matrix using scipy  solver\n
        Parameters
        ----------
        Lambda: np.ndarray
            mobility, 2D array, shape = (Ny, Nx)
        rhs_matrix: np.ndarray
            matrix of sources and boundary conditions, 2D array, shape = (Ny, Nx)    
            """
        rhs_vec = rhs_matrix.reshape(-1)
        lhs_matrix = self.matrix(Lambda)
        
        solution_vec = scipy.sparse.linalg.spsolve(lhs_matrix, rhs_vec)
        solution_matrix = solution_vec.reshape(self.Ny, self.Nx)
        return solution_matrix
    

    def solve_CG(self,
                 Lambda: np.ndarray,
                 rhs_matrix: np.ndarray,
                 p0: np.ndarray,
                 eps: float=10**(-4)) -> np.ndarray:
        """Computes pressure field for given mobility and sources matrix using CG\n
        Parameters
        ----------
        Lambda: np.ndarray
            mobility, 2D array, shape = (Ny, Nx)
        rhs_matrix: np.ndarray
            matrix of sources and boundary conditions, 2D array, shape = (Ny, Nx)    
        """
        rhs_vec = rhs_matrix.reshape(-1)
        lhs_matrix = self.matrix(Lambda)
        
        x_j = 0+p0.reshape(-1)
        
        r_j = rhs_vec - lhs_matrix@x_j 
        p_j = r_j + 0
        r_j_norm_square = np.sum(np.square(r_j))

        i = 0
        while r_j_norm_square >= eps and i<10**4:
            a_j = r_j_norm_square/((lhs_matrix@p_j).T@r_j) 
            b_j = 1/r_j_norm_square
            x_j += a_j*p_j # j+1
            r_j -= a_j*(lhs_matrix@p_j) # j+1

            r_j_norm_square = np.sum(np.square(r_j)) # j+1
            b_j = r_j_norm_square*b_j
            p_j = r_j + b_j*p_j # j+1
            i+=1
        
        solution_matrix = x_j.reshape(self.Ny, self.Nx)
        return solution_matrix #- np.min(solution_matrix)
    