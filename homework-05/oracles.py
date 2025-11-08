import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
         return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
         # вычисляем z, те скалярное произведение каждой строки A на x, умноженное на соответствующую метку b
         z = self.matvec_Ax(x) * self.b  
         # считаем среднее значение функции потерь логистической регрессии: 
         # log(1 + exp(-z)) + L2-регуляризация
         return np.mean(np.log1p(np.exp(-z))) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        # TODO: Implement
         z = self.matvec_Ax(x) * self.b
         sig = expit(-z) # логистическая сигмоидная функция
         grad_loss = -self.matvec_ATx(sig * self.b) / len(self.b) # градиент функции потерь, где m -количество объектов
         return grad_loss + self.regcoef * x # градиент L2-регуляризации

    def hess(self, x):
        # TODO: Implement
         z = self.matvec_Ax(x) * self.b
         sig = expit(-z)
         s = sig * (1 - sig) # вектор из m элементов для диагональной матрицы (участвует в выражении для гессиана)
         # гессиан функции потерь + гессиан L2-регуляризации 
         return self.matmat_ATsA(s) / len(self.b) + self.regcoef * np.eye(len(x))


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
         Ax = self.matvec_Ax(x) # предварительно вычисленный Ax Ad 
         Ad = self.matvec_Ax(d) 
         # скалярное произведение каждой строки A на (x + alpha * d), умноженное на соответствующую метку b
         z = (Ax + alpha * Ad) * self.b 
         return np.mean(np.log1p(np.exp(-z))) + 0.5 * self.regcoef * np.dot(x + alpha * d, x + alpha * d)

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        Ax = self.matvec_Ax(x)
        Ad = self.matvec_Ax(d)
        z = 1 / (1 + np.exp(-(Ax + alpha * Ad)))
        sig = expit(-z)
        # вычисляем градиент по направлению d
        grad_dir = np.dot(self.matvec_ATx(z - self.b), d) + self.reg_coef * np.dot(x + alpha * d, d)
        # проецируем градиент на направление d
        return grad_dir


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A @ x  # TODO: Implement
    matvec_ATx = lambda x: A.T @ x  # TODO: Implement

    def matmat_ATsA(s):
        # TODO: Implement
        A.T @ (np.diag(s) @ A) 
        # умножаем каждую строку матрицы A на соответствующий элемент s
        return A.T @ (np.diag(s) @ A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
    n = len(x) 
    # матрицa единичных векторов, умноженных на eps
    # каждая колонка  x + eps * e
    X = x + eps * np.eye(n) 
    # считаем значения функции f(x + eps * e_i) для всех координат
    fx_eps = np.array([func(X[:, i]) for i in range(n)])
    return (fx_eps - func(x)) / eps 


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian
    n = len(x)
    # вычисляем f(x)
    f_x = func(x)
    # f(x + eps * e_i + eps * e_j) для всех пар (i, j)
    f_ij = np.array([[func(x + eps*np.eye(n)[:,i] + eps*np.eye(n)[:,j]) 
                      for j in range(n)] for i in range(n)])
    # f(x + eps * e_i) сдвиги по каждой оси
    f_i = np.array([func(x + eps*np.eye(n)[:,i]) for i in range(n)])
    # формула конечных разностей второго порядка
    H = f_ij - f_i[:, np.newaxis] - f_i[np.newaxis, :] + f_x
    # делим на eps**2, чтобы получить приближение второй производной
    return H / eps**2
