import pandas as pd
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import quad


class LogisticSymmetric:
    """Simple 4-parameter logistic function.
    
    Parameters
    -----------
    U: Upper asymptote
    L: Lower asymptote
    k: maximum growth rate
    x0: x-location of maximum growth rate
    """
    def __init__(self, x, y, p0=None, lower=None, upper=None, maxfev=None):
        self.x = x
        self.y = y
        
        if maxfev is None:
            self.maxfev = 1000
        else:
            self.maxfev = maxfev
        
        if p0 is None:
            self.p0 = (max(self.y), min(self.y), 0.01, 1.0)
        else:
            self.p0 = p0
            
        if lower is None:
            self.lower = (-np.inf, -np.inf, 0.0001, min(self.x))
        else:
            self.lower = lower
            
        if upper is None:
            self.upper = (np.inf, np.inf, np.inf, np.max(self.x))
        else:
            self.upper = upper
            
        self.popt = None
        self.pcov = None
        
    def func(self, x, U, L, k, x0):
        """Compute function."""
        return L + (U-L) / (1 + np.exp(-k*(x-x0)))
        
        
    def fit(self):
        """Perform least-squares fit on x and y data."""
        popt, pcov = curve_fit(self.func, self.x, self.y, p0=self.p0, bounds=(self.lower, self.upper), maxfev=self.maxfev)
        self.popt = tuple(popt)
        self.pcov = pcov
    
    def symbolic_function(self):
        """Symblic representation of the function."""
        x, U, L, k, x0 = sp.symbols('x U L k x0')
        f = L + (U - L) / (1 + sp.exp(-k*(x-x0)))
        return f
    
    def max_growth_rate(self):
        """Return the maximum growth rate."""
        U, L, k, x0 = self.popt
        return k
    
    def area_under_curve(self, a=None, b=None):
        """Compute area under the curve, bounded by the minimum value of y.
        
        More information: https://www.khanacademy.org/math/ap-calculus-ab/ab-applications-of-integration-new/ab-8-4/v/area-between-curves"""
        if a is None:
            a = min(self.x)
        if b is None:
            b = max(self.x)
        
        I_1 = (b - a) * min(self.y)
        I_2, _ = quad(self.func, args=self.popt, a=a, b=b)
        return abs(I_1 - I_2)
    
        


class RichardsModified:
    """Modified 6-parameter Richard's function for greater asymmetric flexibility.
    
    Parameters
    ----------
    U: Upper asymptote
    L: Lower asymptote
    k1: first rate parameter
    k2: second rate parameter
    v: asymmetry parameter
    x0: x-shift
    """
    def __init__(self, x, y, p0=None, lower=None, upper=None, maxfev=None):
        self.x = x
        self.y = y
        
        if maxfev is None:
            self.maxfev = 1000
        else:
            self.maxfev = maxfev
        
        if p0 is None:
            self.p0 = (max(self.y), min(self.y), 0.01, 0.01, 1.0, 3.0)
        else:
            self.p0 = p0
            
        if lower is None:
            self.lower = (-np.inf, -np.inf, 0.0001, 0.0001, 0.0001, min(self.x))
        else:
            self.lower = lower
            
        if upper is None:
            self.upper = (np.inf, np.inf, np.inf, np.inf, np.inf, max(self.x))
        else:
            self.upper = upper
            
        self.popt = None
        self.pcov = None
        
    def func(self, x, U, L, k1, k2, v, x0):
        """Compute function."""
        return L + ( (U - L)/ ((1 + np.exp(-k1*(x-x0)))**1/v * (1 + np.exp(-k2*(x-x0)))**1/v) )
        
        
    def fit(self):
        """Perform least-squares fit on x and y data."""
        popt, pcov = curve_fit(self.func, self.x, self.y, p0=self.p0, bounds=(self.lower, self.upper), maxfev=self.maxfev)
        self.popt = tuple(popt)
        self.pcov = pcov
        
    def dfunc(self, x, U, L, k1, k2, v, x0):
        """Compute first derivative."""
        alpha = k1*(x-x0)
        beta = k2*(x-x0)
        term1 = k1 * (1+np.exp(-alpha))**(-1/v) * (1+np.exp(-beta))**(-1/v) * np.exp(-alpha) / (1+np.exp(-alpha))
        term2 = k2 * (1+np.exp(-alpha))**(-1/v) * (1+np.exp(-beta))**(-1/v) * np.exp(-beta) / (1+np.exp(-beta))
        return (U - L) / v * (term1 + term2)
    
    def ddfunc(self, x, U, L, k1, k2, v, x0):
        """Compute second derivative."""
        phi = np.exp(-k1*(x-x0))
        psi = np.exp(-k2*(x-x0))
        gamma = 1 + phi
        omega = 1 + psi
        exp = -1/v
        PRE = gamma**exp * omega**exp * (U - L) / v
        T1 = -k1**2 * phi / gamma
        T2 = k1**2 * phi**2 / gamma**2
        T3 = k1**2 * phi**2 / v / gamma**2
        T4 = 2*k1*k2 * phi * psi / v / gamma / omega
        T5 = -k2**2 * psi / omega
        T6 = k2**2 * psi**2 / omega**2
        T7 = k2**2 * psi**2 / v / omega**2
        return PRE * (T1 + T2 + T3 + T4 + T5 + T6 + T7)
    
    
    def symbolic_function(self):
        """Symbolic representation of the function."""
        x, U, L, k1, k2, v, x0 = sp.symbols('x U L k1 k2 v x0')
        f = L + ( (U-L) / ( (1 + sp.exp(-k1*(x-x0)))**(1/v) * (1 + sp.exp(-k2*(x-x0)))**(1/v) ) )
        return f
    
    def symbolic_first_derivative(self):
        """Symbolic representation of first derivative."""
        x = sp.symbols('x')
        return sp.diff(self.symbolic_function(), x)
    
    def symbolic_second_derivative(self):
        """Symbolic representation of second derivative."""
        x = sp.symbols('x')
        dd = self.symbolic_first_derivative()
        return sp.diff(dd, x)
    
    def max_growth_rate(self):
        """Compute the maximum growth rate using derivatives."""
        U, L, k1, k2, v, x0 = self.popt
        root = fsolve(self.ddfunc, args=self.popt, x0=x0)
        return self.dfunc(root, *self.popt)[0]
    
    def area_under_curve(self, a=None, b=None):
        """Compute area under the curve, bounded by the minimum value of y.
        
        More information: https://www.khanacademy.org/math/ap-calculus-ab/ab-applications-of-integration-new/ab-8-4/v/area-between-curves"""
        if a is None:
            a = min(self.x)
        if b is None:
            b = max(self.x)
        
        I_1 = (b - a) * min(self.y)
        I_2, _ = quad(self.func, args=self.popt, a=a, b=b)
        return abs(I_1 - I_2)
    
    
class Richards:
    """Generalized logistic function, aka Richard's function.
    
    Parameters
    -----------
    U: Upper asymptote
    L: Lower asymptote
    k: maximum growth rate
    v: asymmetry parameter
    """
    def __init__(self, x, y, p0=None, lower=None, upper=None, maxfev=None):
        self.x = x
        self.y = y
        
        if maxfev is None:
            self.maxfev = 1000
        else:
            self.maxfev = maxfev
        
        if p0 is None:
            self.p0 = (max(self.y), min(self.y), 0.01, 1.0)
        else:
            self.p0 = p0
            
        if lower is None:
            self.lower = (-np.inf, -np.inf, 0.0001, 0.0001)
        else:
            self.lower = lower
            
        if upper is None:
            self.upper = (np.inf, np.inf, np.inf, np.inf)
        else:
            self.upper = upper
            
        self.popt = None
        self.pcov = None
        
    def func(self, x, U, L, k, v):
        """Compute function."""
        return L + ( (U - L)/ ((1 + np.exp(-k*x))**1/v) )
        
        
    def fit(self):
        """Perform least-squares fit on x and y data."""
        popt, pcov = curve_fit(self.func, self.x, self.y, p0=self.p0, bounds=(self.lower, self.upper), maxfev=self.maxfev)
        self.popt = tuple(popt)
        self.pcov = pcov
    
    
    def symbolic_function(self):
        """Symbolic representation of the function."""
        x, U, L, k, v = sp.symbols('x U L k v')
        return L + (U - L) / (1 + sp.exp(-k*x))**(1/v)
    
    def max_growth_rate(self):
        """Return the maximum growth rate."""
        U, L, k, v = self.popt
        return k
    
    def area_under_curve(self, a=None, b=None):
        """Compute area under the curve, bounded by the minimum value of y.
        
        More information: https://www.khanacademy.org/math/ap-calculus-ab/ab-applications-of-integration-new/ab-8-4/v/area-between-curves"""
        if a is None:
            a = min(self.x)
        if b is None:
            b = max(self.x)
        
        I_1 = (b - a) * min(self.y)
        I_2, _ = quad(self.func, args=self.popt, a=a, b=b)
        return abs(I_1 - I_2)
    