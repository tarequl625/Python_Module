from scipy import integrate
import numpy as np
def f(x):
    return np.exp(np.arctan(x))
print(integrate.quad(f,0,1))