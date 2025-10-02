
import  numpy as np 

class orthogonalGrpDataset(object):
    def __init__(self, num_samples, dim, K1 = 2, K2 = 3, K3 = 4, C1 = 1, C2 = 2, C3 = 3, threshold = 0, sampling_type = "uniform"):
        """
        Generates synthetic dataset of size n x d.
        Let q = x.T @ x
        Equation of the polynomial: P(x) = k1*sin(c1*q) + k2 * sin(c2*q) ** 2 + k3 * cos(c3*q)
        """
        self.num_samples = num_samples
        self.dim = dim
        self.K1 = K1 
        self.K2 = K2 
        self.K3 = K3 
        self.C1 = C1 
        self.C2 = C2 
        self.C3 = C3 
        self.threshold = threshold
        self.sampling_type = sampling_type

    def poly_fn(self, x):
        x_norm = np.linalg.norm(x)

        p_val = self.K1*np.sin(self.C1*x_norm) + self.K2*np.sin(self.C3*x_norm) ** 2 + self.K3*np.cos(self.C3*x_norm)
        return int(p_val > self.threshold)

    def generate_data(self):
        """
        n: int --> no of data points
        d: int --> dimension of each data point
        c1: float --> constant in sin(c1*q)
        c2: float --> constant in sin(c2*q) ** 2
        c3: float --> constant in cos(c3*q)
        k1: float --> coefficient in sin(c1*q)
        k2: float --> coefficient in sin(c2*q) ** 2
        k3: float --> coefficient in cos(c3*q)
        """
        if self.sampling_type == "uniform" : 
            dataset = np.random.rand(self.num_samples, self.dim) # https://numpy.org/doc/2.1/reference/random/generated/numpy.random.rand.html
        else:
            dataset = np.random.randn(self.num_samples, self.dim)

        data_label = list(map(self.poly_fn, dataset))

        assert(len(data_label) == self.num_samples), "check the mapping function.. the len of label not equal to sample size"

        return (dataset, data_label)        
        
    
class PermutationGrpDataset(object):
    def __init__(self, num_samples, dim, l = 5, threshold = 0, sampling_type = "uniform"):
        """
        Generates synthetic dataset of size n x d.
        
        """
        self.num_samples = num_samples
        self.dim = dim
        self.threshold = threshold
        self.sampling_type = sampling_type
        self.l = l

    def poly_fn(self, x):
        hypo = 0.0
        for i in range(1, self.l+1):
            hypo += np.sin(x ** i).sum()
        
        return hypo


    def generate_data(self):
        """
        n: int --> no of data points
        d: int --> dimension of each data point
        c1: float --> constant in sin(c1*q)
        c2: float --> constant in sin(c2*q) ** 2
        c3: float --> constant in cos(c3*q)
        k1: float --> coefficient in sin(c1*q)
        k2: float --> coefficient in sin(c2*q) ** 2
        k3: float --> coefficient in cos(c3*q)
        """
        if self.sampling_type == "uniform" : 
            dataset = np.random.rand(self.num_samples, self.dim) # https://numpy.org/doc/2.1/reference/random/generated/numpy.random.rand.html
        else:
            dataset = np.random.randn(self.num_samples, self.dim)

        data_label = list(map(self.poly_fn, dataset))

        assert(len(data_label) == self.num_samples), "check the mapping function.. the len of label not equal to sample size"
        data_label = np.array(data_label)
        data_label = ((data_label - data_label.mean())>0)*1.0
        return (dataset, data_label)