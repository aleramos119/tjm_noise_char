
#%%
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class transform:
    def __init__(self, displacement, range):

        self.displacement = displacement
        self.range = range

    def displace(self, x):
        return x - self.displacement

    def scale(self, x):
        return x / self.range
    
    def transform(self, x):
        x = self.displace(x)
        x = self.scale(x)
        return x
    
    def undisplace(self, x):
        return x + self.displacement
    
    def unscale(self, x):
        return x * self.range

    def untransform(self, x):
        x = self.unscale(x)
        x = self.undisplace(x)
        return x
    


class normalize(transform):

    def __init__(self, x_input):

        x=np.array(x_input)

        self.displacement = x.min(axis=0)
        self.range = x.max(axis=0) - x.min(axis=0)

        zero_range_mask = self.range == 0
        self.range[zero_range_mask] = np.abs(x).max(axis=0)[zero_range_mask]
        

class standardize(transform):

    def __init__(self, x_input):

        x=np.array(x_input)

        self.displacement = x.mean(axis=0)
        self.range = x.std(axis=0)

        zero_range_mask = self.range == 0
        self.range[zero_range_mask] = np.abs(x).max(axis=0)[zero_range_mask]




class derivative_transform(transform):

    def __init__(self, x_input, y_input):

        x=np.array(x_input)
        y=np.array(y_input)

        d=x.shape[1]

        self.x_displacement = x.min(axis=0)
        self.x_range = x.max(axis=0) - x.min(axis=0)

        self.x_range[self.x_range == 0] = np.abs(x).max(axis=0)[self.x_range == 0]


        self.y_scale = y[:,0].std(axis=0)

        if self.y_scale == 0:
            self.y_scale = np.abs(y[:,0]).max(axis=0)

        self.y_displacement = np.append(y[:,0].mean(axis=0),np.zeros(d))
        self.y_range = np.append(self.y_scale, self.y_scale/self.x_range)



    def displace(self, x, displacement):
        return x - displacement

    def scale(self, x, range):
        return x / range
    
    def undisplace(self, x, displacement):
        return x + displacement
    
    def unscale(self, x, range):
        return x * range
    
    def transform_x(self, x):
        x = self.displace(x, self.x_displacement)
        x = self.scale(x, self.x_range)
        return x

    def transform_y(self, y):
        y = self.displace(y, self.y_displacement)
        y = self.scale(y, self.y_range)
        return y

    def untransform_x(self, x):
        x = self.unscale(x, self.x_range)
        x = self.undisplace(x, self.x_displacement)
        return x
    
    def untransform_y(self, y):
        y = self.unscale(y, self.y_range)
        y = self.undisplace(y, self.y_displacement)
        return y
    

    def transform(self, x, y):
        x = self.transform_x(x)
        y = self.transform_y(y)
        return x, y

    def untransform(self, x, y):
        x = self.untransform_x(x)
        y = self.untransform_y(y)
        return x, y

    def scale_std(self, y):
        y = self.scale(y, self.y_range)
        return y

    def unscale_std(self, y):
        y = self.unscale(y, self.y_range)
        return y
    
