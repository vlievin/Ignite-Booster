"""
This file described the PyTorch modules involved in the Booster project. It contains the 'Booster_module' abastract class that defines modules compatible with Booster

"""

from torch import nn

class Booster_module(nn.Module):
    """
    The Booster_moduel module is an abstract class defining the objects compatible with the Booster module
    """
    def initialize_parameters(self):
        """
        initialize parameters
        """
        raise NotImplementedError
    def getLoss(self,x,y,**kwargs):
        """
        compute loss and statistics given inputs x and y and extra parameters
        Args:
            x: input x
            y: label y (potentially None)
            kwargs: potential addtional parameters
        Returns:
            a tuple (loss,diagnostics,data) where loss is the loss function, diagnostics a nested dictionary containing statistics and data the input data (x and y)
        """
        raise NotImplementedError