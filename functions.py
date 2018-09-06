import numpy as np
from torch.autograd import Variable


def stepFunction(step,offset,period):
    """
    step function varying from 0 to 1 during 'period' steps after 'offset' steps
    Args:
        step (int): step value
        offset (int): number of steps after which the value begins to increase from 0 to 1
        period (int): number of steps used to increase the value from 0 to 1
    Returns:
        value (float): output value
    """
    x = (float(step)-float(offset)) / float(period)
    return float(np.clip(x,0,1))