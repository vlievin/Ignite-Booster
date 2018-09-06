"""
utilitary functions for the Booster project
"""

import warnings
from tensorboardX import SummaryWriter

def create_summary_writer(model, data_loader, log_dir):
    """
    create tensorboardX summary writer (includes the graph definition)
    Args:
        model (Torch.nn.module): Pytorch model to extract graph from
        data_loader (orch.utils.data.DataLoader): PyTorch data loader
        log_dir (str): path to logging directory
    Retursn 
        writer (tensorboardX.SummaryWriter): tensorboardX writer
    """
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        warnings.warn("Failed to save model graph: {}".format(e))
    return writer


