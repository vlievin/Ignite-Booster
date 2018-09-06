import math
import numpy as np
import argparse
from dotmap import DotMap
import os
import warnings
import numbers
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
    
def initEngineState(engine,config):
    """
    initialize Engine state with config and parameters
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
        config (DotMap): config object
    Returns
        None
    """
    engine.state.config = config 
    engine.state.parameters = config.parameters.toDict()

def initAggregator(engine):
    """
    initialize aggregator (used to gather statistics)
    Args:
       engine (ignite.engine.Engine): PyTorch Ignite Engine 
    """
    engine.state.aggregator = {}
    
def setModel(engine,model):
    """
    set model object to engine
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
        model (torch.nn.module): model to be trained or evaluated
    """
    engine.model = model

def setOptimizer(engine,optimizer):
    """
    set optimizer to engine
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
        otpimizer (torch.otpim.optimizer): opitmizer to optimize the model with
    """
    engine.optimizer = optimizer
    
def updateParameters(engine,rules):
    """
    update parameters with a custom set of rules
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
        rules (dictionary of callable or None): a dictionary with parameter name as key and callable parametered by iteration gloabl step as value
    """
    if rules is not None:
        for k,v in ((k,v) for k,v in engine.state.parameters.items() if k in rules.keys()):
            engine.state.parameters[k] = rules[k](engine.state.iteration)
    
def updateAggregator(engine):
    """
    update the aggragator with the engine's output values. The output of the engine must be the Diagnostics objects
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
    """
    for k,v in engine.state.output.items():
        if k not in engine.state.aggregator.keys():
            engine.state.aggregator[k] = {}
        for k_,v_ in v.items():
            if k_ not in engine.state.aggregator[k].keys():
                engine.state.aggregator[k][k_] = []
            engine.state.aggregator[k][k_] += [v_]
        
def logResults(engine,epoch,writer):
    """
    log results using tensorboardX
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
    """
    for k,v in engine.state.aggregator.items():
        for k_,v_ in v.items():
            writer.add_scalar(str(k)+'/'+str(k_), np.mean(v_), epoch)
                       
def drawSample(engine,writer):
    """
    example: draw image sample to Tensorboard and display in Notebook
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
        writer (tensorboardX.SummaryWriter): tensorboardX writer
    """
    
    # generate image
    epoch = engine.state.epoch
    #torch.cuda.manual_seed_all(999) # seems dangerous to use here
    engine.model.eval()
    nrow = 8
    x = F.softmax( engine.model.sampleFromPrior(nrow**2), -1)
    x = make_grid(x[:,None,:,:],nrow=nrow).permute(0,2,1)
    writer.add_image('sample', x, epoch)
    x = x.permute(1,2,0).cpu().data.numpy()
    plt.imshow(x);plt.axis('off')
    plt.show()
    
def handle_exception(engine, e):
    """
    handle keyboard interrupt exception
    """
    if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
        engine.terminate()
        warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
    else:
        raise e
        
def print_progress(engine,timer):
    """
    print progress
    """
    print('# Iter. {} | Epoch {}/{} | loss {:.3f}| Time per batch: {:.3f}[s]'.format(engine.state.iteration, engine.state.epoch, engine.state.config.epochs, engine.state.output['loss']['loss'], timer.value()))
    timer.reset()
    
    
def init_model(model_class,hyperparameters):
    model = model_class(**hyperparameters)
    model.initialize_parameters()
    return model

def training_step(engine,batch):
    """
    perform one training iteration over the whole dataset and evaluate the model every 'validation_period' step for unsupervised models
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
        batch (torch.Tensor): batch of data
    Returns:
        results (nested dictionary of Number): a nested dictionary (see diagnostics in modules)
    """
    engine.model.train()
    batch = batch.to(engine.state.config.device)
    engine.optimizer.zero_grad()
    loss, diagnostics, _ = engine.model.getLoss(batch, **engine.state.parameters)
    loss.backward()
    engine.optimizer.step()
    itemize = lambda x:x if isinstance(x, numbers.Number) else x.item()
    return {k:{k_:itemize(v_) for k_,v_ in v.items()} for k,v in diagnostics.items()}

def validation_step(engine,batch):
    """
    perform one validation iteration over the whole dataset and evaluate the model every 'validation_period' step.
    Args:
        engine (ignite.engine.Engine): PyTorch Ignite Engine
        batch (torch.Tensor): batch of data
    Returns:
        results (nested dictionary of Number): a nested dictionary (see diagnostics in modules)
    """
    with torch.no_grad():
        engine.model.eval()
        batch = batch.to(engine.state.config.device)
        loss, diagnostics, _ = engine.model.getLoss(batch, **engine.state.parameters)
        itemize = lambda x:x if isinstance(x, numbers.Number) else x.item()
        return {k:{k_:itemize(v_) for k_,v_ in v.items()} for k,v in diagnostics.items()}
    
    
def validateAndLog(train_engine,eval_engine,test_loader,train_writer,test_writer):
    """
    validate model against test data and log results to TensorboardX
    Args:
       train_engine (ignite.engine.Engine): training engine
       eval_engine (ignite.engine.Engine): evaluation engine
       test_loader (torch.data.utils.DataLoader): test loader to run the evaluation engine with
       train_writer (tensorboardX.SummaryWriter): tensorboardX writer for training data
       test_writer (tensorboardX.SummaryWriter): tensorboardX writer for evaluation data
    """
    eval_engine.run(test_loader,1)
    epoch = train_engine.state.epoch
    logResults(train_engine,epoch,train_writer)
    logResults(eval_engine,epoch,test_writer) 
   
    
def run(model_class,train_data_loader,test_data_loader,config,rules=None):
    """
    perform the whole training process: initialize model, train and validate
    
    config and rules example:
        # run name
        run_name = str(uuid.uuid4())
        # define hyperparameters
        hyperparams = {'vocabulary_size':vocabulary_size, 
                         'n_sequence_steps':num_steps, 
                         'num_hidden_features':num_hidden_features, 
                         'last_conv_features':last_conv_features,
                         'memory_rows':memory_rows,
                         'memory_size':memory_size,
                         'memory_embedding_size_factor':memory_embedding_size_factor,
                         'ae_out': ae_out,
                         'loss_fn':loss_fn,
                         'aez': aez,
                         'learnable_prior':True,
                         'CPC': cpc}
        # define config object
        config = {
            "lr":lr,
            "epochs":epochs,
            "global_step":0,
            "epoch":0,
            "model_directory": os.path.join("/scratch/runs/booster/",run_name),
            "tensorboard_directory": os.path.join("/scratch/tensorboard/booster/",run_name),
            "hyperparameters":hyperparams, # model hyper parameters
            "parameters": {"beta": 0.0, 'tau':1.0},
            "deterministic_warmup": {"offset":100,"period":100},
            "tau_annealing":{"offset":100,"period":100, "min":0.3,"max":3.0},
            "device":"cuda",
        }
        # parameters update rules
        rules = {'beta': lambda step: stepFunction(step,config.deterministic_warmup.offset,config.deterministic_warmup.period),
                 'tau': lambda step: config.tau_annealing.max - (config.tau_annealing.max-config.tau_annealing.min) * stepFunction(step,config.deterministic_warmup.offset,config.deterministic_warmup.period)
                }
    
    Args:
        model (Booster.Booster_module class): booster object class
        train_data_loader (torch.utils.data.train_data_loader): training data loader
        test_data_loader (torch.utils.data.train_data_loader): testing data loader
        config (DotMap): config object file
        rules (dictionary of callables): parameters update rules
    Returns:
        model (Booster.Booster_module): trained model
    """
    config = DotMap(config)
    # initialize directory
    assert not os.path.exists(config.model_directory)
    os.makedirs(config.model_directory)
    # initialize the model
    model = init_model(model_class,config.hyperparameters.toDict())
    model.to(config.device)
    # optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    # tensorboard writers
    train_writer = create_summary_writer(model, train_data_loader, os.path.join(config.tensorboard_directory,'train'))
    test_writer = create_summary_writer(model, test_data_loader, os.path.join(config.tensorboard_directory,'test'))
    # ignite objects
    timer = Timer(average=True)
    checkpoint_handler = ModelCheckpoint(config.model_directory, config.model_name, save_interval=1, n_saved=1)
    # trainer
    trainer = Engine(training_step)
    trainer.add_event_handler(Events.STARTED, setModel, model)
    trainer.add_event_handler(Events.STARTED, setOptimizer, optimizer)
    trainer.add_event_handler(Events.STARTED, initEngineState, config)
    trainer.add_event_handler(Events.EPOCH_STARTED, initAggregator)
    trainer.add_event_handler(Events.ITERATION_STARTED, updateParameters, rules)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, updateAggregator)
    trainer.add_event_handler(Events.EXCEPTION_RAISED, handle_exception)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, print_progress,timer)
    # automatically adding handlers via a special `attach` method of `Timer` handler
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED) 
    # evaluator
    evaluator = Engine(validation_step)
    evaluator.add_event_handler(Events.STARTED, setModel, model)
    evaluator.add_event_handler(Events.STARTED, initEngineState, config)
    evaluator.add_event_handler(Events.EPOCH_STARTED, initAggregator)
    evaluator.add_event_handler(Events.ITERATION_COMPLETED, updateAggregator)
    evaluator.add_event_handler(Events.EXCEPTION_RAISED, handle_exception)
    evaluator.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler,to_save={ 'model': model})
    # define validation step
    trainer.add_event_handler(Events.EPOCH_COMPLETED,validateAndLog,evaluator,test_data_loader,train_writer,test_writer)
    # run
    trainer.run(train_data_loader, config.epochs)
    return model