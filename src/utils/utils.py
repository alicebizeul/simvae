"""
Module Name: utils.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
"""
import torch
import os
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
import time, datetime
from torchvision.utils import save_image

def get_paths(args):
    path_data = args["path_data"]
    path_project = args["path_output"]
    path_output = path_project + "/outputs/"
    path_graph = path_project + "/lossGraphs/"
    os.makedirs(path_output, exist_ok=True)
    os.makedirs(path_graph, exist_ok=True)
    return path_data, path_project, path_output, path_graph

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind1, ind2 = linear_sum_assignment(w.max() - w)         # fix this 
    return sum([w[i, j] for i, j in zip(list(ind1),list(ind2))]) * 1.0 

def plus_equals_nan(total, new_val):
    if np.isnan(total): return new_val
    return total + new_val

def plus_equals_nan_arr(total, new_val):
    if len(total) != 0:
        if np.isnan(total[0]): return new_val
        else: return [sum(i) for i in zip(total,new_val )]  
    return new_val

def min_sec(duration):
    secs = datetime.timedelta(seconds=round(duration)).seconds
    mins = (secs % 3600) // 60
    secs = (secs % 60)
    return f'{mins:2d}:{secs:02d}'

def set_training_state(model, model_status, head_status):
    if model_status == 0: 
        model.eval()
    else:
        model.train()

    if head_status == 0:
        for m in model.evaluator.values(): m.eval()       
    else:
        for m in model.evaluator.values(): m.train()

def reshape_data(data, labels, num_sim, layers_type):
    if (num_sim > 1) or (len(data.shape) > 4):
        data   = data.view(-1, *data.shape[2:]) 
        duplication = [1] + [num_sim] + [1 for _ in range(len(labels.shape)-1)]
        view = [-1] if len(labels.shape)==1 else [-1,labels.shape[-1]]
        labels = labels.unsqueeze(1).repeat(*duplication).view(*view)
    if not layers_type: 
        data = data.view(data.shape[0], -1)  
    return data, labels

def save_outputs(dec, data, fn_output, num_rows=8, layers_type=False, shape=(1,32,32)):
    to_save = []
    if dec is not None:
        mu_x, _ = dec
        to_save += [("data", data), ("recon", mu_x)]
        if layers_type:
            images = torch.cat( [ y.view(-1, *mu_x.shape[-3:]).cpu()[:num_rows] for (_, y) in to_save ] )
        else: 
            images = torch.cat( [ y.view(-1, *shape).cpu()[:num_rows] for (_, y) in to_save])
        save_image(images, fn_output + ".png", nrow=num_rows)

def save_clustering(model, fn_output):
    c_mus  = model.evaluator["clustering"].params["c_means"].detach().cpu()
    c_vars = model.evaluator["clustering"].params["c_vars"].detach().cpu()
    torch.save({"clust_cent": c_mus, "clust_vars": c_vars}, fn_output + ".pt")
