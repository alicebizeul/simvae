"""
Module Name: main.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, math
from tqdm import tqdm

import src.utils.utils
import src.model.models
from src.data.data_loader import get_loader
from src.utils.options import args, fn_attr_str, title_attribs
from src.utils.plotting import loss_plot

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def train( model, dataloader, device, optimizer, binarise, fn_output, num_sim=1, schedule=("train",1,0), eval_optim=None, eval_optim_mlp=None, shape=(1,32,32),first_eval=False):
    phase, model_status, head_status, = schedule
    src.utils.utils.set_training_state(model, model_status, head_status)

    num_x, bs = len(dataloader.dataset), dataloader.batch_size
    num_x_epoch, num_x_batch = 1, bs*num_sim
    num_batch = int(num_x // bs)
    epoch_loss, epoch_acc_pred, epoch_loss_reg = np.nan, np.nan, np.nan
    epoch_recon, epoch_entropy, epoch_prior, epoch_infonce, epoch_ce, metrics = [], [], [], [], [], {"diff_means":[],"sigma_z_x":[],"diff_sigma":[]}

    got_BaseVAE = isinstance(model, src.model.models.BaseVAE)
    pbar = tqdm_enumerate(dataloader, total=num_batch, message=f"  T-eval: {head_status:<6d} F-eval: {(phase=='evaluate')*1} ")
    tqdm_active = isinstance(pbar,tqdm)
    print("TQDM",tqdm_active,pbar)
    optimizer.zero_grad()

    for i, (data, labels) in pbar:
        data, labels = data.to(device), labels.to(device)

        if first_eval or model_status == 1 or (model_status == 0 and head_status == 1 and phase=="train"):
            data,   labels = src.utils.utils.reshape_data(data, labels, num_sim, layers_type=(model.resnet18 ))
        
        if phase in ["train"] and model_status == 1:
            batch_loss = None
            dec  = None   
            enc_c, dec, c = model(data)  
            recon, entropy, prior = model.loss(data, enc_c, dec, c)
            batch_loss = recon + prior + entropy
            temp_recon, temp_entropy, temp_prior = recon.item(), entropy.item(), prior.item()

            epoch_recon.append(recon.item())
            epoch_entropy.append(-entropy.item())
            epoch_prior.append(prior.item())

            temp_loss = batch_loss.item()
            epoch_loss = src.utils.utils.plus_equals_nan(epoch_loss, temp_loss)

            batch_loss.backward()     
            optimizer.step()
            optimizer.zero_grad()

            if fn_output and (i == (num_batch - 1)):
                src.utils.utils.save_outputs(dec,data, fn_output, shape=shape)

            if tqdm_active: 
                message = f"{temp_recon/num_x_batch:8.2f} {temp_entropy/num_x_batch:6.2f} {temp_prior/num_x_batch:6.2f}" if got_BaseVAE else f"{temp_loss/num_x_batch:7.2f}"
                pbar.set_description(f"  T-loss {message}")


        if head_status == 1:
            if len(labels.shape) == 2: keep_index = (torch.sum(1*(labels>0.),axis=-1)==1)                               
            else: keep_index=torch.ones(labels.shape).bool()

            num_x_epoch+=len(keep_index) if num_x_epoch != 1 else num_x_epoch-1+len(keep_index)            
            if first_eval or phase =="train": 
                with torch.no_grad():
                    model.encoder.to(device)
                    _, c, = model.sample_z(data[keep_index] if len(labels.shape) == 2 else data) 
                    model.encoder.to("cpu")

                if first_eval: dataloader.dataset.set_object(c,labels[keep_index])
            else:
                c = data[keep_index] if len(labels.shape) == 2 else data

            temp_acc, temp_loss = eval_train(optimizer_lin=eval_optim,optimizer_mlp=eval_optim_mlp, model=model, c=c, labels=torch.argmax(labels,dim=-1)[keep_index] if len(labels.shape)==2 else labels, final_eval=phase=="evaluate", ) 
            epoch_acc_pred = src.utils.utils.plus_equals_nan(epoch_acc_pred, temp_acc["logreg"])
            epoch_loss_reg = src.utils.utils.plus_equals_nan(epoch_loss_reg, temp_loss)

            if tqdm_active: 
                pbar.set_description(f"  E-loss {temp_acc['logreg']/num_x_batch:8.2f} {temp_loss/num_x_batch:6.2f}                 ")

            if phase == "evaluate" and first_eval:
                for m in model.evaluator.keys():
                    if m in ["knn","clustering"]:
                        model.evaluator[m].e_step(c,torch.argmax(labels,dim=-1)[keep_index] if len(labels.shape) == 2 else labels )


    return {"train":(epoch_loss / num_batch), "evaluate":(epoch_loss_reg / num_x_epoch), "recon":(epoch_recon ), "entropy":(epoch_entropy), "prior":(epoch_prior), "infonce":(epoch_infonce), "ce":(epoch_ce)}, {"logreg":(epoch_acc_pred / num_x_epoch),"metrics":metrics}


""" VALIDATION =========================================================== """

def validate( model, dataloader, device, num_sim=1, k=1, schedule=("train",1,0), first_eval=False):
    phase, model_status, head_status, = schedule
    model.eval()
    num_x, bs = len(dataloader.dataset), dataloader.batch_size
    num_batch = num_x // bs
    num_x_epoch, num_x_batch = 1, bs*num_sim
    epoch_loss, epoch_acc_pred, epoch_acc_pred_mlp, epoch_acc_clust, epoch_nmi_clust, epoch_ari_clust =  (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    epoch_acc_knn = np.nan * np.ones(15)
    epoch_recon, epoch_entropy, epoch_prior, = np.nan, np.nan, np.nan
    got_BaseVAE = isinstance(model, src.model.models.BaseVAE)

    pbar = tqdm_enumerate(dataloader, total=num_batch, message=f"  Val-Fin-Eval: {(phase=='evaluate')*1}")
    tqdm_active = isinstance(pbar,tqdm)
    with torch.no_grad():
        for i, (data, labels) in pbar:
            data, labels = data.to(device), labels.to(device)

            if first_eval or (model_status == 0 and head_status == 1 and phase=="train"):
                data,   labels = src.utils.utils.reshape_data(data, labels, num_sim, layers_type=(model.resnet18))
            validating = True

            if phase in ["train"]:
                batch_loss, acc_pred, dec = np.nan, np.nan, None

                if model_status == 1: 
                    enc_c, dec, c = model(data)
                    recon, entropy, prior = model.loss(data, enc_c, dec, c )
                    batch_loss = recon.item() + prior.item() + entropy.item()

                    epoch_recon   = src.utils.utils.plus_equals_nan(epoch_recon,   recon.item())
                    epoch_entropy = src.utils.utils.plus_equals_nan(epoch_entropy, -entropy.item())
                    epoch_prior   = src.utils.utils.plus_equals_nan(epoch_prior,   prior.item())
                    epoch_loss    = src.utils.utils.plus_equals_nan(epoch_loss, batch_loss)
                    message = f"{recon.item()/num_x_batch:8.2f} {entropy.item()/num_x_batch:6.2f} {prior.item()/num_x_batch:6.2f}"

                if tqdm_active and validating: 
                    pbar.set_description(f"  V-loss {message}" + (f" Acc {acc_pred/num_x_batch:5.2f}" if not np.isnan(acc_pred) else " "*10))

            elif phase == "evaluate":  

                if len(labels.shape) == 2: keep_index = (torch.sum(1*(labels>0.),axis=-1)==1)                                    # final evaluation: only want evaluation metrics:
                else: keep_index=torch.ones(labels.shape).bool()

                if first_eval: 
                    model.encoder.to(device)
                    _, c, = model.sample_z(data[keep_index] if len(labels.shape) == 2 else data) 
                    model.encoder.to("cpu")
                    dataloader.dataset.set_object(c,labels[keep_index])
                else: 
                    c=data[keep_index] if len(labels.shape) == 2 else data

                num_x_epoch+=len(keep_index) if num_x_epoch != 1 else num_x_epoch-1+len(keep_index)
                temp_acc      = eval_valid(model=model, c=c, labels=torch.argmax(labels,dim=-1)[keep_index] if len(labels.shape) == 2 else labels)
                acc_pred      = temp_acc["logreg"]
                acc_pred_mlp  = temp_acc["mlp"]
                acc_clust     = temp_acc["clustering"]
                acc_clust_nmi = temp_acc["clustering_nmi"]
                acc_clust_ari = temp_acc["clustering_ari"]
                acc_knn       = temp_acc["knn"]

                epoch_acc_pred      = src.utils.utils.plus_equals_nan(epoch_acc_pred,  acc_pred)
                epoch_acc_pred_mlp  = src.utils.utils.plus_equals_nan(epoch_acc_pred_mlp,  acc_pred_mlp)
                epoch_acc_clust     = src.utils.utils.plus_equals_nan(epoch_acc_clust, acc_clust)
                epoch_nmi_clust     = src.utils.utils.plus_equals_nan(epoch_nmi_clust, acc_clust_nmi)
                epoch_ari_clust     = src.utils.utils.plus_equals_nan(epoch_ari_clust, acc_clust_ari)
                epoch_acc_knn       = src.utils.utils.plus_equals_nan_arr(epoch_acc_knn,acc_knn)
                if tqdm_active: pbar.set_description(f"  V-acc {acc_pred/num_x_batch:6.2f}  Clust {acc_clust/num_batch:6.2f} KNN {max(acc_knn)/num_x_batch:6.2f}")
            
            else:
                raise NotImplementedError(f"(main) validate-2: this shouldn't happen {phase:s}")           
    return {"train":(epoch_loss / num_x_epoch), "recon":(epoch_recon / num_x_epoch), "entropy":(epoch_entropy / num_x_epoch), "prior":(epoch_prior / num_x_epoch)}, {"logreg":(epoch_acc_pred / num_x_epoch),"mlp":(epoch_acc_pred_mlp / num_x_epoch),"clustering":(epoch_acc_clust / num_x_epoch),"clustering_nmi":(epoch_nmi_clust/num_batch),"clustering_ari":(epoch_ari_clust/num_batch),"knn":(max(epoch_acc_knn)/num_x_epoch)}


""" EVALUATION =========================================================== """
def get_preds(classifier, z, y):
    
    logits = classifier(z)
    if logits.shape[-1]==1:
        preds = torch.squeeze(1*(logits > 0.5))
    else:
        preds  = logits.argmax(-1)
    acc    = (preds == y).sum().item() 
    return logits, preds, acc

def eval_train(optimizer_lin, optimizer_mlp, model, c, labels, final_eval=False):
    acc = {}
    c = c.detach()
    optimizer_lin.zero_grad()

    classifier = model.evaluator["logreg"]
    logits, _, acc["logreg"] = get_preds(classifier, c, labels)
    if torch.unique(labels).shape[0] > 2:
        classif_loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)          
    else:
        classif_loss = nn.BCELoss(reduction='sum')(torch.squeeze(logits), labels.to(torch.float) )   
    classif_loss.backward()
    optimizer_lin.step()

    # clustering - gmm
    if final_eval:
        optimizer_mlp.zero_grad()
        classifier = model.evaluator["mlp"]
        logits, _, acc["mlp"] = get_preds(classifier, c, labels)
        if torch.unique(labels).shape[0] > 2:
            classif_loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)           
        else:
            classif_loss = nn.BCELoss(reduction='sum')(torch.squeeze(logits), labels.to(torch.float) )   
        classif_loss.backward()
        optimizer_mlp.step()
            
    return acc, classif_loss.item()

def eval_valid(model, c, labels):
    acc = {}

    # log regression
    classifier = model.evaluator["logreg"]
    _, _, acc["logreg"] = get_preds(classifier, c, labels)

    # mlp clasisfier
    classifier = model.evaluator["mlp"]
    _, _, acc["mlp"] = get_preds(classifier, c, labels)

    acc["clustering"], acc["clustering_nmi"], acc["clustering_ari"], acc["knn"] = 0.0, 0.0, 0.0, [0.0]
    return acc

""" ========================================================================================================  """
""" ========================================================================================================  """

if __name__ == "__main__":
    print("\n*** New Training Run ***\n\nModel params:")
    print(title_attribs)

    """ =================================================== PARAMETERS =====================================================  """
    # general model 
    num_sim     = args["n_augments"]
    sup_type    = args["sup_type"]
    self_sup    = (sup_type == 1)
    bs          = args["bs"]
    seed        = args["seed"]
    device      = "cuda" if args["cuda"] else "cpu";  print("\nDevice found: %s" % device)
    device      = torch.device(device)
    c_dim       = args["zdim"]
    training    = not (args["no_train"])  # else graphing

    # variance p(z'|z)
    beta_start  = args["beta_start"]
    p_y_prior   = args["p_y_prior"]

    # epochs 
    epochs      = args["epochs"]             # how much we train the model
    epochs_eval = args["eval_epochs"]   # how much we train the head for final evaluation

    # networks
    num_layer   = args["num_layer"]
    resnet18    = num_layer=='R18'
    bn          = not args["dec_bn_off"]            # batchnorm in the decoder layers

    # variance p(x|z)
    log_var_x_start  = args["log_var_x_start"]

    # dataset
    ds_name     = args["ds_name"]
    num_class   = args["num_class"]
    strength    = args["trans_strength"]
    assert strength in [0,1,2,3,None]
    target_transform = args["target_transform"]

    # optimizer 
    lr          = args["lr"]
    lr_decay    = args["lr_schedule"]
    wd          = args["wt_decay"]
    lr_eval     = args["lr_eval"]

    # inter eval
    eval_freq     = args["eval_freq"]
    eval_duration = args["eval_duration"]

    #checkpoint
    checkpoint  = args["checkpoint"]
    if checkpoint: checkpoint_epoch = checkpoint.split("/")[-1].split(".")[0]
    else: checkpoint_epoch="epoch0"

    if ds_name == "celeba":
        torch.multiprocessing.set_start_method('spawn')

    # adapt parameters based on number of augmentations
    bs = bs // num_sim
    num_sim_val = num_sim                       
    bs_val = bs * (num_sim//num_sim_val)    

    len_eval, nb_eval_block = 10, 0
    FORCE_TQDM, tag, quick_epochs, quick_run = True, "", 0, False

    if ds_name == "cifar10":
        x_dim, channels, size = 3072, 3, 32
        m_dims = None
        binarise = False
    elif ds_name == "mnist":
        x_dim, channels, size = 784, 1, 28
        m_dims = [500,500,2000]
        binarise = True 
    elif ds_name == "fashionmnist":
        x_dim, channels, size = 784, 1, 28
        m_dims = [500,500,2000]
        binarise = True 
    elif ds_name == "celeba":
        x_dim, channels, size = 64*64*3, 3, 64
        m_dims = [500,500,2000]
        binarise=False

    # create folders
    tag = "_Q%s" % tag
    path_data, path_project, path_output, path_graph = src.utils.utils.get_paths(args)
    fn_output = os.path.join( path_output, ds_name, str(sup_type), str(epochs), str(num_sim), str(lr), str(beta_start), str(log_var_x_start), f"output_{fn_attr_str}{tag}_q%03d" )  # save epoch data / pics
    fn_loss   = os.path.join( path_graph,  ds_name, str(sup_type), str(epochs), str(num_sim), str(lr), str(beta_start), str(log_var_x_start), f"loss_{fn_attr_str}{tag}_q-02.pt" )  # loss + accuracy saving
    fn_graph  = os.path.join( path_graph,  ds_name, str(sup_type), str(epochs), str(num_sim), str(lr), str(beta_start), str(log_var_x_start), f"train_{fn_attr_str}{tag}%s.jpg" )  # save epoch graphs

    os.makedirs(os.path.dirname(fn_output), exist_ok=True)
    os.makedirs(os.path.dirname(fn_loss), exist_ok=True)
    os.makedirs(os.path.dirname(fn_graph), exist_ok=True)

    # data loader function
    def get_data(bs_loader, mode_loader, num_sim_loader,strength,target_transform,seed,eval):
        return get_loader(bs_loader, path_data, ds_name, size, mode=mode_loader, num_workers=args["num_workers"], num_sim=num_sim_loader,strength=strength, seed=seed, target_transform=args["target_transform"] if (ds_name=="celeba") else "0", eval=eval)

    if training:
        src.utils.utils.set_seed(seed)

        # ------------     data   -------------#
        train_loader = get_data(bs_loader=bs,     mode_loader="train", num_sim_loader=num_sim, strength=strength, target_transform=target_transform, seed=seed,eval=False)
        valid_loader = get_data(bs_loader=bs_val, mode_loader="valid", num_sim_loader=num_sim_val, strength=strength, target_transform=target_transform, seed=seed, eval=False)
        num_train = len(train_loader.dataset)
        num_valid = len(valid_loader.dataset)

        # ------------     model parameters dictionnaries   -------------#
        net_args = {'x_dim':x_dim, 'm_dims':m_dims,'resnet18':resnet18, 'n_channels':channels, 'bn':bn,'c_dim':int(c_dim)} #, 'bn_momentum':0.01}
        vae_args = {'log_var_x':log_var_x_start ,'binarise':binarise}
        selfsup_args = {'var_z_y':beta_start, 'num_data':num_train, 'p_y_prior':p_y_prior}

        # ------------     model   -------------#
        if sup_type == 1:
            model = src.model.models.SelfSup(net_args, vae_args, selfsup_args, num_sim=num_sim, seed=seed, binary=True if num_class==2 else False ).to(device)   #, num_class=num_class)
        elif sup_type == 0:
            model = src.model.models.GmmVAEWithAugments(net_args, vae_args, num_sim=num_sim, seed=seed,).to(device)  #, num_class=num_class)
        
        # ------------     checkpoint loading   -------------#
        if checkpoint is not None:
            print(model.load_state_dict(torch.load(checkpoint)["model_state_dict"],strict=False))
            try:
                model.load_state_dict(torch.load(checkpoint)["model_state_dict"],strict=False)
            except: 
                print("Checkpoint does not exist or does not fit the current model settings")

        # ------------     optimizer   -------------#
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, eps=1e-4)

        def tqdm_enumerate(iterator, total, on_gpu=args['cuda'], force_tqdm=FORCE_TQDM, message=""):
            if on_gpu and not force_tqdm: 
                return enumerate(iterator)
            return tqdm(enumerate(iterator), total=total, desc=f"{message}")

        if isinstance(model, (src.model.models.BaseVAE)):

            eval_phase, eval_ON, eval_off = epochs_eval*["evaluate"], epochs_eval*[1], epochs_eval*[0]
            train_model = [1]*epochs
            train_head  = [1-x for x in train_model]
            train_phase = len(train_model) * ["train"]

            # intermediate evaluation
            shift=0
            for i in list(range(eval_freq, len(train_model), eval_freq))[:-1]:
                train_model[i+shift:i+shift] = eval_duration*[0]
                train_head[i+shift:i+shift]  = eval_duration*[1]
                train_phase[i+shift:i+shift] = eval_duration*["train"]
                shift += eval_duration

            schedule_phase = train_phase  + eval_phase
            schedule_model = train_model  + eval_off
            schedule_head  = train_head   + eval_ON
        
        else: raise NotImplementedError

        assert len(schedule_phase) == len(schedule_model), f'{schedule_phase}\n{schedule_model}\n{schedule_head}'
        scheduler = zip(schedule_phase, schedule_model, schedule_head)
        query_epochs = ( list(range(10, epochs, 2)) + [epochs - 1] )

        # ------------- Initialize ----------------------- #
        best_scores = {"logreg": 0.0, "mlp": 0.0, "clustering":0.0, "clustering_nmi":0., "clustering_ari":0.0, "knn": 0.0}
        losses = {"train":{},"validate":{},"evaluate":{}}
        part_loss = {"infonce":{},"ce":{},"entropy":{},"prior":{},"recon":{}}
        accuracy = {"logreg":{},"mlp":{},"clustering":{},"knn":{},"clustering_nmi":{},"clustering_ari":{}}
        metrics =  {"diff_means":{},"sigma_z_x":{},"diff_sigma":{},"current_pinch":{}}

        # ------------     train & evaluate  -------------#
        eval_optimizer, eval_optimizer_mlp, final_eval = None, None, False
        got_BaseVAE = isinstance(model, src.model.models.BaseVAE)
        epoch_trn = -1
        for epoch, schedule in enumerate(scheduler):
            phase, model_status, head_status = schedule

            if (phase == 'train') and (model_status == 1): epoch_trn += 1 
            fn_val, fn_trn = ( (fn_output % epoch_trn, f"{fn_output}_tr" % epoch_trn) if epoch_trn in query_epochs else ("", "") )

            if ((epoch>0) and (epochs !=0)) or ((epochs==0) and (epochs_eval!=0) and (epoch==0)):
                if ((epoch>0) and (epochs !=0)) :
                    head_turned_on   = ((head_status - schedule_head[epoch-1]) == 1)
                    head_turned_off  = ((head_status - schedule_head[epoch-1]) == -1)
                    final_eval = phase=='evaluate'
                    new_phase  = (phase != schedule_phase[epoch-1])

                else: 
                    head_turned_on = True 
                    head_turned_off = False 
                    final_eval = True
                    new_phase = True

                if head_turned_on or (new_phase and final_eval):
                    model.decoder.to('cpu')
                    eval_optimizer, eval_optimizer_mlp = model.eval_init(num_class, lr=lr_eval, final_eval=final_eval, binary=True if num_class==2 else False)
                    model.evaluator["logreg"].to(device)

                if head_turned_off:
                    del model.evaluator["logreg"]
                    model.decoder.to(device)
                    model.encoder.to(device)
            elif (epochs==0) and (epoch==0):
                if head_status == 1: 
                    eval_optimizer, eval_optimizer_mlp = model.eval_init(num_class, lr=lr_eval, final_eval=final_eval, binary=True if num_class==2 else False)
            elif (epochs==0) and (epoch==1) and (epochs_eval!=0):
                new_phase=False

            # ----------- train & validate -------------#
            if (model_status == 1):
                if final_eval and new_phase:
                    train_loader = get_data(bs_loader=64, mode_loader="train", num_sim_loader=1, strength=strength, target_transform=target_transform, seed=seed, eval=True)
                    valid_loader = get_data(bs_loader=64, mode_loader="valid", num_sim_loader=1, strength=strength, target_transform=target_transform, seed=seed, eval=True)
                    bs, bs_val=64, 64
                    num_sim, num_sim_val=1, 1

                train_epoch_loss, train_epoch_acc = train( 
                    model, train_loader, device, optimizer, binarise, fn_trn, num_sim=num_sim,  
                    eval_optim=eval_optimizer, eval_optim_mlp=eval_optimizer_mlp, schedule=schedule, shape=(channels,size,size),first_eval=(final_eval and new_phase))
                if final_eval and new_phase:train_loader.dataset.switch_off()

                if ((phase != 'train') or (phase == 'train' and model_status != 1)) and (ds_name != "celeba"): 
                    valid_epoch_loss, valid_epoch_acc = validate( 
                        model, valid_loader, device, num_sim=num_sim_val, schedule=schedule,first_eval=(final_eval and new_phase))
                else:
                    valid_epoch_loss, valid_epoch_acc = {"train":0.0, "recon":0.0, "entropy":0.0, "prior":0.0}, {"logreg":0.0,"mlp":0.0,"clustering":0.0,"clustering_nmi":0.0,"clustering_ari":0.0,"knn":0.0}
                if final_eval and new_phase:valid_loader.dataset.switch_off()

                # saving losses & accuracies
                if not math.isnan(train_epoch_loss["train"]) :      losses["train"][epoch]          = train_epoch_loss["train"]
                if train_epoch_loss["recon"] != [] :                part_loss["recon"][epoch]       = {i: x for i,x in enumerate(train_epoch_loss["recon"])} 
                if train_epoch_loss["entropy"] != [] :              part_loss["entropy"][epoch]     = {i: x for i,x in enumerate(train_epoch_loss["entropy"])} 
                if train_epoch_loss["prior"] != [] :                part_loss["prior"][epoch]       = {i: x for i,x in enumerate(train_epoch_loss["prior"])} 
                if train_epoch_loss["infonce"] != []:               part_loss["infonce"][epoch]     = {i: x for i,x in enumerate(train_epoch_loss["infonce"])}    
                if train_epoch_loss["ce"] != []:                    part_loss["ce"][epoch]          = {i: x for i,x in enumerate(train_epoch_loss["ce"])} 
                if not math.isnan(train_epoch_loss["evaluate"]):    losses["evaluate"][epoch]       = train_epoch_loss["evaluate"]
                if not math.isnan(train_epoch_acc["logreg"]) :      accuracy["logreg"][epoch]       = train_epoch_acc["logreg"]
                if not math.isnan(valid_epoch_loss["train"]) :          losses["validate"][epoch]           = valid_epoch_loss["train"]
                if not math.isnan(valid_epoch_acc["logreg"]) :          accuracy["logreg"][epoch]           = valid_epoch_acc["logreg"]
                if not math.isnan(valid_epoch_acc["mlp"]) :             accuracy["mlp"][epoch]              = valid_epoch_acc["mlp"]
                if not math.isnan(valid_epoch_acc["clustering"]) :      accuracy["clustering"][epoch]       = valid_epoch_acc["clustering"]
                if not math.isnan(valid_epoch_acc["clustering_nmi"]) :  accuracy["clustering_nmi"][epoch]   = valid_epoch_acc["clustering_nmi"]
                if not math.isnan(valid_epoch_acc["clustering_ari"]) :  accuracy["clustering_ari"][epoch]   = valid_epoch_acc["clustering_ari"]
                if not math.isnan(valid_epoch_acc["knn"]) :             accuracy["knn"][epoch]              = valid_epoch_acc["knn"]

                if (phase != 'train') or (phase == 'train' and model_status != 1):
                    if best_scores['logreg'] < (valid_epoch_acc['logreg']*100):
                        best_scores['logreg'] = (valid_epoch_acc['logreg']*100)
                    if best_scores['mlp'] < (valid_epoch_acc['mlp']*100):
                        best_scores['mlp'] = (valid_epoch_acc['mlp']*100)
                    if best_scores['clustering'] < (valid_epoch_acc['clustering']*100):
                        best_scores['clustering'] = (valid_epoch_acc['clustering']*100)
                    if best_scores['clustering_nmi'] < (valid_epoch_acc['clustering_nmi']*100):
                        best_scores['clustering_nmi'] = (valid_epoch_acc['clustering_nmi']*100)
                    if best_scores['clustering_ari'] < (valid_epoch_acc['clustering_ari']*100):
                        best_scores['clustering_ari'] = (valid_epoch_acc['clustering_ari']*100)
                    if best_scores['knn'] < (valid_epoch_acc['knn']*100):
                        best_scores['knn'] = (valid_epoch_acc['knn']*100)

                if list(train_epoch_loss['recon']) == []:
                    train_epoch_loss['recon'], train_epoch_loss['entropy'], train_epoch_loss['prior'] = [0.], [0.], [0.]

                message = f"{phase[0]}{model_status} Epoch {epoch+1:6d} of {epochs:6d}:  "  + \
                        f"T-L: {train_epoch_loss['train']:9.2f} " + \
                        f"({(list(train_epoch_loss['recon'])[-1]/bs):9.1f} " + \
                        f"{(list(train_epoch_loss['entropy'])[-1]/bs):5.1f} " + \
                        f"{(list(train_epoch_loss['prior'])[-1]/bs):5.1f})  " + \
                        f"V-L: {valid_epoch_loss['train']:9.2f} " + \
                        f"E-T-Ac: {train_epoch_acc['logreg']*100:5.2f}  " + \
                        f"E-V-Ac: {best_scores['logreg']:5.2f}  "  + \
                        f"E-V-MLP: {best_scores['mlp']:5.2f}  "  + \
                        f"E-C-Ac: {best_scores['clustering']:5.2f}  "  + \
                        f"E-C-NMI: {best_scores['clustering_nmi']:5.2f}  "  + \
                        f"E-C-ARI: {best_scores['clustering_ari']:5.2f}  "  + \
                        f"E-KNN-Ac: {best_scores['knn']:5.2f}  " 

                if epochs>0:
                    loss_plot(part_loss,fn_output,metrics)

                if (epoch+1)%100==0 or (epoch+1)==epochs: 
                    print("Saved here",fn_loss.split(".pt")[0]+f"_epoch{epoch}.pt",flush=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'loss': losses,
                        'accuracy': accuracy,
                        'metrics': metrics,
                        }, fn_loss.split(".pt")[0]+f"_epoch{epoch}.pt")
                    

    