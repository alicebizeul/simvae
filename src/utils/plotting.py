"""
Module Name: plotting.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import itertools

matplotlib.style.use("ggplot")

def loss_plot(part_loss,fn_output,part_metric):
    ref = [part_loss[key] for key in list(part_loss.keys()) if part_loss[key] != {}][0]

    fig1, axs    = plt.subplots(3, 1, layout='constrained', figsize=(10, 10))

    epoch_range  = list(ref.keys())
    step_range   = list(ref[list(ref.keys())[0]].keys())

    loss_prior   = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_loss["prior"].values())]))
    loss_recon   = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_loss["recon"].values())]))
    loss_entropy = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_loss["entropy"].values())]))
    loss_infonce = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_loss["infonce"].values())]))
    loss_ce      = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_loss["ce"].values())]))

    metric_means      = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_metric["diff_means"].values())]))
    metric_sigma      = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_metric["sigma_z_x"].values())]))
    metric_diff_sigma = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_metric["diff_sigma"].values())]))

    if loss_recon != []:
        if metric_means==[]:fig1, axs    = plt.subplots(3, 1, layout='constrained', figsize=(10, 10))
        else: fig1, axs    = plt.subplots(4, 1, layout='constrained', figsize=(10, 10))

        axs[0].plot(list(range(len(loss_prior))),loss_prior,label="prior")
        axs[1].plot(list(range(len(loss_recon))),loss_recon,label="recon")
        axs[2].plot(list(range(len(loss_entropy))),loss_entropy,label="entropy")
        axs[0].plot(list(range(len(loss_entropy)))[::len(step_range)],[np.mean(x) for x in np.array_split(loss_prior,int(len(loss_prior)/len(step_range)))],linestyle="--")
        axs[1].plot(list(range(len(loss_entropy)))[::len(step_range)],[np.mean(x) for x in np.array_split(loss_recon,int(len(loss_recon)/len(step_range)))],linestyle="--")
        axs[2].plot(list(range(len(loss_entropy)))[::len(step_range)],[np.mean(x) for x in np.array_split(loss_entropy,int(len(loss_entropy)/len(step_range)))],linestyle="--")
        axs[0].set_xticks(list(range(len(loss_entropy)))[::len(step_range)],epoch_range)
        axs[1].set_xticks(list(range(len(loss_entropy)))[::len(step_range)],epoch_range)
        axs[2].set_xticks(list(range(len(loss_entropy)))[::len(step_range)],epoch_range)
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        if metric_means != []:
            axs[3].plot(list(range(len(metric_means))),metric_means,label="means")
            axs[3].plot(list(range(len(metric_diff_sigma))),metric_diff_sigma,label="diff_sigma")
            ax_tmp=axs[3].twinx()
            ax_tmp.plot(list(range(len(metric_sigma))),metric_sigma,label="sigma",c="purple")

            axs[3].set_xticks(list(range(len(metric_means)))[::len(step_range)],epoch_range)
            axs[3].set_ylabel("Diff means / Rel diff means")
            ax_tmp.set_ylabel("Sigma")

            axs[3].legend()
            ax_tmp.legend()

    elif loss_infonce != []:
        fig1, axs    = plt.subplots(1, 1, layout='constrained', figsize=(10, 10))
        axs.plot(list(range(len(loss_infonce))),loss_infonce,label="infonce")
        axs.set_xticks(list(range(len(loss_infonce)))[::len(step_range)],epoch_range)
        axs.legend()
    elif loss_ce != []:
        fig1, axs    = plt.subplots(1, 1, layout='constrained', figsize=(10, 10))
        axs.plot(list(range(len(loss_ce))),loss_ce,label="ce")
        axs.legend()

    title=os.path.basename(fn_output).split(".")[0]
    plt.savefig(os.path.dirname(fn_output)+f'/losses_{title}.png')

    if loss_recon != []:
        fig1, axs = plt.subplots(1,1,layout="constrained", figsize=(10,10))
        axs.plot(list(range(len(loss_prior))),[x + y + z for x, y, z in zip(loss_entropy,loss_recon,loss_recon)],label="rec+ent+pr")
        axs.set_xticks(list(range(len(loss_recon)))[::len(step_range)],epoch_range)
        axs.legend()
        title=os.path.basename(fn_output).split(".")[0]
        plt.savefig(os.path.dirname(fn_output)+f'/losses_all_{title}.png')

        fig1, axs = plt.subplots(1,1,layout="constrained", figsize=(10,10))
        axs.plot(list(range(len(loss_prior))),[x + y + z for x, y, z in zip(loss_entropy,loss_recon,loss_recon)],label="rec+ent+pr")
        axs.set_xticks(list(range(len(loss_recon)))[::len(step_range)],epoch_range)
        axs.legend()
        try:
            axs.set_yscale('log')
        except: pass
        title=os.path.basename(fn_output).split(".")[0]
        plt.savefig(os.path.dirname(fn_output)+f'/losses_all_log_{title}.png')


    if loss_recon != []:
        
        if metric_means==[]:
            fig1, axs    = plt.subplots(3, 1, layout='constrained', figsize=(10, 10))
        else: fig1, axs    = plt.subplots(4, 1, layout='constrained', figsize=(10, 10))

        epoch_range  = list(part_loss["prior"].keys())
        step_range   = list(part_loss["prior"][list(part_loss["prior"].keys())[0]].keys())
        loss_prior   = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_loss["prior"].values())]))
        loss_recon   = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_loss["recon"].values())]))
        loss_entropy = list(itertools.chain.from_iterable([list(x.values()) for x in list(part_loss["entropy"].values())]))
        axs[0].plot(list(range(len(loss_prior))),loss_prior,label="prior")
        axs[1].plot(list(range(len(loss_recon))),loss_recon,label="recon")
        axs[2].plot(list(range(len(loss_entropy))),loss_entropy,label="entropy")
        axs[0].plot(list(range(len(loss_entropy)))[::len(step_range)],[np.mean(x) for x in np.array_split(loss_prior,int(len(loss_prior)/len(step_range)))],linestyle="--")
        axs[1].plot(list(range(len(loss_entropy)))[::len(step_range)],[np.mean(x) for x in np.array_split(loss_recon,int(len(loss_recon)/len(step_range)))],linestyle="--")
        axs[2].plot(list(range(len(loss_entropy)))[::len(step_range)],[np.mean(x) for x in np.array_split(loss_entropy,int(len(loss_entropy)/len(step_range)))],linestyle="--")
        axs[0].set_xticks(list(range(len(loss_entropy)))[::len(step_range)],epoch_range)
        axs[1].set_xticks(list(range(len(loss_entropy)))[::len(step_range)],epoch_range)
        axs[2].set_xticks(list(range(len(loss_entropy)))[::len(step_range)],epoch_range)
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        if metric_means != []:
            axs[3].plot(list(range(len(metric_means))),metric_means,label="means")
            axs[3].plot(list(range(len(metric_diff_sigma))),metric_diff_sigma,label="diff_sigma")
            
            ax_tmp=axs[3].twinx()
            ax_tmp.plot(list(range(len(metric_sigma))),metric_sigma,label="sigma",c="purple")

            nb_epochs=len(list(part_metric["current_pinch"].values()))
            ax_tmp.hlines(list(part_metric["current_pinch"].values()),xmin=list(np.linspace(0,nb_epochs-1,nb_epochs)),xmax=list(np.linspace(1,nb_epochs,nb_epochs)),color='k', linestyle='-')

            axs[3].set_xticks(list(range(len(metric_means)))[::len(step_range)],epoch_range)
            axs[3].set_ylabel("Diff means / Rel diff means")
            ax_tmp.set_ylabel("Sigma^2")

            axs[3].legend()
            ax_tmp.legend()
        try:
            axs[0].set_yscale('log')
        except: pass
        try:
            axs[1].set_yscale('log')
        except: pass
        try:
            axs[2].set_yscale('log')
        except: pass
        try:
            axs[3].set_yscale('log')
        except: pass
        try:
            ax_tmp.set_yscale('log')
        except: pass
        
        title=os.path.basename(fn_output).split(".")[0]
        plt.savefig(os.path.dirname(fn_output)+f'/losses_log_{title}.png')
    
    plt.close('all')
    return


