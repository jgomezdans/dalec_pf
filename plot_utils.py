#!/usr/bin/env python
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dalec import dalec, test_dalec


__author__ = "J Gomez-Dans"
__version__ = "1.0 (09.03.2015)"
__email__ = "j.gomez-dans@ucl.ac.uk"


DATA_DIR=Path("./data")


def pretty_axes( ax ):
    """This function takes an axis object ``ax``, and makes it purrty.
    Namely, it removes top and left axis & puts the ticks at the
    bottom and the left"""

    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(True)  
    ax.spines["right"].set_visible(False)              
    ax.spines["left"].set_visible(True)  

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    loc = plt.MaxNLocator( 6 )
    ax.yaxis.set_major_locator( loc )
    

    # ax.tick_params(axis="both", which="both", bottom="off", top="off",  
    #         labelbottom="on", left="off", right="off", labelleft="on")  
    
def plot_dalec ( outputs ):
    pools = [r'$NEE$', r'$GPP$',  '$Ra$', '$Rh_1 + Rh_2$', '$A_f$','$A_r$', \
        '$A_w$','$L_f$', '$L_r$', '$L_w$', '$D$',\
        r'$C_f$',r'$C_r$',r'$C_w$',r'$C_{lit}$',r'$C_{SOM}$']
    doys = np.arange( outputs.shape[1] )
    tx = np.arange ( len(doys))
    fig, axs = plt.subplots (nrows=4, ncols=4, sharex="col", figsize=(11,13) )
    
    for i, ax in enumerate(axs.flatten() ):
        pretty_axes ( ax )
        ax.plot ( tx, outputs[ i, :], '-' )
        ax.set_title (pools[i], fontsize=12 )
        
        try:
            if pools[i] == r'$C_f$':
                d = np.loadtxt ( DATA_DIR/"meas_flux_cf.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$C_{lit}$':
                d = np.loadtxt ( DATA_DIR/"meas_flux_cl.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$C_w$':
                d = np.loadtxt ( DATA_DIR/"meas_flux_cw.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$C_r$':
                d = np.loadtxt ( DATA_DIR/"meas_flux_cr.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$L_f$':
                d = np.loadtxt ( DATA_DIR/"meas_flux_lf.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$GPP$':
                d = np.loadtxt ( DATA_DIR/"meas_flux_gpp.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none")
                
            elif pools[i] == r'$NEE$':
                d = np.loadtxt ( DATA_DIR/"meas_flux_nee.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none")
                
            elif pools[i] == r'$Ra$':
                d = np.loadtxt ( DATA_DIR/"meas_flux_ra.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                

        except:
            pass

        ax.set_xlim ( 0, 1100 )
        ax.xaxis.set_ticks([1,365, 365*2, 365*3])
    plt.subplots_adjust ( wspace=0.4 )
    
def plot_fluxes ( model, states):
    
    
    # Run the vanilla model and store the fluxes...
    vanilla_dalec = test_dalec ( )
    fwd_model = np.zeros(( states.shape[0], states.shape[1], 16 ))

    clist = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", \
             "#FFD92F", "#E5C494", "#B3B3B3" ]

    for i in range ( states.shape[0] ):
        for p in range ( states.shape[1] ):
            fwd_model[i, p, :] = model.run_model ( states[i,p,:], i )

    fig1, axs = plt.subplots (nrows=5, ncols=1, sharex="col", figsize=(10, 18) )
    tx = np.arange ( states.shape[0] )
    fluxes=["NEE", "GPP", "Ra", "Rh1", "Rh2"] 
    for i, ax in enumerate(axs.flatten() ):
        
        ax.plot ( tx, fwd_model[:,:, i].mean(axis=1), '-', color=clist[i] )
        
        lb = [ np.percentile(fwd_model[j,:,i], 5) for j in range(1095)]
        ub = [ np.percentile(fwd_model[j,:,i], 95) for j in range(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.3  )
        ax.plot([], [],  color=clist[i], alpha=0.3, linewidth=10, label="5-95% CI")
        lb = [ np.percentile(fwd_model[j,:,i], 25) for j in range(1095)]
        ub = [ np.percentile(fwd_model[j,:,i], 75) for j in range(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.7  )
        ax.plot([], [],  color=clist[i], alpha=0.7, linewidth=10, label="25-75% CI")
        ax.plot ( np.arange(1095), vanilla_dalec[ i, :-1], '--k', label="No DA" )
        ax.set_title (fluxes[i], fontsize=12 ) 
        
        ax.set_xlim ( 0, 1100 )
        ax.xaxis.set_ticklabels ([])
        if fluxes[i] == 'GPP':
            d = np.loadtxt ( DATA_DIR/"meas_flux_gpp.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none", alpha=0.5 )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], alpha=0.5 )
            ax.plot ( np.arange(1095), vanilla_dalec[ i, :-1], '--', c="0.8", alpha=0.5 )
        elif fluxes[i] == 'NEE':
            d = np.loadtxt ( DATA_DIR/"meas_flux_nee.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" , alpha=0.5 )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], alpha=0.5 )
            ax.plot ( np.arange(1095), vanilla_dalec[ i, :-1], '--', c="0.8", alpha=0.5 )
        elif fluxes[i] == 'Ra':
            d = np.loadtxt ( DATA_DIR/"meas_flux_ra.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none", alpha=0.5 )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], alpha=0.5 )
            ax.plot ( np.arange(1095), vanilla_dalec[ i, :-1], '--', c="0.8", alpha=0.5 )
        ax.set_ylabel(r'$[gCm^{-2}d^{-1}]$')
        pretty_axes ( ax )
    ax.xaxis.set_ticks([1,365, 365*2, 365*3])
    ax.xaxis.set_ticklabels([1,365, 365*2, 365*3])
    ax.set_xlabel("Days after 01/01/2000")
    plt.subplots_adjust ( wspace=0.3 )


def plot_pools_fluxes ( model, states, \
    pools = [r'$C_f$',r'$C_r$',r'$C_w$',r'$C_{lit}$',r'$C_{SOM}$'] ):
    
    # Run the vanilla model and store the fluxes...
    vanilla_dalec = test_dalec ( )
    fwd_model = np.zeros(( states.shape[0], states.shape[1], 16 ))

    clist = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", \
             "#FFD92F", "#E5C494", "#B3B3B3" ]

    for i in range ( states.shape[0] ):
        for p in range ( states.shape[1] ):
            fwd_model[i, p, :] = model.run_model ( states[i,p,:], i )

    fig1, axs = plt.subplots (nrows=5, ncols=1, sharex="col", figsize=(13,18) )
    tx = np.arange ( states.shape[0] )
    fluxes=["NEE", "GPP", "Ra", "Rh1", "Rh2"] + pools
    for i, ax in enumerate(axs.flatten() ):
        pretty_axes ( ax )
        ax.plot ( tx, fwd_model[:,:, i].mean(axis=1), '-', color=clist[i] )
        
        lb = [ np.percentile(fwd_model[j,:,i], 5) for j in range(1095)]
        ub = [ np.percentile(fwd_model[j,:,i], 95) for j in range(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.3  )
        ax.plot([], [],  color=clist[i], alpha=0.3, linewidth=10, label="5-95% CI")
        lb = [ np.percentile(fwd_model[j,:,i], 25) for j in range(1095)]
        ub = [ np.percentile(fwd_model[j,:,i], 75) for j in range(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.7  )
        ax.plot([], [],  color=clist[i], alpha=0.7, linewidth=10, label="25-75% CI")
        ax.plot ( np.arange(1095), vanilla_dalec[ i, :-1], '--k', label="No DA" )
        ax.set_title (fluxes[i], fontsize=12 ) 
        
        ax.set_xlim ( 0, 1100 )
        ax.xaxis.set_ticklabels ([])
        if fluxes[i] == 'GPP':
            d = np.loadtxt ( DATA_DIR/"meas_flux_gpp.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            ax.plot ( np.arange(1095), vanilla_dalec[ i, :-1], '--k' )
        elif fluxes[i] == 'NEE':
            d = np.loadtxt ( DATA_DIR/"meas_flux_nee.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            ax.plot ( np.arange(1095), vanilla_dalec[ i, :-1], '--k' )
        elif fluxes[i] == 'Ra':
            d = np.loadtxt ( DATA_DIR/"meas_flux_ra.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            ax.plot ( np.arange(1095), vanilla_dalec[ i, :-1], '--k' )
        ax.set_ylabel(r'$[gCm^{-2}d^{-1}]$')
        pretty_axes ( ax )
    ax.xaxis.set_ticks([1,365, 365*2, 365*3])
    ax.xaxis.set_ticklabels([1,365, 365*2, 365*3])
    ax.set_xlabel("Days after 01/01/2000")
    plt.subplots_adjust ( wspace=0.3 )


    fig2, axs = plt.subplots (nrows=5, ncols=1, figsize=(13,18) )
        
    for i, ax in enumerate(axs.flatten() ):
        pretty_axes ( ax )
        ax.plot ( tx, fwd_model[:, :,11+i].mean(axis=1), '-', color=clist[i] )
        
        lb = [ np.percentile(fwd_model[j,:,i+11], 5) for j in range(1095)]
        ub = [ np.percentile(fwd_model[j,:,i+11], 95) for j in range(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.2  )
        ax.plot([], [],  color=clist[i], alpha=0.2, linewidth=10, label="5-95% CI")
        lb = [ np.percentile(fwd_model[j,:,i+11], 25) for j in range(1095)]
        ub = [ np.percentile(fwd_model[j,:,i+11], 75) for j in range(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.7  )
        ax.plot([], [],  color=clist[i], alpha=0.7, linewidth=10, label="25-75% CI")
        ax.set_ylabel(r'$[gCm^{-2}]$')
        ax.plot ( np.arange(1095), vanilla_dalec[ 11+i, :-1], '--k', label="No DA" )
        ax.set_title (pools[i], fontsize=12 )            
        ax.set_xlim ( 0, 1100 )
        ax.xaxis.set_ticklabels ([])
        
        if pools[i] == r'$C_f$':
            d = np.loadtxt ( DATA_DIR/"meas_flux_cf.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            
        elif pools[i] == r'$C_{lit}$':
            d = np.atleast_2d ( np.loadtxt ( DATA_DIR/"meas_flux_cl.txt.gz" ) )
            ax.plot ( d[:, 0], d[:,1], 'ko' )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            
        elif pools[i] == r'$C_w$':
            d = np.loadtxt ( DATA_DIR/"meas_flux_cw.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko' )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            
        elif pools[i] == r'$C_r$':
            d = np.loadtxt ( DATA_DIR/"meas_flux_cr.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko' )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            
        
        pretty_axes ( ax )
    ax.xaxis.set_ticks([1,365, 365*2, 365*3])
    ax.xaxis.set_ticklabels([1,365, 365*2, 365*3])
    plt.subplots_adjust ( wspace=0.3 )
    ax.set_xlabel("Days after 01/01/2000")
    
    fig3, axs = plt.subplots (nrows=2, ncols=3, figsize=(13,18) )
    fluxes2=[r'$A_f$',r'$A_r$', r'$A_w$',r'$L_f$',r'$L_r$',r'$L_w$','$D$' ]
    for i, ax in enumerate(axs.flatten() ):
        pretty_axes ( ax )
        ax.plot ( tx, fwd_model[:, :,5+i].mean(axis=1), '-', color=clist[i] )
        
        lb = [ np.percentile(fwd_model[j,:,i+5], 5) for j in range(1095)]
        ub = [ np.percentile(fwd_model[j,:,i+5], 95) for j in range(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.2  )
        ax.plot([], [],  color=clist[i], alpha=0.2, linewidth=10, label="5-95% CI")
        lb = [ np.percentile(fwd_model[j,:,i+5], 25) for j in range(1095)]
        ub = [ np.percentile(fwd_model[j,:,i+5], 75) for j in range(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.7  )
        ax.plot([], [],  color=clist[i], alpha=0.7, linewidth=10, label="25-75% CI")
        ax.plot ( np.arange(1095), vanilla_dalec[ i + 5, :-1], '--k', label="No DA" )
        ax.set_ylabel(r'$[gCm^{-2}d^{-1}]$')

        if fluxes2[i] == r'$L_f$':
            d = np.loadtxt ( DATA_DIR/"meas_flux_lf.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko' )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            
        elif fluxes2[i] == r'$L_w$':
            d = np.loadtxt ( DATA_DIR/"meas_flux_lw.txt.gz" )
            ax.plot ( d[:, 0], d[:,1], 'ko' )
            ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )


        ax.set_title (fluxes2[i], fontsize=12 )            
        ax.set_xlim ( 0, 1100 )
        ax.xaxis.set_ticklabels ([])
        pretty_axes ( ax )
        ax.xaxis.set_ticks([1,365, 365*2, 365*3])
        
        plt.subplots_adjust ( wspace=0.3 )
        if i > 2:
            ax.xaxis.set_ticklabels([1,365, 365*2, 365*3])
            ax.set_xlabel("Days after 01/01/2000")
    
    
    
    
    return fig1, fig2, fig3, fwd_model

def pf_plots ( DALEC, observations, results ):
    """Plots the results of the assimilation using a particle filter"""
    fig = plt.figure(figsize=(13,18))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.08) 
    ax = plt.gca()
    clist = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", \
             "#FFD92F", "#E5C494", "#B3B3B3" ]
    lb = [ np.percentile(results[i,:,0]/observations.sla, 5) for i in range(1095)]
    ub = [ np.percentile(results[i,:,0]/observations.sla, 95) for i in range(1095)]
    plt.fill_between ( np.arange(1095), lb, ub,color=clist[0],  alpha=0.2  )
    plt.plot([], [],  color=clist[0], alpha=0.2, linewidth=10, label="5-95% CI")
    
    lb = [ np.percentile(results[i,:,0]/observations.sla, 25) for i in range(1095)]
    ub = [ np.percentile(results[i,:,0]/observations.sla, 75) for i in range(1095)]
    plt.fill_between ( np.arange(1095), lb,ub,  color=clist[0], alpha=0.6)
    plt.plot([], [], alpha=0.6, linewidth=10, color=clist[0], label="25-75% CI")
    m = [ np.percentile(results[i,:,0]/observations.sla, 50) for i in range(1095)]
    plt.plot(np.arange(1095), m, ls='-', color=clist[1], lw=1.8, label="Mean DA state" )
    plt.plot( observations.fluxes['lai'][:,0], observations.fluxes['lai'][:,1], \
        'o', color=clist[2],label="MODIS LAI")
    plt.vlines ( observations.fluxes['lai'][:,0], observations.fluxes['lai'][:,1] - \
        observations.fluxes['lai'][:,2], observations.fluxes['lai'][:,1] + \
        observations.fluxes['lai'][:,2],color=clist[2] )
    plt.xlabel("Days after 01/01/2000")
    plt.ylabel("LAI $[m^2\cdot m^{-2}]$")
    plt.legend(loc="upper left", fancybox=True, numpoints=1 )
    ax = plt.gca()
    pretty_axes ( ax )
    fig2, fig3, fig4, fwd_model = plot_pools_fluxes ( DALEC, results )
    return fwd_model
