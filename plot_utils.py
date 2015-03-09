#!/usr/bin/env python
import json
import numpy as np
import matplotlib.pyplot as plt


__author__ = "J Gomez-Dans"
__version__ = "1.0 (09.03.2015)"
__email__ = "j.gomez-dans@ucl.ac.uk"


def get_observations ( fname="dalec_drivers.OREGON.MW_obs.dat" ):
    """This is an utility function that extracts the observations 
    from the original ASCII file.
    """
    fluxes = {}
    for flux in ["lai", "gpp","nee", "ra", "af", "aw", "ar", "lf", "lw","lr","cf","cw","cr",\
        "rh1","rh2","decomp", "cl", "rs","rt", "npp","nep","agb","tbm"]:
        fp = open( fname, 'r')
        this_flux = []
        for i, line in enumerate( fp ):
            sline = line.split()
            if sline.count( flux ) > 0:
                j = sline.index( flux )
                this_flux.append ( [ float(i), float(sline[j+1]), float(sline[j+2] ) ] )
        fluxes[flux] = np.array( this_flux )
        fp.close()

    for flux in fluxes.iterkeys():
        if len( fluxes[flux] ) > 0:
            print "Saving obs stream: %s (%d obs)" % ( flux, len( fluxes[flux] ) )
            np.savetxt ( "meas_flux_%s.txt.gz" % flux, fluxes[flux] )



def plot_config ():
    """Update the MPL configuration"""
    config_json='''{
            "lines.linewidth": 2.0,
            "axes.edgecolor": "#bcbcbc",
            "patch.linewidth": 0.5,
            "legend.fancybox": true,
            "axes.color_cycle": [
                "#FC8D62",
                "#66C2A5",
                "#8DA0CB",
                "#E78AC3",
                "#A6D854",
                "#FFD92F",
                "#E5C494",
                "#B3B3B3"
            ],
            "axes.facecolor": "#eeeeee",
            "axes.labelsize": "large",
            "axes.grid": false,
            "patch.edgecolor": "#eeeeee",
            "axes.titlesize": "x-large",
            "svg.embed_char_paths": "path",
            "xtick.direction" : "out",
            "ytick.direction" : "out",
            "xtick.color": "#262626",
            "ytick.color": "#262626",
            "axes.edgecolor": "#262626",
            "axes.labelcolor": "#262626",
            "axes.labelsize": 12,
            "font.size": 12,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
            
    }
    '''
    s = json.loads ( config_json )
    plt.rcParams.update(s)
    plt.rcParams["axes.formatter.limits"] = [-4,4]
    

def pretty_axes( ax ):
    """This function takes an axis object ``ax``, and makes it purrty.
    Namely, it removes top and left axis & puts the ticks at the
    bottom and the left"""

    ax.spines["top"].set_visible(False)  
    ax.spines["bottom"].set_visible(True)  
    ax.spines["right"].set_visible(False)              
    ax.spines["left"].set_visible(True)  

    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    loc = plt.MaxNLocator( 6 )
    ax.yaxis.set_major_locator( loc )
    

    ax.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")  
