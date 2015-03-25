#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import time

from dalec import dalec
from plot_utils import pretty_axes


__author__ = "J Gomez-Dans"
__version__ = "1.0 (09.03.2015)"
__email__ = "j.gomez-dans@ucl.ac.uk"

    
def safe_log(x, minval=0.0000000001):
    """This functions just does away with numerical 
    warnings."""
    return np.log(x.clip(min=minval))

class Observations ( object ):
    """A storage for the observational data"""
    def __init__ ( self, fname="dalec_drivers.OREGON.MW_obs.dat" ):
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
        self.fluxes = {}
        for flux in ["lai", "gpp","nee", "ra", "af", "aw", "ar", "lf", "lw","lr","cf","cw","cr",\
            "rh1","rh2","decomp", "cl", "rs","rt", "npp","nep","agb","tbm"]:
            if flux == "lai":
                self._read_LAI_obs()
            elif len( fluxes[flux] ) > 0:
                self.fluxes[flux] = np.loadtxt ( "meas_flux_%s.txt.gz" % flux )
        
    def _read_LAI_obs ( self, fname="Metolius_MOD15_LAI.txt"):
        """This function reads the LAI data, and rejiggles it so that it is
        in the same time axis as the drivers"""
        d = np.loadtxt( fname )
        lai_obs = []
        lai_unc = []
        lai_time = []
        time_track = 1
        for i in xrange( d.shape[0] ):
            year = d[i,0]
            if year < 2003:
                lai_obs.append ( d[i,2] )
                lai_unc.append ( d[i,3] )
                lai_time.append ( d[i,1] + (d[i,0]-2000)*365 )
        self.fluxes['lai'] = np.c_[np.array ( lai_time ), np.array ( lai_obs ), \
            np.array ( lai_unc )]
        print self.fluxes['lai']            
    
        
class Model ( object ):
    """A class for updating the DALEC model"""
    def __init__ ( self, model_params, drivers="dalec_drivers.OREGON.no_obs.dat" ):
        """This method basically stores the model parametes as a vector, as well as
        indicating where the drivers CSV data file is. The model parameters are 
        stored as
        
        1. Latitude
        2. SLA (specific leaf area)
        3+. These are ACM parameters (GPP/Photosynthesis internal model)
        """
        self.model_params = model_params
        self.drivers = self._process_drivers ( drivers )
        self.previous_state = []

    def _process_drivers ( self, drivers ):
        """A method to e.g. read meteo drivers"""
        return np.loadtxt ( drivers )

    def run_model ( self, x, i, store=False ):
        """Runs the model forward to ``i+1`` using input state ``x``, 
        assuming parameter vectors stored in the class. The output
        is returned to the user, but also stored in ``self.previous_state``,
        if you told the method to store the data. This is an option as
        one might be using this code within an ensemble.
        
        NOTE
        ----
        Not all fluxes are returned! For example, respiration and litter
        fluxes are ignored. Feel free to add them in.
        
        Parameters
        -----------
        x: iter
            The state, defined as Cf, Cr, Cw, Clit, Csom
            
        i: int
            The time step
            
        Returns
        --------
        nee, gpp, Cf, Cr, Cw, Clit, Csom 
            A tuple containing the updated state, as well as the fluxes
        """
        # Extract drivers...

        doy, temp, tmx, tmn, irad, psid, ca, rtot, nitro = self.drivers[i,:]
        tmp = 0.5* (tmx-tmn)

        # Extract the state components from the input
        Cf, Cr, Cw, Clit, Csom = x
        # Run DALEC
        ( nee, gpp, Ra, Rh1, Rh2, Af, Ar, Aw, Lw, Lr, D, \
            Cf, Cr, Cw, Clit, Csom, lai ) = dalec ( doy, \
            tmn, tmp, tmx, irad, ca, nitro, \
            self.model_params[0], self.model_params[1], \
            self.model_params[2:], \
            Cf, Cr, Cw, Clit, Csom, psid=psid, rtot=rtot )
        # Do we store the output?
        if store:
            self.previous_state.append ( [  nee, gpp, Ra, Rh1, Rh2, Af, Ar, Aw, Lw, Lr, D, \
            Cf, Cr, Cw, Clit, Csom, lai  ] )

        return ( nee, gpp, Ra, Rh1, Rh2, Af, Ar, Aw, Lw, Lr, D, \
            Cf, Cr, Cw, Clit, Csom )



def assimilate_obs ( timestep, ensemble, model, model_unc,  obs, obs_unc ):
    
    
    # Get the number of particles
    n_particles, state_size= ensemble.shape
    # The assimilated state will be stored here
    x = ensemble*0. 
    # The first particle is the median of the old state to start with
    x[0,:] = np.median ( ensemble, axis=0 )
    # Also get what the LAI of the median would be, using the SLA
    lai = x[0,0]/110.
    # Calculate the (log)likelihood
    log_prev = - safe_log ( 2.*np.pi*obs_unc ) -0.5*( lai - obs)**2/obs_unc**2
    # Store the initial state
    proposed_prev = x[0,:]
    # Next, we have a way of selecting random particles from the array
    part_sel = np.zeros ( n_particles, dtype=int )
    part_sel[1:] = np.random.choice(np.arange(n_particles),size=n_particles-1)
    # Main assimilation loop
    for particle in xrange ( 1, n_particles ):
        # Select one random particle, and advance it using the model.
        # We store the proposed state in ``proposed`` (creative, yeah?)
        proposed = model ( ensemble[part_sel[particle],:], timestep )[-5:] + \
                    np.random.randn( state_size )*model_unc
        while np.all ( proposed < 0 ):
            # Clips -ve values, that make no sense here
            proposed = model ( ensemble[part_sel[particle],:], timestep )[-5:] + \
                    np.random.randn( state_size )*model_unc
        # Calculate the predicted observations, using our (embarrassing) 
        # observation operator, in this case, divide by SLA
        lai = proposed[0]/110. # Using SLA directly here...
        if lai < 0:
            # Out of bounds
            log_proposed = -np.inf
        else:
            # Calculate the (log)likelihood
            log_proposed = - safe_log ( 2.*np.pi*obs_unc ) -0.5*( lai - obs)**2/obs_unc**2
        # Metropolis acceptance scheme
        alpha = min ( 1, np.exp ( log_proposed - log_prev ) )
        u = np.random.rand()
        if (u <= alpha) :
            x[particle, : ] = proposed
            proposed_prev = proposed
            log_prev = log_proposed
            
        else:
            x[particle, : ] = proposed_prev
        
    
    return x
        
    
    
    

def sequential_mh ( x0, \
                    model, model_unc, \
                    observations, obs_unc, \
                    time_axs, obs_time ):

    n_particles, state_size = x0.shape
    ensemble = x0
    state = np.zeros ( ( len(time_axs), n_particles, state_size ))
    for timestep in time_axs:
        # Check whether we have an observation for this time step
        
        if np.in1d ( timestep, obs_time ):

            obs_loc = np.nonzero( obs_time == timestep )[0]
            state[ timestep, :, : ] = assimilate_obs ( timestep, \
                ensemble, model, model_unc, \
                observations[obs_loc], obs_unc[obs_loc] )

        else:
            # No observation, just advance the model for all the particles
            for particle in xrange ( n_particles ):
                
                new_state = model ( \
                    ensemble[particle, :], timestep )[-5:] + \
                    np.random.randn( state_size )*model_unc
                while np.all ( new_state < 0 ):
                   new_state = model ( \
                        ensemble[particle, :], timestep )[-5:] + \
                       np.random.randn( state_size )*model_unc
                    
                state[ timestep, particle, : ] = new_state
        # Update the ensemble
        ensemble = state[ timestep, :, : ]*1.
        
    return state


def plot_pools_fluxes ( model, states, \
    pools = [r'$C_f$',r'$C_r$',r'$C_w$',r'$C_{lit}$',r'$C_{SOM}$'] ):
    
    
    fwd_model = np.zeros(( states.shape[0], states.shape[1], 16 ))

    clist = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", \
             "#FFD92F", "#E5C494", "#B3B3B3" ]

    for i in xrange ( states.shape[0] ):
        for p in xrange ( states.shape[1] ):
            fwd_model[i, p, :] = model.run_model ( states[i,p,:], i )

    fig1, axs = plt.subplots (nrows=5, ncols=1, sharex="col", figsize=(13,7) )
    tx = np.arange ( states.shape[0] )
    fluxes=["NEE", "GPP", "Ra", "Rh1", "Rh2"] + pools
    for i, ax in enumerate(axs.flatten() ):
        pretty_axes ( ax )
        ax.plot ( tx, fwd_model[:,:, i].mean(axis=1), '-', color=clist[i] )
        
        lb = [ np.percentile(fwd_model[j,:,i], 5) for j in xrange(1095)]
        ub = [ np.percentile(fwd_model[j,:,i], 95) for j in xrange(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.3  )
        ax.plot([], [],  color=clist[i], alpha=0.3, linewidth=10, label="5-95% CI")
        lb = [ np.percentile(fwd_model[j,:,i], 25) for j in xrange(1095)]
        ub = [ np.percentile(fwd_model[j,:,i], 75) for j in xrange(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.7  )
        ax.plot([], [],  color=clist[i], alpha=0.7, linewidth=10, label="25-75% CI")
        ax.set_title (fluxes[i], fontsize=12 )            
        ax.set_xlim ( 0, 1100 )
        ax.xaxis.set_ticklabels ([])
        try:
            if pools[i] == 'GPP':
                d = np.loadtxt ( "meas_flux_gpp.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            elif pools[i] == 'NEE':
                d = np.loadtxt ( "meas_flux_nee.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            elif pools[i] == 'Ra':
                d = np.loadtxt ( "meas_flux_ra.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
        except:
            pass
        ax.set_ylabel(r'$[gCm^{-2}d^{-1}]$')
        pretty_axes ( ax )
    ax.xaxis.set_ticks([1,365, 365*2, 365*3])
    ax.xaxis.set_ticklabels([1,365, 365*2, 365*3])
    ax.set_xlabel("Days after 01/01/2000")
    plt.subplots_adjust ( wspace=0.3 )


    fig2, axs = plt.subplots (nrows=5, ncols=1, figsize=(13,7) )
        
    for i, ax in enumerate(axs.flatten() ):
        pretty_axes ( ax )
        ax.plot ( tx, fwd_model[:, :,11+i].mean(axis=1), '-', color=clist[i] )
        
        lb = [ np.percentile(fwd_model[j,:,i+11], 5) for j in xrange(1095)]
        ub = [ np.percentile(fwd_model[j,:,i+11], 95) for j in xrange(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.2  )
        ax.plot([], [],  color=clist[i], alpha=0.2, linewidth=10, label="5-95% CI")
        lb = [ np.percentile(fwd_model[j,:,i+11], 25) for j in xrange(1095)]
        ub = [ np.percentile(fwd_model[j,:,i+11], 75) for j in xrange(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.7  )
        ax.plot([], [],  color=clist[i], alpha=0.7, linewidth=10, label="25-75% CI")
        ax.set_ylabel(r'$[gCm^{-2}]$')

        ax.set_title (pools[i], fontsize=12 )            
        ax.set_xlim ( 0, 1100 )
        ax.xaxis.set_ticklabels ([])
        try:
            if pools[i] == r'$C_f$':
                d = np.loadtxt ( "meas_flux_cf.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            elif pools[i] == r'$C_{lit}$':
                d = np.loadtxt ( "meas_flux_cl.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko' )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            elif pools[i] == r'$C_w$':
                d = np.loadtxt ( "meas_flux_cw.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko' )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            elif pools[i] == r'$C_r$':
                d = np.loadtxt ( "meas_flux_cr.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko' )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
        except:
            pass

        pretty_axes ( ax )
    ax.xaxis.set_ticks([1,365, 365*2, 365*3])
    ax.xaxis.set_ticklabels([1,365, 365*2, 365*3])
    plt.subplots_adjust ( wspace=0.3 )
    ax.set_xlabel("Days after 01/01/2000")
    
    fig3, axs = plt.subplots (nrows=2, ncols=3, figsize=(13,7) )
    fluxes2=[r'$A_f$',r'$A_r$', r'$A_w$',r'$L_f$',r'$L_r$',r'$L_w$','$D$' ]
    for i, ax in enumerate(axs.flatten() ):
        pretty_axes ( ax )
        ax.plot ( tx, fwd_model[:, :,5+i].mean(axis=1), '-', color=clist[i] )
        
        lb = [ np.percentile(fwd_model[j,:,i+5], 5) for j in xrange(1095)]
        ub = [ np.percentile(fwd_model[j,:,i+5], 95) for j in xrange(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.2  )
        ax.plot([], [],  color=clist[i], alpha=0.2, linewidth=10, label="5-95% CI")
        lb = [ np.percentile(fwd_model[j,:,i+5], 25) for j in xrange(1095)]
        ub = [ np.percentile(fwd_model[j,:,i+5], 75) for j in xrange(1095)]
        ax.fill_between ( np.arange(1095), lb, ub,color=clist[i],  alpha=0.7  )
        ax.plot([], [],  color=clist[i], alpha=0.7, linewidth=10, label="25-75% CI")
        ax.set_ylabel(r'$[gCm^{-2}d^{-1}]$')
        try:
            if pools[i] == r'$L_f$':
                d = np.loadtxt ( "meas_flux_lf.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko' )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
            elif pools[i] == r'$L_w$':
                d = np.loadtxt ( "meas_flux_lw.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko' )
                ax.vlines (d[:,0], d[:,1] - d[:,2], d[:,1]+d[:,2], )
        except:
            pass


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

def assimilate( sla=110, n_particles=750, Cf0=58., Cr0=102., Cw0=770.,\
                Clit0=40., Csom0=9897., Cfunc=5, Crunc=10, Cwunc=77, Clitunc=20,Csomunc=100  ):
    t0 = time.time()
    model_unc=np.array([Cfunc, Crunc, Cwunc, Clitunc, Csomunc])
    lat = 44.4 # Latitude
    sla = 110.
    n_particles = 500
    #params = np.array([ lat, sla, 4.4e-6, 0.47, 0.31, 0.43,0.0027, \
        #0.00000206, 0.00248, 0.0228, 0.00000265 ] )
    params = np.array([ lat, sla, 4.4e-6, 0.47, 0.31, 0.43,0.0010, \
        0.00000206, 0.00248, 0.0228, 0.00000265 ] )

    # Initial pool composition
    x0 = np.array( [ Cf0, Cr0, Cw0, Clit0, Csom0 ] )
    
    DALEC = Model ( params )
    
    
    
    lai_time, lai_obs, lai_unc = read_LAI_obs ()
    lai_unc = lai_unc*1.2 
    s0 = x0[:, None] + \
       np.random.randn(5, n_particles)*model_unc[:, None]
    s0 = s0.T
    
    results = sequential_mh ( s0, \
                    DALEC.run_model, model_unc, \
                    lai_obs, lai_unc, \
                    np.arange(1095), lai_time )
    
    
    fig = plt.figure(figsize=(13,7))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.08) 
    ax = plt.gca()
    clist = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", \
             "#FFD92F", "#E5C494", "#B3B3B3" ]
    lb = [ np.percentile(results[i,:,0]/110, 5) for i in xrange(1095)]
    ub = [ np.percentile(results[i,:,0]/110, 95) for i in xrange(1095)]
    plt.fill_between ( np.arange(1095), lb, ub,color=clist[0],  alpha=0.2  )
    plt.plot([], [],  color=clist[0], alpha=0.2, linewidth=10, label="5-95% CI")
    
    lb = [ np.percentile(results[i,:,0]/110, 25) for i in xrange(1095)]
    ub = [ np.percentile(results[i,:,0]/110, 75) for i in xrange(1095)]
    plt.fill_between ( np.arange(1095), lb,ub,  color=clist[0], alpha=0.6)
    plt.plot([], [], alpha=0.6, linewidth=10, color=clist[0], label="25-75% CI")
    m = [ np.percentile(results[i,:,0]/110, 50) for i in xrange(1095)]
    plt.plot(np.arange(1095), m, ls='-', color=clist[1], lw=1.8, label="Mean DA state" )
    plt.plot(lai_time, lai_obs, 'o', color=clist[2],label="MODIS LAI")
    plt.vlines ( lai_time, lai_obs - lai_unc, lai_obs + lai_unc,color=clist[2], )
    plt.xlabel("Days after 01/01/2000")
    plt.ylabel("LAI $[m^2\cdot m^{-2}]$")
    plt.legend(loc="upper left", fancybox=True, numpoints=1 )
    ax = plt.gca()
    pretty_axes ( ax )
    fig2, fig3, fig4, fwd_model = plot_pools_fluxes ( DALEC, results )
    elapsed_time = time.time() - t0
    print "Assimilation finished! Took %d seconds" % elapsed_time
    return fwd_model
