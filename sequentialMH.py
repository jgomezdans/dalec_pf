#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
from dalec import dalec

def simpleaxis(ax):
    """ Remove the top line and right line on the plot face """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
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
        ( nee, gpp, Cf, Cr, Cw, Clit, Csom, lai ) = dalec ( doy, \
            tmn, tmp, tmx, irad, ca, nitro, \
            self.model_params[0], self.model_params[1], \
            self.model_params[2:], \
            Cf, Cr, Cw, Clit, Csom, psid=psid, rtot=rtot )
        # Do we store the output?
        if store:
            self.previous_state.append ( [ nee, gpp, Cf, Cr, Cw, \
                Clit, Csom ] )

        return nee, gpp, Cf, Cr, Cw, Clit, Csom

def read_LAI_obs ( fname="Metolius_MOD15_LAI.txt"):
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
            
    return np.array ( lai_time ), np.array ( lai_obs ), np.array ( lai_unc )            

def assimilate_obs (timestep, ensemble, model, model_unc,  obs, obs_unc ):
    
    
    # Get the number of particles
    n_particles, state_size= ensemble.shape
    # The assimilated state will be stored here
    x = ensemble*0. 
    # The first particle is the median of the old state to start with
    x[0,:] = np.median ( ensemble, axis=0 )
    # Also get what the LAI of the median would be, using the SLA
    lai = x[0,0]/110.
    # Calculate the (log)likelihood
    log_prev = - np.log ( 2.*np.pi*obs_unc ) -0.5*( lai - obs)**2/obs_unc**2
    # Store the initial state
    proposed_prev = x[0,:]
    # Next, we have a way of selecting random particles from the array
    part_sel = np.zeros ( n_particles, dtype=int )
    part_sel[1:] = np.random.choice(np.arange(n_particles),size=n_particles-1)
    # Main assimilation loop
    for particle in xrange ( 1, n_particles ):
        # Select one random particle, and advance it using the model.
        # We store the proposed state in ``proposed`` (creative, yeah?)
        proposed = model ( ensemble[part_sel[particle],:], timestep )[2:] + \
                    np.random.randn( state_size )*model_unc 
        # Calculate the predicted observations, using our (embarrassing) 
        # observation operator, in this case, divide by SLA
        lai = proposed[0]/110. # Using SLA directly here...
        if lai < 0:
            # Out of bounds
            log_proposed = -np.inf
        else:
            # Calculate the (log)likelihood
            log_proposed = - np.log ( 2.*np.pi*obs_unc ) -0.5*( lai - obs)**2/obs_unc**2
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
                state[ timestep, particle, : ] = model ( \
                    ensemble[particle, :], timestep )[2:] + \
                    np.random.randn( state_size )*model_unc
        # Update the ensemble
        ensemble = state[ timestep, :, : ]*1.
        print timestep, ensemble.mean(axis=0)[0], ensemble.std(axis=0)[0]
    return state

            
            
if __name__ == "__main__":
    lat = 44.4 # Latitude
    sla = 110.
    n_particles = 500
    #params = np.array([ lat, sla, 4.4e-6, 0.47, 0.31, 0.43,0.0027, \
        #0.00000206, 0.00248, 0.0228, 0.00000265 ] )
    params = np.array([ lat, sla, 4.4e-6, 0.47, 0.31, 0.43,0.0010, \
        0.00000206, 0.00248, 0.0228, 0.00000265 ] )

    # Initial pool composition
    x0 = np.array( [ 58., 102., 770., 40., 9897.] )
    
    DALEC = Model ( params )
    
    model_unc = np.random.randn()*x0/10.
    
    lai_time, lai_obs, lai_unc = read_LAI_obs ()
    
    s0 = x0[:, None] + \
       np.random.randn(5, n_particles)*x0[:,None]*0.1 # 10% error
    s0 = s0.T
    
    results = sequential_mh ( s0, \
                    DALEC.run_model, model_unc, \
                    lai_obs, lai_unc, \
                    np.arange(1095), lai_time )
    
    colour_list = brewer2mpl.get_map('Set2', 'qualitative', 8).mpl_colors
    fig = plt.figure(figsize=(13,7))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.08) 
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    
    almost_black = '#262626'
    # change the tick colors also to the almost black
    plt.rcParams['ytick.color'] = almost_black
    plt.rcParams['xtick.color'] = almost_black

    # change the text colors also to the almost black
    plt.rcParams['text.color'] = almost_black
    
    # Change the default axis colors from black to a slightly lighter black,
    # and a little thinner (0.5 instead of 1)
    plt.rcParams['axes.edgecolor'] = almost_black
    plt.rcParams['axes.labelcolor'] = almost_black

    lb = [ np.percentile(results[i,:,0]/110, 5) for i in xrange(1095)]
    ub = [ np.percentile(results[i,:,0]/110, 95) for i in xrange(1095)]
    plt.fill_between ( np.arange(1095), lb, ub, color=colour_list[0], alpha=0.2, label="5-95% CI" )
    lb = [ np.percentile(results[i,:,0]/110, 25) for i in xrange(1095)]
    ub = [ np.percentile(results[i,:,0]/110, 75) for i in xrange(1095)]
    plt.fill_between ( np.arange(1095), lb,ub, color=colour_list[0], alpha=0.6, label="25-75% CI")
    m = [ np.percentile(results[i,:,0]/110, 50) for i in xrange(1095)]
    plt.plot(np.arange(1095), m, ls='-', c=colour_list[1], lw=1.8, label="Mean DA state" )
    plt.plot(lai_time, lai_obs, 'o', c=colour_list[2], label="MODIS LAI")
    plt.vlines ( lai_time, lai_obs - lai_unc, lai_obs + lai_unc )
    plt.xlabel("Days after 01/01/2000")
    plt.ylabel("LAI $[m^2\cdot m^{-2}]$")
    plt.legend(loc="upper left", fancybox=True, numpoints=1 )
    ax = plt.gca()
    simpleaxis ( ax )
    plt.show()