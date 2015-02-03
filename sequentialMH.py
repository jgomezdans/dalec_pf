#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from dalec import dalec

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

def assimilate_obs (timestep, ensemble, model, model_unc,  obs, obs_unc ):
    
    
    # Get the numberof particles
    n_particles = ensemble.shape[0]
    # The assimilated state will be stored here
    x = ensemble*0. 
    # The first particle is the median of the old state to start with
    x[0,:] = np.median ( ensemble, axis=0 )
    # Also get what the LAI of the median would be, using the SLA
    lai = x[0,2]/111.
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
        proposed = model ( ensemble[part_sel[timestep],:], timestep ) + \
                    np.random.randn( state_size )*model_unc 
        # Calculate the predicted observations, using our (embarrassing) 
        # observation operator, in this case, divide by SLA
        lai = proposed[2]/111. # Using SLA directly here...
        # Calculate the (log)likelihood
        log_proposed = - np.log ( 2.*np.pi*obs_unc ) -0.5*( lai - obs)**2/obs_unc**2
        # Metropolis acceptance scheme
        alpha = min ( 1, np.exp ( log_proposed - log_prev ) )
        u = np.random.rand()
        if u <= alpha:
            x[particle, : ] = proposed
            proposed_prev = proposed
            log_prev = log_proposed
        else:
            x[particle, : ] = proposed_prev
    return x
        
    
    
    

def sequential_mh ( n_particles, x0, \
                    model, model_unc, \
                    observations, obs_unc, \
                    time_axs, obs_time ):

    state_size = len( x0 )
    ensemble = x0
    for timestep in time_axs:
        # Check whether we have an observation for this time step
        if np.in1d ( timestep, obs_time ):
            state[ timestep, :, : ] = assimilate_obs ( timestep, \
                ensemble, model, model_unc, \
                observations[timestep], obs_unc[timestep] )

        else:
            # No observation, just advance the model for all the particles
            for particle in xrange ( n_particles ):
                state[ timestep, particle, : ] = model ( \
                    ensemble[timestep,:], timestep ) + \
                    np.random.randn( state_size )*model_unc
        # Update the ensemble
        ensemble[ particle, : ] = state[ timestep, particle, : ]*1.

def test_dalec():
    
    lat = 44.4 # Latitude
    sla = 111.
    params = np.array([ lat, sla, 0.0000044, 0.47, 0.31, 0.43,0.0027, \
        0.00000206, 0.00248, 0.0228, 0.00000265 ] )
    initial_state = ( [ 58., 102., 770., 40., 9897.] )
    DALEC = Model ( params )
    output = np.zeros ( (7, 1095) )
    for i in xrange(1095):
        nee, gpp, Cf, Cr, Cw, Clit, Csom = DALEC.run_model ( initial_state, i )
        output[:,i] = np.array([nee, gpp, Cf, Cr, Cw, Clit, Csom])
    t = np.arange ( 1095 ) + 1
    titles = ["NEE", "GPP", "Cf", "Cr", "Cw", "Clit", "Csom"]
    fig, axs = plt.subplots ( 3, 3, sharex="col", squeeze=True )
    for i, ax in enumerate ( axs.flatten() ):
        try:
            ax.plot ( t, output[i, :], '-', lw=0.5 )
            ax.set_title ( titles[i] )
            ax.set_xlim ( 1, 1096 )
        except IndexError:
            ax.set_visible ( False )
        