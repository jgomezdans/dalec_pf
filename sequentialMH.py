#!/usr/bin/env python
import numpy as np
import time
from dalec import dalec



__author__ = "J Gomez-Dans"
__version__ = "1.0 (09.03.2015)"
__email__ = "j.gomez-dans@ucl.ac.uk"

    
def safe_log(x, minval=0.0000000001):
    """This functions just does away with numerical 
    warnings."""
    return np.log(x.clip(min=minval))

class Observations ( object ):
    """A storage for the observational data"""
    def __init__ ( self, fname="dalec_drivers.OREGON.MW_obs.dat", verbose=False ):
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
                if verbose:
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
        for i in xrange( d.shape[0] ):
            year = d[i,0]
            if year < 2003:
                lai_obs.append ( d[i,2] )
                lai_unc.append ( d[i,3] )
                lai_time.append ( d[i,1] + (d[i,0]-2000)*365 )
        self.fluxes['lai'] = np.c_[np.array ( lai_time ), np.array ( lai_obs ), \
            np.array ( lai_unc )]
        

    def set_lai_options ( self, sla, lai_unc_scalar, lai_thin ):
        
        self.sla = sla
        self.lai_unc_scalar = lai_unc_scalar
        self.lai_thin = lai_thin
        
    def has_obs ( self, current_timestep, obs_to_assim ):
        """A method to see wether there are observations to assimilate in the
        queried timestep ``current_timestep``, provided these observation
        streams are indeed assimilated according to ``obs_to_assim``
        
        Parameters
        -------------
        current_timestep: int
            The currrent timestep
        obs_to_assim: array
            A boolean array with observation streams to assimilate
            
        Returns
        --------
        Returns True if **any** of the selected assimilation streams in ``obs_to_assim``
        has an observation. Otherwise, returns False
        """
        
        # Loop over all available observations
        for i, obs_stream in enumerate ( ["lai", "cf", "cr", "cw", "cl", "csom"] ):
            if obs_to_assim[i]: # If we are assimilating this observation...
                if np.in1d ( current_timestep, self.fluxes[obs_stream][:,0] ):
                    # We have an observations for a flux in this time step
                    # Stop here and return true
                    return True
        # No observations for this time step in any of the assimilated fluxes...
        # Return false
        return False
    
    def calc_likelihood ( self, proposed_candidate, current_timestep, obs_to_assim ):
        """This method calculates the log-likelihood. The method is a bit specialist, as
        it will deal with some stuff of interest in the LAI record.
        
        Parameters
        ------------
        proposed_candidate: array
            This is the proposed candidate state vector, subset in the order of
            LAI, Cw, Cr, Cf, Cl (we are not assimilating other pools/fluxes)
        current_timestep: int 
            The current timestep (1..1095 in the experiment)
        obs_to_assim: array
            Boolean array with observations to assimilate
        sla: float
            The specific leaf area (or LMA, can't quite remember)
        lai_unc_scalar: float
            A scalar to inflate LAI uncertainty, useful for some experiments
            
        Returns
        --------
        
        The log-likelihood for the particular proposed candidate, for the current time
        step, using all available observations in that timestep (and assuming they are 
        independent).
        """
        
        
        log_proposed = 0.
        for i, obs_stream in enumerate ( ["lai", "cf", "cr", "cw", "cl", "csom"] ):
            if obs_to_assim[i]: # If we are assimilating this observation...
                if np.in1d ( current_timestep, self.fluxes[obs_stream][:,0] ):
                    # Location of the observation in the self.fluxes arrray
                    # Also grab the observation and associated uncertainties
                    # from the relevant array
                    passer = self.fluxes[obs_stream][:,0] == current_timestep
                    obs_unc = self.fluxes[obs_stream][passer,2]
                    obs = self.fluxes[obs_stream][passer,1]
                    if obs_stream == "lai":
                        lai = proposed_candidate[i]/self.sla # Using SLA directly here...
                        if lai < 0:
                            # Out of bounds
                            log_proposed = -np.inf
                        else:
                            # Calculate the (log)likelihood
                            obs_unc *= self.lai_unc_scalar
                            
                            log_proposed += - safe_log ( 2.*np.pi*obs_unc ) - \
                                0.5*( lai - obs)**2/obs_unc**2
                    else:
                        # Not LAI
                        
                        log_proposed += - safe_log ( 2.*np.pi*obs_unc ) - \
                                0.5*( proposed_candidate[i] - obs)**2/obs_unc**2
        return log_proposed
                        
    
        
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
    



def assimilate_obs ( timestep, ensemble, observations, model, model_unc, obs_to_assim ):
    """A function to assimilate all available observations for a particular
    time step, where there is at least one observation available. This
    function takes care of advancing the model (e.g. DALEC) forward, add
    the stochastic forcing, and so on"""
    
    # Get the number of particles
    n_particles, state_size= ensemble.shape
    # The assimilated state will be stored here
    x = ensemble*0. 
    
    # The first particle is the median of the old state to start with
    proposed = np.median ( ensemble, axis=0 )
    log_prev = observations.calc_likelihood ( proposed, timestep, obs_to_assim )
    # Store the initial state
    proposed_prev = proposed
    x[ 0, : ] = proposed
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
        # Calculate the log-likelihood
        log_proposed = observations.calc_likelihood ( proposed, timestep, obs_to_assim )
        
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
                    observations, obs_to_assim, \
                    time_axs ):

    n_particles, state_size = x0.shape
    ensemble = x0
    state = np.zeros ( ( len(time_axs), n_particles, state_size ))
    for timestep in time_axs:
        # Check whether we have an observation for this time step
        if observations.has_obs ( timestep, obs_to_assim ):
            state[ timestep, :, : ] = assimilate_obs ( timestep, \
                ensemble, observations,  model, model_unc, \
                obs_to_assim )

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



def assimilate( sla=110, n_particles=150, Cf0=58., Cr0=102., Cw0=770.,\
         Clit0=40., Csom0=9897., \
         Cfunc=5, Crunc=10, Cwunc=77, Clitunc=20, Csomunc=100, \
         do_lai=True, do_cw=False, do_cr=False, do_cf=False, do_cl=False, do_csom=False, \
         lai_thin=0, lai_unc_scalar=1.):
    
    t0 = time.time()
    # The following sets the fluxes we would like to assimilate
    obs_to_assim = np.zeros(6).astype ( np. bool )
    obs_to_assim[0] = do_lai
    obs_to_assim[1] = do_cf
    obs_to_assim[2] = do_cr
    obs_to_assim[3] = do_cw
    obs_to_assim[4] = do_cl
    obs_to_assim[5] = do_csom
    
    model_unc=np.array([Cfunc, Crunc, Cwunc, Clitunc, Csomunc])
    lat = 44.4 # Latitude
#    sla = 110.
#    n_particles = 500
    #params = np.array([ lat, sla, 4.4e-6, 0.47, 0.31, 0.43,0.0027, \
        #0.00000206, 0.00248, 0.0228, 0.00000265 ] )
    params = np.array([ lat, sla, 4.4e-6, 0.47, 0.31, 0.43,0.0010, \
        0.00000206, 0.00248, 0.0228, 0.00000265 ] )

    # Initial pool composition
    x0 = np.array( [ Cf0, Cr0, Cw0, Clit0, Csom0 ] )
    
    DALEC = Model ( params )
    
    observations = Observations ()
    observations.set_lai_options ( sla, lai_unc_scalar, lai_thin )
    
    s0 = x0[:, None] + \
       np.random.randn(5, n_particles)*model_unc[:, None]
    s0 = s0.T
    
    results = sequential_mh ( s0, \
                    DALEC.run_model, model_unc, \
                    observations, obs_to_assim, \
                    np.arange(1095) )
    print "Assimilation done in %d seconds" % ( time.time() - t0 )
    return DALEC, observations, results
    
if __name__ == "__main__":
    from plot_utils import pf_plots
    DALEC, observations, results = assimilate()
    pf_plots ( DALEC, observations, results )