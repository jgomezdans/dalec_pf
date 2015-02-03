#!/usr/bin/env python
import numpy as np
from dalec import dalec

class Model ( object ):
    """A class for updating the DALEC model"""
    def __init__ ( self, model_params, drivers="dalec_drivers.OREGON.no_obs.dat" ):
        self.model_params = model_params
        self.drivers = self._process_drivers ( drivers )
        self.previous_state = []

    def _process_drivers ( self, drivers ):
        """A method to e.g. read meteo drivers"""
        return np.loadtxt ( drivers )

    def run_model ( self, x, i ):
        """Runs the model forward to ``i+1`` using input state ``x``, 
        parameter vector ``theta``"""
        # Extract drivers...

        doy, temp, tmx, tmn, irad, psid, ca, rtot, nitro = self.drivers[i,:]
        tmp = 0.5* (tmx-tmn)

        
        Cf, Cr, Cw, Clit, Csom = x
        ( nee, gpp, Cf, Cr, Cw, Clit, Csom, lai ) = dalec ( doy, \
            tmn, tmp, tmx, irad, ca, nitro, \
            self.model_params[0], self.model_params[1], \
            self.model_params[2:], \
            Cf, Cr, Cw, Clit, Csom, psid=psid, rtot=rtot )

        self.previous_state.append ( [ Cf, Cr, Cw, Clit, Csom ] )

        return nee, gpp, Cf, Cr, Cw, Clit, Csom



if __name__ == "__main__":
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
