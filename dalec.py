#!/usr/bin/env python
"""
Mat Williams ACM & DALEC models
That's all folks.
"""


import numpy as np
import matplotlib.pyplot as plt

def acm ( lat, lai, doy, tmx, tmn, irad,  ca, nitrogen, \
            a = np.array( [ 2.155, 0.0142, 217.9, 0.980, 0.155, 2.653, \
            4.309, 0.060, 1.062, 0.0006]),
            psid=-2., rtot=1.):
    """
    The ACM model. The default parameters in ``a`` are taken from
    Williams et al (2005), 
    """
    tmpr = 0.5 * (tmx - tmn )
    
    gs = np.abs(psid)**a[9] / (a[5] * rtot + tmpr)
    pp = lai * nitrogen / gs * a[0] * np.exp(a[7] * tmx)
    qq = a[2] - a[3]
    ci = 0.5 * (ca + qq - pp + np.sqrt((ca + qq - pp)**2 - 4.0 * (ca * qq - pp * a[2])))
    e0 = (a[6] * lai**2) / (lai**2 + a[8])
    dec = -23.4 * np.cos((360.0 * (doy + 10.0) / 365.0) * np.pi / 180.0) * np.pi / 180.0
    m = np.tan(lat * np.pi / 180.0) * np.tan(dec)
    if m >= 1:
        dayl = 24.
    elif m <= -1:
        dayl = 0.
    else:
        dayl = m
        
    if dayl > -1. or dayl < 1.:
        dayl = 24.*np.arccos(-m)/np.pi
    else:
        dayl    
   

    cps = e0 * irad * gs * (ca - ci) / (e0 * irad + gs * (ca - ci))
    gpp = cps * (a[1] * dayl + a[4])

    return gpp
    
def dalec ( doy, tmn, tmp, tmx, irad, ca, nitrogen, \
            lat, lma, p, \
            Cf, Cr, Cw, Clit, Csom, psid=-2, rtot=1, a=None, lai=None ):
    """
    DALEC, the function
    ---------------------
    
    This function is the daily update of the model. It takes a number of drivers
    as well as the state of the C pools, and some parameters that control rates
    and stuff.
    
    
    """
    if lai is None:
        lai = max ( 0.1, Cf/lma ) # LAI based on foliar pool
    
    if a is None:
        gpp = acm ( lat, lai, doy, tmx, tmn, irad, ca, nitrogen, \
                psid=psid, rtot=rtot )
    else:
        gpp = acm ( lat, lai, doy, tmx, tmn, irad, ca, nitrogen, a=a, \
            psid=psid, rtot=rtot )
        
    temp_rate = 0.5* np.exp ( 0.0692*0.5*( tmx + tmn ) )
    
    Ra = p[1]*gpp  # Autotrophic respiration
    Af = (gpp-Ra)*p[2] # Foliate assimilate? 
    Ar = (gpp-Ra-Af)*p[3] # Root assimilate?
    Aw = gpp-Ra-Af-Ar  # Woody assimilate
    Lf = p[4]*Cf  # Foliar litter
    Lw = p[5]*Cw  # Woody litter                  
    Lr = p[6]*Cr  # root litter                  
    Rh1 = p[7]*Clit*temp_rate  # Heterotrophic respiration  litter
    Rh2 = p[8]*Csom*temp_rate  # Heterotrophic respiration soil
    D = p[0]*Clit*temp_rate # Litter sometnihg or other
    
    Cf = Cf + Af - Lf                                               
    Cw = Cw+ Aw - Lw                                                        
    Cr = Cr+ Ar - Lr                                                 
    Clit = Clit + Lf + Lr - Rh1 - D          
    Csom = Csom+ D - Rh2 +Lw     
    
    nee = Ra+Rh1+Rh2-gpp
    
    return ( nee, gpp, Cf, Cr, Cw, Clit, Csom, lai )
    
if __name__ == "__main__":
    # Meteolius First Young Pine
    driver_data = np.loadtxt ( "dalec_drivers.OREGON.no_obs.dat" )
    doys = driver_data[ :, 0 ] 
    temp = driver_data [ :, 1] 
    tmx = driver_data [ :, 2]
    tmn = driver_data [ :, 3] 
    tmp = 0.5 * ( tmx - tmn )
    irad = driver_data [ :, 4]
    psid = driver_data [ :, 5 ]
    ca = driver_data [ :, 6]
    rtot = driver_data [ :, 7]
    nitrogen = driver_data [ :, 8]
    lat = 44.4
    lma = 111.
    Cf = 57.705
    Cw = 769.86
    Cr = 102.00
    Clit = 40.449
    Csom = 9896.7
    # These parameters are from Williams et al (2005)
    p_vect=np.array([ 4.41e-6, 0.47, 0.3150, 0.4344,0.002665, 2.06e-6, 2.48e-3, \
                2.28e-2, 2.65e-6 ] )

    GPP = []
    NEE = []
    CF = []
    for (i, doy ) in enumerate ( doys ) :
        ( nee, gpp, Cf, Cr, Cw, Clit, Csom, lai ) = dalec ( doy, \
            tmn[i], tmp[i], tmx[i], irad[i], ca[i], nitrogen[i], \
            lat, lma, p_vect, \
            Cf, Cr, Cw, Clit, Csom, psid=psid[i], rtot=rtot[i] )
        NEE.append ( nee )
        CF.append ( Cf )
        
    plt.plot( NEE)
    plt.show()
