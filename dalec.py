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
    The ACM model. 
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
    #dayl = m
    #dayl[dayl >= 1.0] = 24.0
    #dayl[dayl <= -1.0] = 0.0
    if dayl > -1. or dayl < 1.:
        dayl = 24.*np.arccos(-m)/np.pi
    else:
        dayl
    #dayl = np.where(np.logical_or(dayl < 1.0, dayl > -1.0), \
        #24.0 * np.arccos(-m) / np.pi, dayl )
    
   

    cps = e0 * irad * gs * (ca - ci) / (e0 * irad + gs * (ca - ci))
    gpp = cps * (a[1] * dayl + a[4])

    return gpp
    
def dalec ( doy, tmn, tmp, tmx, irad, ca, nitrogen, \
            lat, sla, p, \
            Cf, Cr, Cw, Clit, Csom, psid=-2, rtot=1, a=None, lai=None ):
    """
    DALEC, the function
    ---------------------
    
    This function is the daily update of the model. It takes a number of drivers
    as well as the state of the C pools, and some parameters that control rates
    and stuff.
    
    
    """
    if lai is None:
        lai = max ( 0.1, Cf/sla ) # LAI based on foliar pool
    
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
    sla = 111.
    Cf = 58
    Cw = 770
    Cr = 102
    Clit = 40
    Csom = 9897
    p_vect=np.array([ 0.0000044, 0.47, 0.31, 0.43,0.0027, 0.00000206, 0.00248, \
                0.0228, 0.00000265 ] )

    GPP = []
    NEE = []
    CF = []
    for (i, doy ) in enumerate ( doys ) :
        ( nee, gpp, Cf, Cr, Cw, Clit, Csom, lai ) = dalec ( doy, \
            tmn[i], tmp[i], tmx[i], irad[i], ca[i], nitrogen[i], \
            lat, sla, p_vect, \
            Cf, Cr, Cw, Clit, Csom, psid=psid[i], rtot=rtot[i] )
        NEE.append ( nee )
        CF.append ( Cf )
        
    plt.plot( NEE)
    plt.show()
