#!/usr/bin/env python
"""
Mat Williams ACM & DALEC models
That's all folks.
"""


import numpy as np
import matplotlib.pyplot as plt

from plot_utils import pretty_axes

def acm ( lat, lai, doy, tmx, tmn, irad,  ca, nitrogen, \
            a = np.array( [ 2.155, 0.0142, 217.9, 0.980, 0.155, 2.653, \
            4.309, 0.060, 1.062, 0.0006]),
            psid=-2., rtot=1.):
    """
    The ACM model. The default parameters in ``a`` are taken from
    Williams et al (2005), and are an option. Otherwise, the ACM model
    takes a number of meteo and evironmental inputs. For a full description
    of the model, see Williams et al (1997) 
    http://www.geosciences.ed.ac.uk/homes/mwilliam/WilliamsEA97.pdf
    or the appendix of Williams et al (2005).
    
    Parameters
    -----------
    lat: float
        Latitude of the site in decimal degrees [deg]
    lai: float
        The leaf area index (LAI) [m^2/m^2]
    doy: int
        The Day of year [day]
    tmx: float
        Maximum daily temperature [degC]
    tmn: float
        Minimum daily temperature [degC]
    irad: float
        Irradiance [MJ m^{-2} day^{-1}].
    ca: float
        CO2 concentration [ppm]
    nitrogen: float
        Average leaf nitrogen [gNm^{2}leaf area]
    a: array
        ACM parameters
    psid: float
        Maximum soil-leaf water potential difference [MPa]
    rtot: float
        Total plant-soil hydraulic resistance [MPa m^2 s mmol^{-1}]
    
    Returns
    ---------
    GPP
    
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
    and stuff. In order to calculate GPP, it uses the ACM model, which is in
    a different function, called ``acm``.
    
    Parameters
    -----------
    
    doy: int
        The Day of year [day]
    tmn: float
        Minimum daily temperature [degC]
    tmx: float
        Maximum daily temperature [degC]
    irad: float
        Irradiance [MJ m^{-2} day^{-1}].    
    ca: float
        CO2 concentration [ppm]
    nitrogen: float
        Average leaf nitrogen [gNm^{2}leaf area]
    lat: float
        Latitude of the site in decimal degrees [deg]
    lma: float
        Leaf mass per unit area [gm^2leaf area]
    p: array
        Parameter vector
    Cf: float
        Foliar C pool (gCm^2)
    Cr: float
        Root C pool (gCm^2)
    Cw: float
        Woody C pool (gCm^2)
    Clit: float
        Litter C pool (gCm^2)
    Csom: float
        SOM C pool (gCm^2)
    psid: float
        Maximum soil-leaf water potential difference [MPa]
    rtot: float
        Total plant-soil hydraulic resistance [MPa m^2 s mmol^{-1}]
    a: array
        ACM parameters
    lai: float (optional)
        You might drive the model directly with LAI, rather than with Cf/lma...
        
    Returns
    ---------
    
    ( nee, gpp, Ra, Rh1, Rh2, Af, Ar, Aw, Lw, Lr, D, \
            Cf, Cr, Cw, Clit, Csom, lai )    """
            
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
    
    return ( nee, gpp, Ra, Rh1, Rh2, Af, Ar, Aw, Lw, Lr, D, \
            Cf, Cr, Cw, Clit, Csom, lai )
    
def test_dalec():
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

    outputs = np.zeros( (17, len (doys )) )
    for (i, doy ) in enumerate ( doys ) :
         ( nee, gpp, Ra, Rh1, Rh2, Af, Ar, Aw, Lw, Lr, D, \
            Cf, Cr, Cw, Clit, Csom, lai ) = dalec ( doy, \
            tmn[i], tmp[i], tmx[i], irad[i], ca[i], nitrogen[i], \
            lat, lma, p_vect, Cf, Cr, Cw, Clit, Csom, \
            psid=psid[i], rtot=rtot[i] )
         outputs[:, i] =  ( nee, gpp, Ra, Rh1, Rh2, Af, Ar, Aw, Lw, Lr, D, \
            Cf, Cr, Cw, Clit, Csom, lai )
    tx = np.arange ( len(doys))

    pools = [r'$NEE$', r'$GPP$',  '$Ra$', '$Rh_1 + Rh_2$', '$A_f$','$A_r$', \
        '$A_w$','$L_f$', '$L_r$', '$L_w$', '$D$',\
        r'$C_f$',r'$C_r$',r'$C_w$',r'$C_{lit}$',r'$C_{SOM}$']

    fig, axs = plt.subplots (nrows=4, ncols=4, sharex="col", figsize=(11,13) )
    
    for i, ax in enumerate(axs.flatten() ):
        pretty_axes ( ax )
        ax.plot ( tx, outputs[ i, :], '-' )
        ax.set_title (pools[i], fontsize=12 )
        
        try:
            if pools[i] == r'$C_f$':
                d = np.loadtxt ( "meas_flux_cf.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$C_{lit}$':
                d = np.loadtxt ( "meas_flux_cl.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$C_w$':
                d = np.loadtxt ( "meas_flux_cw.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$C_r$':
                d = np.loadtxt ( "meas_flux_cr.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$L_f$':
                d = np.loadtxt ( "meas_flux_lf.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                
            elif pools[i] == r'$GPP$':
                d = np.loadtxt ( "meas_flux_gpp.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none")
                
            elif pools[i] == r'$NEE$':
                d = np.loadtxt ( "meas_flux_nee.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none")
                
            elif pools[i] == r'$Ra$':
                d = np.loadtxt ( "meas_flux_ra.txt.gz" )
                ax.plot ( d[:, 0], d[:,1], 'ko', mfc="none" )
                

        except:
            pass

        ax.set_xlim ( 0, 1100 )
        ax.xaxis.set_ticks([1,365, 365*2, 365*3])
    plt.subplots_adjust ( wspace=0.4 )
    return  outputs
    