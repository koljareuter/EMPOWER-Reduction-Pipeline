from __future__ import print_function
import bottleneck as bn
import numpy as np
import os
from astropy.modeling import (models, fitting)

__all__ = ["fit_psf"]


def fit_psf(image):
    #beta_fact = 2.*np.sqrt(2.**(1./2.7)-1) #to convert FWHM to alpha

    #initialize the fitter
    fit = fitting.LevMarLSQFitter()
    
    #first pass: 1D fits along each axis to get initial guesses
    xsum = bn.nansum(image, axis=0)
    ysum = bn.nansum(image, axis=1)

    gx_init = models.Gaussian1D(amplitude=max(xsum), mean=np.arange(14)[np.argmax(xsum[1:-1])+1], stddev=1.2) +\
              models.Polynomial1D(degree=1)
    gy_init = models.Gaussian1D(amplitude=max(ysum), mean=np.arange(14)[np.argmax(ysum[1:-1])+1], stddev=1.2) +\
              models.Polynomial1D(degree=1)

    model_x = fit(gx_init, np.arange(1,13), xsum[1:-1])
    model_y = fit(gy_init, np.arange(1,13), ysum[1:-1])


    #second pass: 2D Guassian fit, try multiple guesses for initial parameters
    amp_init = (model_x.amplitude_0.value + model_y.amplitude_0.value)/2.
    x_init, y_init = model_x.mean_0.value, model_y.mean_0.value
    x_std, y_std = model_x.stddev_0.value, model_y.stddev_0.value

    ix, iy = image.shape
    wts = np.ones_like(image)
    wts[0,:], wts[-1,:], wts[:,0], wts[:,-1] = 0., 0., 0., 0.
    ygrid, xgrid = np.mgrid[:ix, :iy]

    res_total = np.zeros(4, dtype=float)
    fparams = []
    fmodels = []
    icnt = 0
    #for dx in np.arange(-1.0,1.1,1.0):
    #    for dy in np.arange(-1.0,1.1,1.0):
    for dx in np.arange(-0.5,0.6,1.0):
        for dy in np.arange(-0.5,0.6,1.0):
            g_init = models.Gaussian2D(amplitude=amp_init,
                                       x_mean=x_init + dx, y_mean=y_init + dy,
                                       x_stddev=x_std, y_stddev=y_std,
                                       theta=None) +\
                     models.Polynomial2D(degree=1)

            model = fit(g_init, xgrid, ygrid, image, weights=wts, maxiter=1000)

            #generate separate polynomial and source models
            source_mod = models.Gaussian2D(amplitude=model.amplitude_0.value, 
                                           x_mean=model.x_mean_0.value,
                                           y_mean=model.y_mean_0.value,
                                           x_stddev=model.x_stddev_0.value,
                                           y_stddev=model.y_stddev_0.value,
                                           theta=model.theta_0.value)
            poly_mod = models.Polynomial2D(degree=2, 
                                           c0_0=model.c0_0_1.value,
                                           c0_1=model.c0_1_1.value,
                                           c1_0=model.c1_0_1.value)

            fmodels.append((source_mod(xgrid,ygrid),poly_mod(xgrid,ygrid)))
            fparams.append([model.x_mean_0.value, model.y_mean_0.value, 
                           model.x_stddev_0.value*2.355, model.y_stddev_0.value*2.355 
                          ])
            res_total[icnt] = np.sum(np.abs(image-model(xgrid,ygrid))[1:-1,1:-1])
            icnt += 1

    return fparams[np.argmin(res_total)], fmodels[np.argmin(res_total)] 



