# Experiments with the DALEC ecosystem model and a sequential Metropolis particle filter

## J Gómez-Dans (NCEO & UCL)

This repository contains Python code to carry out some computational experiments with an
ecosystem model (the [DALEC](http://www.geos.ed.ac.uk/homes/mwilliam/DALEC.html) of Mat Williams in Edinburgh) using a particle filter. The particular flavour of particle filter that's being used is the sequential Metropolis-Hastings filter, described in [Dowd (2007)](http://www2.geog.ucl.ac.uk/~mdisney/teaching/teachingNEW/methods/diff/Dowd.Bayesian_DA.JMSys.2007.pdf). You can fidn the Python code files, but also an IPython notebook with a lab on exploring this DA frameweork. An HTML-rendered version of the IPython notebook is available [here](http://jgomezdans.github.io/dalec_pf/DA_practical.html), or you can see it in [nbviewer](http://nbviewer.ipython.org/github/jgomezdans/dalec_pf/blob/master/DA_practical.ipynb).

Also some <a href="http://jgomezdans.github.io/dalec_pf/PF_presentation.slides.html">introductory slides</a>.

### Using on UCL's UNIX system

Note that in order to use the notebook, we need to do the following:
    
    source activate ipy3
    
This lets you use the IPython notebook version 3.0.0. To have access to the notebooks etc, you can change the directory to your data storage (typically ``~/DATA``), and

    git clone https://github.com/jgomezdans/dalec_pf.git
    
If you're running this on a computer without ``git``, you can just download [the zipfile](https://github.com/jgomezdans/dalec_pf/archive/master.zip). Note that the only dependencies here are

* IPython notebook (>= v 3.0.0)
* Matplotlib
* Numpy
* Scipy

These are all available from e.g. [Anaconda](http://continuum.io/downloads) standard distributions.
