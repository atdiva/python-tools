#!/usr/bin/env python
"""
An example of how to use bilby to perform parameter estimation for
non-gravitational wave data. In this case, fitting a linear function to
data with background Gaussian noise

"""
import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils.random import rng, seed
seed(123)

# setup
label = "linear_regression"
outdir = "outdir"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


#signal model
def model(time, m, c):
    return time * m + c


# inject signal
injection_parameters = dict(m=0.5, c=0.2)

#time domain setup 
sampling_frequency = 10
time_duration = 10
time = np.arange(0, time_duration, 1 / sampling_frequency)
N = len(time)
sigma = 0.1
data = model(time, **injection_parameters) + rng.normal(0, sigma, N) #gaussian random normal noise


# gaussian likelihood
likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

# priors
priors = dict()
priors["m"] = bilby.core.prior.Uniform(0, 5, "m")
priors["c"] = bilby.core.prior.Uniform(-2, 2, "c")

# run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=250,
    injection_parameters=injection_parameters,
    outdir=outdir,
    label=label,
)

# corner plot
result.plot_corner()