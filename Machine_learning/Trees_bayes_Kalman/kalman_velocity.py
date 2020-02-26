#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:10:33 2019

@author: czoppson
"""
import numpy as np 
import seaborn as sns; sns.set()
fmri = sns.load_dataset("fmri")







import matplotlib.pyplot as plt
#@title Kalman Simulationb
initial_sigma = 10 #@param
motion_sigma = 5 #@param
gps_sigma = 20 #@param
n_steps = 50 #@param

loc_df = simulate(initial_sigma, motion_sigma, gps_sigma, n_steps)
predictions_df = kalman_predict(loc_df, initial_sigma, motion_sigma, gps_sigma)
plt.plot(loc_df.x, 'r', label='true position')
plt.plot(loc_df.gps, 'go', label='gps readout')
plt.plot(predictions_df.mu, 'b', label='kalman position')
plt.fill_between(range(len(predictions_df)),
                 predictions_df.mu + predictions_df.sigma,
                 predictions_df.mu - predictions_df.sigma, color='b', alpha=0.2)
plt.fill_between(range(len(predictions_df)),
                 predictions_df.mu + 3 * predictions_df.sigma,
                 predictions_df.mu - 3 * predictions_df.sigma, color='b', alpha=0.1)
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=True)
plt.xlabel('time')
plt.ylabel('position')
plt.title('Kalman filtering of location data')
None



import pandas as pd


def simulate(initial_sigma, motion_sigma, gps_sigma, n_steps,velocity_sigma):
    """Simulate a sequence of locations and noisy GPS measurements

    Args:
        initial_sigma, motion_sigma, gps_sigma: parameters of the simulation
        n_steps: number of timesteps

    Returns:
        a DataFrame with columns 'x' and 'gps' giving the true location and 
        gps readouts.
    """
    
    # Sample an initial location from the distribution ovetr the initial loc
    x = np.random.normal(0,initial_sigma)
    #initial velocity
    v = 0
    loc_hist = []
    for s in range(n_steps):
        # TODO: sample a new x and gps readout
        velocity = v + np.random.normal(0,velocity_sigma)
        x = np.random.normal(x,motion_sigma) + v
        gps_readout = np.random.normal(x,gps_sigma)
        loc_hist.append((x, gps_readout,velocity))
    loc_df = pd.DataFrame(loc_hist, columns=['x', 'gps','velocity'])
    return loc_df


initial_sigma = 10 #@param
motion_sigma = 5 #@param
gps_sigma = 10
velocity_sigma = 5 #@param
n_steps = 50

loc_df= simulate(initial_sigma, motion_sigma, gps_sigma, n_steps,velocity_sigma)
loc_df.plot()

### W tej funkcji jeszce nic nie zmieniłem
def kalman_predict(loc_df, initial_sigma, motion_sigma, gps_sigma,velocity_sigma):
    # Set our initial belief about our location
    prior_mu = 0
    prior_sigma = initial_sigma 
    predictions = []
    for gps_readout in loc_df.gps:
        # expand the prior by the movement
        prior_sigma =np.sqrt(prior_sigma**2 + motion_sigma**2)
        # now do the bayes update
        posterior_mu = (gps_readout * prior_sigma**2 + prior_mu * gps_sigma**2)/(gps_sigma**2 + prior_sigma**2)
        posterior_sigma  =np.sqrt( 1/((1/prior_sigma**2) +(1/gps_sigma**2)))
        predictions.append((posterior_mu, posterior_sigma))
        prior_mu  = posterior_mu
        prior_sigma = posterior_sigma

    predictions_df = pd.DataFrame(predictions, columns=['mu', 'sigma'])
    return predictions_df




import matplotlib.pyplot as plt
#@title Kalman Simulationb
initial_sigma = 10 #@param
motion_sigma = 5 #@param
gps_sigma = 20 #@param
n_steps = 50 #@param
velocity_sigma = 5 

loc_df = simulate(initial_sigma, motion_sigma, gps_sigma, n_steps, velocity_sigma)
predictions_df = kalman_predict(loc_df, initial_sigma, motion_sigma, gps_sigma)
plt.plot(loc_df.x, 'r', label='true position')
plt.plot(loc_df.gps, 'go', label='gps readout')
plt.plot(predictions_df.mu, 'b', label='kalman position')
plt.plot(loc_df.velocity,'k' ,label = 'velocity' )
plt.fill_between(range(len(predictions_df)),
                 predictions_df.mu + predictions_df.sigma,
                 predictions_df.mu - predictions_df.sigma, color='b', alpha=0.2)
plt.fill_between(range(len(predictions_df)),
                 predictions_df.mu + 3 * predictions_df.sigma,
                 predictions_df.mu - 3 * predictions_df.sigma, color='b', alpha=0.1)
plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=True)
plt.xlabel('time')
plt.ylabel('position')
plt.title('Kalman filtering of location data')
None
