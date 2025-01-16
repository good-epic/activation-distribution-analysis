#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import plotly.express as px
import matplotlib.pyplot as plt
from functools import partial, lru_cache
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
import os
# from openai import OpenAI
import json
import numpy as np
from tqdm.auto import tqdm  # Automatically chooses between notebook and terminal versions

from concurrent.futures import ThreadPoolExecutor, as_completed
    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import pandas as pd
from dataclasses import dataclass
from numpy.linalg import norm

from datetime import datetime
import re
from math import isnan as is_nan
#from pdffit.distfit import BestFitDistribution as BFD

import scipy.stats as st
from scipy.optimize import curve_fit
import statsmodels.api as sm
from scipy.stats._continuous_distns import _distn_names
from scipy import signal
import time

import sys
import traceback
import warnings
warnings.filterwarnings('ignore', 'Tight layout not applied')
import logging
from typing import Callable, Tuple, List
from collections.abc import Sequence
import random
from itertools import chain
import ast
import gc

from IPython import display
import matplotlib
plt.close('all')
matplotlib.use('inline')


class BestFitDistribution:
    """
    Optimized version of distribution fitting class
    """
    def __init__(
        self,
        data,
        n_bins: int = 200,
        distributionNames: list = None,
        debug: bool = False,
    ):
        self.debug = debug
        
        # Pre-compute histogram once
        self.n_bins = n_bins
        self.y, bins = np.histogram(data, bins=self.n_bins, density=True)
        self.bins = bins
        self.x = (bins[:-1] + bins[1:]) / 2.0
        
        # Store data statistics to help with initial parameter guesses
        self.data = data
        self.data_min = np.min(self.data)
        self.data_max = np.max(self.data)
        stdesc = st.describe(self.data)
        self.data_mean = stdesc.mean
        self.data_std = np.sqrt(stdesc.variance)
        self.data_skew = stdesc.skewness
        self.data_kurtosis = stdesc.kurtosis
        q75, self.data_median, q25 = np.percentile(self.data, [75, 50, 25])
        self.data_iqr = q75 - q25        
        
        if distributionNames is None:
            self.distributionNames = ['alpha', 'beta', 'crystalball', 'dgamma', 'dweibull', 'genlogistic', 'gennorm',
                                      'hypsecant', 'jf_skew_t', 'johnsonsb', 'johnsonsu', 'laplace',
                                      'laplace_asymmetric', 'loggamma', 'logistic', 'moyal', 'nakagami', 'nct',
                                      'norm', 'norminvgauss', 'pearson3', 'powernorm', 'skewnorm', 't', 'truncnorm',
                                      'vonmises', 'vonmises_line']
        else:
            self.distributionNames = distributionNames
            
        # Pre-compute distribution objects
        self.distributions = {
            name: getattr(st, name) for name in self.distributionNames
        }
        
        # Initial parameter guess functions for common distributions
        self.param_guesses = {
            'norm': lambda: (self.data_mean, self.data_std),
            'gamma': lambda: ((self.data_mean/self.data_std)**2, 
                               self.data_mean/(self.data_std**2)),
            'lognorm': lambda: (np.log(self.data_std), np.log(self.data_mean)),
            'expon': lambda: (self.data_mean,),
            'beta': lambda: (self.data_mean * 2, (1 - self.data_mean) * 2),
            'logistic': lambda: (
                self.data_median,  # loc parameter (center)
                self.data_iqr / (2 * np.log(3))  # scale parameter
            ),
            'genlogistic': lambda: (
                1.0 + np.abs(self.data_skew),  # shape parameter (c), adjusted for skewness
                self.data_median,  # location
                self.data_iqr / (2 * np.log(3))  # scale
            ),
            'hypesecant': lambda: (
                self.data_median,  # location
                self.data_iqr / (2 * np.arctanh(0.5))  # scale
            ),
            'nct': lambda: (
                max(2.1, 4 + self.data_kurtosis),  # df (degrees of freedom)
                self.data_skew * np.sqrt(max(2.1, 4 + self.data_kurtosis)),  # non-centrality
                self.data_median,  # location
                self.data_std  # scale
            ),
            'jf_skew_t': lambda: (
                max(-3, min(self.data_skew, 3)),  # skewness: more bounded
                max(2.1, min(self.data_kurtosis + 3, 30)),  # df: more conservative bounds
                self.data_median,  # location
                max(0.1, self.data_std)  # scale: ensure positive
            ),
           't': lambda: (
                max(2.1, min(30, 6 / (self.data_kurtosis/4 + 1))),  # df
                self.data_median,  # location
                self.data_std * np.sqrt((max(2.1, min(30, 6/(self.data_kurtosis/4 + 1))) - 2) 
                                      / max(2.1, min(30, 6/(self.data_kurtosis/4 + 1))))  # scale
            ),
            'crystalball': lambda: (
                1.0,  # beta: start with a more conservative value
                max(0.1, min(abs(self.data_skew), 3.0)),  # m: bounded skew
                self.data_median,  # location
                max(0.1, self.data_std)  # scale: ensure positive
            )
        }
        #print(self.param_guesses['logistic'])

    def get_initial_guess(self, dist_name):
        """Get initial parameter guess for optimization"""
        if dist_name in self.param_guesses:
            return self.param_guesses[dist_name]()
        return None

    def fit_single_distribution(self, distribution, dist_name):
        """Fit a single distribution with optimized settings"""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # Get initial guess if available
                initial_guess = self.get_initial_guess(dist_name)
                # Optimize with better initial conditions and bounds
                if initial_guess is not None:
                    # scipy annoying takes positional shape parameters but requires named loc and scale
                    loc, scale = initial_guess[-2:]  # Last two parameters are loc, scale
                    args = initial_guess[:-2]  # Any remaining parameters
                    if args:
                        params = distribution.fit(self.data, *args, loc=loc, scale=scale)
                    else:
                        params = distribution.fit(self.data, loc=loc, scale=scale)
                else:
                    params = distribution.fit(self.data)
                # Separate parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Vectorized PDF calculation
                pdf = distribution.pdf(self.x, loc=loc, scale=scale, *arg)
                
                # Fast SSE calculation using numpy
                sse = np.sum(np.square(self.y - pdf))
            
            return {
                "dist": distribution,
                "dist_name": dist_name,
                "params": params,
                "sse": sse
            }
                
        except Exception as ex:
            if self.debug:
                print(f"Fit failed for {dist_name}: {ex}\n", file=sys.stderr)
            return None

    def best_fit_distribution(self):
        """
        Optimized version of distribution fitting
        """
        # Use ThreadPoolExecutor for parallel fitting
        best_distributions = []
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_dist = {
                executor.submit(self.fit_single_distribution, dist, name): name 
                for name, dist in self.distributions.items()
            }
            
            for future in as_completed(future_to_dist):
                result = future.result()
                if result is not None:
                    best_distributions.append(result)
                    
        return sorted(best_distributions, key=lambda x: x["sse"])

    def make_pdf(self, dist, params: list):
        """
        Optimized PDF generation
        """
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        
        # Return vectorized function instead of lambda
        return np.vectorize(
            lambda x: dist.pdf(x, *arg, loc=loc, scale=scale)
        )


def model_function(x, beta, alpha, gamma):
    return gamma * np.exp(-beta * x**alpha)


## Used only in data and results exploration, not main run
def plot_main_dist_and_tail_exps(all_acts        : np.ndarray,
                 best_dist       : dict,
                 outer_key       : str,
                 inner_key       : str,
                 hook_name       : str,
                 layer_num       : str,
                 fig_outdir      : str,
                 n_bins          : int,
                 n_sd_plus_left  : float=4.0,
                 n_sd_plus_right : float=10.0):
    
    plt.figure(figsize=(12, 8))
    counts, bins, _ = plt.hist(all_acts, bins=n_bins, density=True, label='Post Residuals Activations', color="red", alpha=0.6)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    x = np.linspace(min(all_acts), max(all_acts), 1000)
    y = best_dist['dist'].pdf(x, *best_dist['params'])
    shape_str = '' if len(best_dist['params']) < 3 else '\nShape = ' + ', '.join([f'{p:.3g}' for p in best_dist['params'][:-2]])
    loc_str = f'{best_dist["params"][-2]:.3g}'
    scale_str = f'{best_dist["params"][-1]:.3g}'
    hist_label = (f'Fitted {best_dist["dist"].name.title()}\n' +
                f'Loc = {loc_str}\nScale = {scale_str}{shape_str}')
    plt.plot(x, y, lw=2, ls='-', color='green', label=hist_label)
    
    plt.title(f'Distribution for {outer_key} : {inner_key}\nLayer {layer_num} ({hook_name})')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{fig_outdir}/dist_hist_layer_{layer_num}_{outer_key.replace(" ", "_")}__{inner_key.replace(" ", "_")}.png')


    aa_mu  = np.mean(all_acts)
    aa_std = np.std(all_acts)
    cutp2 = aa_mu + 2 * aa_std
    cutp3 = aa_mu + 3 * aa_std
    cutp4 = aa_mu + 4 * aa_std
    cutm2 = aa_mu - 2 * aa_std
    cutm3 = aa_mu - 3 * aa_std
    cutm4 = aa_mu - 4 * aa_std

    inds1p2 = np.where(bin_centers > cutp2)[0]
    inds2p2 = inds1p2[np.where(counts[inds1p2] > 0)]
    (beta_fit_p2, alpha_fit_p2,
    gamma_fit_p2, x_min_p2, x_max_p2) = get_tail_weight(dist=None, params=None,
                                                        x_values=bin_centers[inds2p2],
                                                        y_values=counts[inds2p2],
                                                        n_sd_plus_left=n_sd_plus_left,
                                                        n_sd_plus_right=n_sd_plus_right)

    inds1p3 = np.where(bin_centers > cutp3)[0]
    inds2p3 = inds1p3[np.where(counts[inds1p3] > 0)]
    (beta_fit_p3, alpha_fit_p3,
    gamma_fit_p3, x_min_p3, x_max_p3) = get_tail_weight(dist=None, params=None,
                                                        x_values=bin_centers[inds2p3],
                                                        y_values=counts[inds2p3],
                                                        n_sd_plus_left=n_sd_plus_left,
                                                        n_sd_plus_right=n_sd_plus_right)

    inds1p4 = np.where(bin_centers > cutp4)[0]
    inds2p4 = inds1p4[np.where(counts[inds1p4] > 0)]
    (beta_fit_p4, alpha_fit_p4,
    gamma_fit_p4, x_min_p4, x_max_p4) = get_tail_weight(dist=None, params=None,
                                                        x_values=bin_centers[inds2p4],
                                                        y_values=counts[inds2p4],
                                                        n_sd_plus_left=n_sd_plus_left,
                                                        n_sd_plus_right=n_sd_plus_right)

    inds1m2 = np.where(bin_centers < cutm2)[0]
    inds2m2 = inds1m2[np.where(counts[inds1m2] > 0)]
    xvals2 = sorted([-1.0 * x for x in bin_centers[inds2m2]])
    (beta_fit_m2, alpha_fit_m2,
    gamma_fit_m2, x_min_m2, x_max_m2) = get_tail_weight(dist=None, params=None,
                                                        x_values=xvals2,
                                                        y_values=list(counts[inds2m2][::-1]),
                                                        n_sd_plus_left=n_sd_plus_left,
                                                        n_sd_plus_right=n_sd_plus_right)


    inds1m3 = np.where(bin_centers < cutm3)[0]
    inds2m3 = inds1m3[np.where(counts[inds1m3] > 0)]
    xvals3 = sorted([-1.0 * x for x in bin_centers[inds2m3]])
    (beta_fit_m3, alpha_fit_m3,
    gamma_fit_m3, x_min_m3, x_max_m3) = get_tail_weight(dist=None, params=None,
                                                        x_values=xvals3,
                                                        y_values=list(counts[inds2m3][::-1]),
                                                        n_sd_plus_left=n_sd_plus_left,
                                                        n_sd_plus_right=n_sd_plus_right)


    inds1m4 = np.where(bin_centers < cutm4)[0]
    inds2m4 = inds1m4[np.where(counts[inds1m4] > 0)]
    xvals4 = sorted([-1.0 * x for x in bin_centers[inds2m4]])
    (beta_fit_m4, alpha_fit_m4,
    gamma_fit_m4, x_min_m4, x_max_m4) = get_tail_weight(dist=None, params=None,
                                                        x_values=xvals4,
                                                        y_values=list(counts[inds2m4][::-1]),
                                                        n_sd_plus_left=n_sd_plus_left,
                                                        n_sd_plus_right=n_sd_plus_right)


    label1 = fr'${{{gamma_fit_p2:.3g}}}\,\exp(-{{{beta_fit_p2:.3g}}}\,x^{{{alpha_fit_p2:.3g}}})$'
    plot_ys = [model_function(x, beta_fit_p2, alpha_fit_p2, gamma_fit_p2) for x in bin_centers[inds2p2]]
    plt.plot(bin_centers[inds2p2], plot_ys, ls='-', lw=2, color='blue', label=label1)

    plot_ys = [model_function(x, beta_fit_p3, alpha_fit_p3, gamma_fit_p3) for x in bin_centers[inds2p3]]
    label2 = fr'${{{gamma_fit_p3:.3g}}}\,\exp(-{{{beta_fit_p3:.3g}}}\,x^{{{alpha_fit_p3:.3g}}})$'
    plt.plot(bin_centers[inds2p3], plot_ys, ls='--', lw=2, color='blue', label=label2)

    plot_ys = [model_function(x, beta_fit_p4, alpha_fit_p4, gamma_fit_p4) for x in bin_centers[inds2p4]]
    label3 = fr'${{{gamma_fit_p4:.3g}}}\,\exp(-{{{beta_fit_p4:.3g}}}\,x^{{{alpha_fit_p4:.3g}}})$'
    plt.plot(bin_centers[inds2p4], plot_ys, ls=':', lw=2, color='blue', label=label3)


    plot_xs = np.linspace(min(all_acts), cutm2, 1000)
    plot_ys = [model_function(-x, beta_fit_m2, alpha_fit_m2, gamma_fit_m2) for x in plot_xs]
    label4 = fr'${{{gamma_fit_m2:.3g}}}\,\exp(-{{{beta_fit_m2:.3g}}}\,x^{{{alpha_fit_m2:.3g}}})$'
    plt.plot(plot_xs, plot_ys, ls='-.', lw=2, color='blue', label=label4)

    plot_xs = np.linspace(min(all_acts), cutm3, 1000)
    plot_ys = [model_function(-x, beta_fit_m3, alpha_fit_m3, gamma_fit_m3) for x in plot_xs]
    label5 = fr'${{{gamma_fit_m3:.3g}}}\,\exp(-{{{beta_fit_m3:.3g}}}\,x^{{{alpha_fit_m3:.3g}}})$'
    plt.plot(plot_xs, plot_ys, ls=(0, (10, 1, 10, 1, 1, 1, 1, 1)), lw=2, color='blue', label=label5)

    plot_xs = np.linspace(min(all_acts), cutm4, 1000)
    plot_ys = [model_function(-x, beta_fit_m4, alpha_fit_m4, gamma_fit_m4) for x in plot_xs]
    label6 = fr'${{{gamma_fit_m4:.3g}}}\,\exp(-{{{beta_fit_m4:.3g}}}\,x^{{{alpha_fit_m4:.3g}}})$'
    plt.plot(plot_xs, plot_ys, ls=(0, (3, 1, 1, 1, 1, 1)), lw=2, color='blue', label=label6)


    plt.axvline(x=cutp2, color='grey', linestyle='--', alpha=0.6, label=r'$\mu + 2\sigma$')
    plt.axvline(x=cutp3, color='grey', linestyle=':', alpha=0.6, label=r'$\mu + 3\sigma$')
    plt.axvline(x=cutp4, color='grey', linestyle='-.', alpha=0.6, label=r'$\mu + 4\sigma$')
    plt.axvline(x=cutm2, color='grey', linestyle='--', alpha=0.6, label=r'$\mu - 2\sigma$')
    plt.axvline(x=cutm3, color='grey', linestyle=':', alpha=0.6, label=r'$\mu - 3\sigma$')
    plt.axvline(x=cutm4, color='grey', linestyle='-.', alpha=0.6, label=r'$\mu - 4\sigma$')

    plt.ylim(0,0.0015)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{fig_outdir}/dist_hist_zoom_layer_{layer_num}_{outer_key.replace(" ", "_")}__{inner_key.replace(" ", "_")}.png')
    plt.close()

    return {"alpha_fit_p2" : alpha_fit_p2,
            "beta_fit_p2"  : beta_fit_p2,
            "gamma_fit_p2" : gamma_fit_p2,
            "alpha_fit_p3" : alpha_fit_p3,
            "beta_fit_p3"  : beta_fit_p3,
            "gamma_fit_p3" : gamma_fit_p3,
            "alpha_fit_p4" : alpha_fit_p4,
            "beta_fit_p4"  : beta_fit_p4,
            "gamma_fit_p4" : gamma_fit_p4,
            "alpha_fit_m2" : alpha_fit_m2,
            "beta_fit_m2"  : beta_fit_m2,
            "gamma_fit_m2" : gamma_fit_m2,
            "alpha_fit_m3" : alpha_fit_m3,
            "beta_fit_m3"  : beta_fit_m3,
            "gamma_fit_m3" : gamma_fit_m3,
            "alpha_fit_m4" : alpha_fit_m4,
            "beta_fit_m4"  : beta_fit_m4,
            "gamma_fit_m4" : gamma_fit_m4,
            "mu_hat"       : aa_mu,
            "std_hat"      : aa_std}


def plot_distribution_with_modes(activations: np.ndarray,
                                 best_dist: dict,
                                 mode_analysis: dict,
                                 outer_key: str,
                                 inner_key: str,
                                 hook_name: str,
                                 layer_num: int,
                                 fig_outdir: str,
                                 n_bins: int,
                                 mean: float = None,
                                 std: float = None,
                                 x_min: float = None,
                                 x_max: float = None):
    """
    Create two plots:
    1. Overall distribution with best fit
    2. Zoomed view with KDE and mode detection
    """
    # Use provided stats or calculate if not provided
    if mean is None:
        mean = np.mean(activations)
    if std is None:
        std = np.std(activations)
    if x_min is None:
        x_min = mean - 20 * std
    if x_max is None:
        x_max = mean + 20 * std
    
    x_kde = mode_analysis['x_kde']
    density = mode_analysis['density']

    ## Create standard deviation lines positions
    #std_lines = {
    #    f'μ {sign}{i}σ': mean + (sign_val * i * std)
    #    for i in range(2, 5)
    #    for sign, sign_val in [('+', 1), ('-', -1)]
    #}
    
    # First plot: Overall distribution
    plt.figure(figsize=(12, 8))
    counts, bins, _ = plt.hist(activations, bins=n_bins, density=True, 
                              label='Post Residuals Activations', color="red", alpha=0.6)
    
    # Plot fitted distribution
    x = np.linspace(x_min, x_max, 1000)
    y = best_dist['dist'].pdf(x, *best_dist['params'])
    
    # Create label for the fitted distribution
    shape_str = '' if len(best_dist['params']) < 3 else ('\nShape = ' +
                                                         ', '.join([f'{p:.3g}' for p in best_dist['params'][:-2]]))
    loc_str = f'{best_dist["params"][-2]:.3g}'
    scale_str = f'{best_dist["params"][-1]:.3g}'
    hist_label = (f'Fitted {best_dist["dist"].name.title()}\n' +
                 f'Loc = {loc_str}\nScale = {scale_str}{shape_str}')
    
    plt.plot(x, y, lw=2, ls='-', color='green', label=hist_label)
    plt.title(f'Distribution for {outer_key} : {inner_key}\nLayer {layer_num} ({hook_name})')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.xlim(mean - 6 * std, mean + 6 * std)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{fig_outdir}/dist_hist_layer_{layer_num}_' + \
                f'{outer_key.replace(" ", "_")}__{inner_key.replace(" ", "_")}.png')
    plt.close()
    
    # Second plot: Zoomed view with KDE and modes
    plt.figure(figsize=(12, 8))
    
    # Plot the histogram but don't recalculate
    plt.stairs(counts, bins, color='red', alpha=0.6, fill=True, label="Post Residual Activations")

    # Plot pre-computed KDE
    plt.plot(x_kde, density, 'k-', lw=2, label='Kernel Density Estimate')
    
    # Plot main mode
    main_mode = mode_analysis['main_mode']
    plt.axvline(x=main_mode['location'], color='purple', linestyle='-', linewidth=2,
                label=f'Main Mode (h={main_mode["density"]:.2e})')
    
    # Plot left modes with sequential colors
    colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(mode_analysis['left_modes'])))
    for i, mode in enumerate(mode_analysis['left_modes']):
        plt.axvline(x=mode['location'], color=colors[i], linestyle='--', linewidth=2,
                   label=f'Left Mode {i+1} (h={mode["relative_height"]:.2e})')
    
    # Plot right modes with sequential colors
    colors = plt.cm.Reds(np.linspace(0.5, 0.9, len(mode_analysis['right_modes'])))
    for i, mode in enumerate(mode_analysis['right_modes']):
        plt.axvline(x=mode['location'], color=colors[i], linestyle='--', linewidth=2,
                   label=f'Right Mode {i+1} (h={mode["relative_height"]:.2e})')
    
    ## Add standard deviation lines
    #for label, x_val in std_lines.items():
    #    plt.axvline(x=x_val, color='grey', linestyle=':', alpha=0.6, label=label)
    
    # Set y-limit for better visibility of small modes
    # Find max density of minor modes
    minor_mode_densities = []
    if mode_analysis['left_modes']:
        minor_mode_densities.extend(mode['relative_height'] * main_mode['density'] 
                                for mode in mode_analysis['left_modes'])
    if mode_analysis['right_modes']:
        minor_mode_densities.extend(mode['relative_height'] * main_mode['density'] 
                                for mode in mode_analysis['right_modes'])

    if minor_mode_densities:
        max_minor_density = max(minor_mode_densities)
        plt.ylim(0, max_minor_density * 1.4)  # Set to 1.4 times the highest minor mode
    else:
        # Fallback if no minor modes found
        max_density = np.max(density)
        plt.ylim(0, max_density * 0.005)
    
    plt.title(f'Mode Analysis for {outer_key} : {inner_key}\nLayer {layer_num} ({hook_name})')
    plt.xlabel('Activation Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{fig_outdir}/dist_hist_zoom_layer_{layer_num}_' + 
                  f'{outer_key.replace(" ", "_")}__{inner_key.replace(" ", "_")}.png',
                bbox_inches='tight')
    plt.close()
    
    return {
        "mu_hat": mean,
        "std_hat": std,
        "main_mode_location": main_mode['location'],
        "main_mode_height": main_mode['density'],
        "main_mode_spread": main_mode['spread']
    }



def analyze_secondary_modes(activations: np.ndarray, 
                            n_bins_kde: int = 1000,
                            kernel_width_scalar: float = 4.5,
                            prominence_threshold: float = 0.0001,
                            distance_threshold: float = 0.5,
                            mean: float = None,
                            std: float = None,
                            x_min: float = None,
                            x_max: float = None) -> dict:
    """
    Detect and characterize secondary modes in activation distribution.
    
    Args:
        activations: 1D array of activation values
        n_bins_kde: Number of bins for density estimation
        prominence_threshold: Minimum relative height of peaks compared to main peak
        distance_threshold: Minimum distance between peaks in standard deviations
    
    Returns:
        Dictionary containing:
        - Main mode characteristics
        - List of left modes with their characteristics
        - List of right modes with their characteristics
    """
    # Calculate basic statistics
    if mean is None:
        mean = np.mean(activations)
    if std is None:
        std = np.std(activations)
    
    # Get kernel density estimate for smoother peak detection
    # Use Scott's rule for bandwidth selection
    kde = st.gaussian_kde(activations, bw_method='scott')
    kde.set_bandwidth(kde.factor * kernel_width_scalar)
    
    # Generate points for density evaluation
    # Use wider range to catch outlier modes
    if x_min is None:
        x_min = np.min(activations)
    if x_max is None:
        x_max = np.max(activations)
    x = np.linspace(x_min, x_max, n_bins_kde)
    density = kde(x)
    
    # Find peaks in the density
    peaks, peak_properties = signal.find_peaks(
        density,
        prominence=(prominence_threshold * np.max(density), None),  # Minimum prominence
        distance=int(distance_threshold * n_bins_kde / 16)  # Minimum distance between peaks
    )
    
    if len(peaks) == 0:
        return {"error": "No peaks detected"}
    
    # Find the main mode (highest peak)
    main_peak_idx = peaks[np.argmax(density[peaks])]
    main_peak_x = x[main_peak_idx]
    main_peak_density = density[main_peak_idx]
    
    # Find width of main peak at half height
    main_peak_width = signal.peak_widths(
        density, [main_peak_idx], rel_height=0.5
    )
    main_peak_spread = main_peak_width[0][0]
    
    # Separate left and right secondary modes
    left_modes = []
    right_modes = []
    
    for idx in peaks:
        if idx == main_peak_idx:
            continue
            
        peak_x = x[idx]
        peak_density = density[idx]
        
        # Get width at half height for this peak
        peak_width = signal.peak_widths(
            density, [idx], rel_height=0.5
        )
        peak_spread = peak_width[0][0]
        
        # Calculate characteristics
        mode_info = {
            'location': peak_x,
            'density': peak_density,
            'spread': peak_spread,
            'relative_height': peak_density / main_peak_density,
            'distance_from_main': (peak_x - main_peak_x) / std,
            'distance_from_mean': (peak_x - mean) / std
        }
        
        if peak_x < main_peak_x:
            left_modes.append(mode_info)
        else:
            right_modes.append(mode_info)
    
    # Sort modes by distance from main peak
    left_modes.sort(key=lambda x: abs(x['distance_from_main']))
    right_modes.sort(key=lambda x: abs(x['distance_from_main']))
    
    return {
        'main_mode': {
            'location': main_peak_x,
            'density': main_peak_density,
            'spread': main_peak_spread,
            'distance_from_mean': (main_peak_x - mean) / std
        },
        'left_modes': left_modes,
        'right_modes': right_modes,
        'stats': {
            'mean': mean,
            'std': std,
            'n_left_modes': len(left_modes),
            'n_right_modes': len(right_modes)
        },
        "density": density,
        "x_kde": x
    }


def get_gpu_memory_info():
    """
    Returns detailed information about GPU memory usage.
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = t - (a + r)  # free inside reserved
    
    return {
        'total': t / 1024**3,  # Convert to GB
        'reserved': r / 1024**3,
        'allocated': a / 1024**3,
        'free': f / 1024**3
    }


def get_gpu_memory_str():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = t - (a + r)  # free inside reserved
    
    ret_str = f"Total:     {t / 1024**3:.2f}GB\n" + \
              f"Allocated: {a / 1024**3:.2f}GB\n" + \
              f"Reserved:  {r / 1024**3:.2f}GB\n" + \
              f"Free:      {r / 1024**3:.2f}GB\n" + \
              f"********************\n"
    return ret_str


def do_simulations_all(model: HookedTransformer,
                       sentences: dict,
                       hook_names: list[str],
                       dnames: list,
                       fig_outdir: str,
                       n_bins: int,
                       n_bins_kde: int = 1000,
                       kernel_width_scalar: float = 4.5,
                       n_sd_plus_left: float = 2.0,
                       n_sd_plus_right: float = 10.0,
                       batch_size: int = 20,
                       device: str = "cuda"):
    """
    GPU-optimized version focusing on the most impactful optimizations
    """
    dist_res = []
    
    # Set up logging so can print updates without ruining progress bars
    logger = setup_tqdm_logging()
    in_notebook = is_notebook()
    # In notebooks, we might want to adjust these parameters
    outer_tqdm_params = {
        "desc": "Processing files",
        "position": 0 if not in_notebook else None,
        "leave": True,
        "desc" : "Processing pairs", 
        "bar_format" : "Pair {n_fmt}/{total_fmt} [{rate_fmt}] {desc}"
    }
    
    inner_tqdm_params = {
        "position": 1 if not in_notebook else None,
        "leave": False if not in_notebook else True,
        "desc" : "Processing hooks", 
        "bar_format" : "Hook {n_fmt}/{total_fmt} [{rate_fmt}] {desc}"
    }

    # Validate inner keys consistency
    outer_keys = list(sentences.keys())
    reference_inner_keys = set(sentences[outer_keys[0]].keys())
    for outer_key in outer_keys[1:]:
        if set(sentences[outer_key].keys()) != reference_inner_keys:
            raise ValueError(f"Inner keys don't match across outer keys. {outer_key} has different inner keys")
    inner_keys = list(reference_inner_keys)
    
    # Pre-compile regex pattern
    layer_pattern = re.compile(r'blocks\.(\d+)\.')
    names_filter = hook_names
    
    # Pre-compute threshold multipliers on GPU once
    std_multiples = torch.arange(2, 11, device=device)
    
    # At the start of the function, calculate total combinations
    total_pairs = len(outer_keys) * len(inner_keys)
    total_hooks = len(hook_names)

    # Create the nested progress bars
    with tqdm(total=total_pairs, **outer_tqdm_params) as pbar_outer:
        for outer_key in outer_keys:
            for inner_key in inner_keys:
                all_sentences = sentences[outer_key][inner_key]
                n_samples = len(all_sentences)
                
                # Initialize dictionary for all hooks
                hook_activations = {hook_name: [] for hook_name in hook_names}
                
                # Process sentences in batches
                for i in range(0, n_samples, batch_size):
                    batch_sentences = all_sentences[i:i + batch_size]
                    seq_lengths = [len(model.to_str_tokens(s)) for s in batch_sentences]
                    
                    with torch.inference_mode():
                        batch_logits, batch_cache = model.run_with_cache(
                            batch_sentences,
                            names_filter=names_filter
                        )
                        
                        # Process all hooks for this batch
                        for hook_name in hook_names:  # Move hook loop here
                            batch_activations = torch.stack([
                                batch_cache[hook_name][b, length-1, :]
                                for b, length in enumerate(seq_lengths)
                            ])
                            hook_activations[hook_name].append(batch_activations)
                    
                    del batch_cache
                    
                # Process each hook's activations
                with tqdm(total=total_hooks, **inner_tqdm_params) as pbar_inner:
                    for hook_name in hook_names:
                        combined_activations_gpu = torch.cat(hook_activations[hook_name], dim=0).reshape(-1)
                        
                        # Compute statistics on GPU
                        mean = combined_activations_gpu.mean().cpu().item()
                        std = combined_activations_gpu.std().cpu().item()
                        
                        # Compute all thresholds at once
                        upper_thresholds = mean + std_multiples * std
                        lower_thresholds = mean - std_multiples * std
                        # Calculate total number of values for proportion calculation
                        total_values = combined_activations_gpu.numel()
                        
                        # Compute proportions efficiently 
                        above_pcts = [(combined_activations_gpu.unsqueeze(1) >
                                       upper_thresholds).sum(0).float() / total_values]
                        below_pcts = [(combined_activations_gpu.unsqueeze(1) <
                                       lower_thresholds).sum(0).float() / total_values]
                        
                        # Move to CPU for distribution fitting
                        combined_activations = combined_activations_gpu.cpu().numpy()
                        
                        layer_num = int(layer_pattern.search(hook_name).group(1))
                        
                        # Distribution fitting and tail weight calculation
                        bfd = BestFitDistribution(combined_activations, n_bins=n_bins, distributionNames=dnames)
                        all_fits = bfd.best_fit_distribution()
                        best_dist = all_fits[0]
                        
                        beta_fit, alpha_fit, gamma_fit, x_min, x_max = get_tail_weight(
                            best_dist['dist'], best_dist['params'],
                            x_values=None, y_values=None,
                            n_sd_plus_left=n_sd_plus_left,
                            n_sd_plus_right=n_sd_plus_right
                        )
                        
                        mode_analysis = analyze_secondary_modes(combined_activations,
                                                                n_bins_kde=n_bins_kde,
                                                                kernel_width_scalar=kernel_width_scalar,
                                                                prominence_threshold=0.0001,
                                                                distance_threshold=0.5,
                                                                mean=mean, std=std
                                                                
                        )
                        plot_distribution_with_modes(combined_activations, best_dist, mode_analysis, outer_key,
                                                    inner_key, hook_name, layer_num, fig_outdir, n_bins,
                                                    mean, std, bfd.data_min, bfd.data_max)
                        
                        # Store results
                        result_dict = {
                            'dist_name': best_dist['dist_name'],
                            'params': best_dist['params'],
                            'sse': best_dist['sse'],
                            'subject': outer_key,
                            'attribute': inner_key,
                            'n_sentences': n_samples,
                            'hook': hook_name,
                            'layer_num': layer_num,
                            'alpha_hat': alpha_fit,
                            'beta_hat': beta_fit,
                            'gamma_hat': gamma_fit,
                            'mu_hat': mean,
                            'std_hat': std,
                            'x_kde': mode_analysis['x_kde'],
                            'density': mode_analysis['density'],
                            'main_mode_location': mode_analysis['main_mode']['location'],
                            'main_mode_spread': mode_analysis['main_mode']['spread'],
                            'main_mode_density': mode_analysis['main_mode']['density'],
                            'n_left_modes': mode_analysis['stats']['n_left_modes'],
                            'n_right_modes': mode_analysis['stats']['n_right_modes']
                        }
                        
                        # Add counts
                        above_pcts_list = above_pcts[0].tolist()
                        below_pcts_list = below_pcts[0].tolist()
                        
                        for i, (above, below) in enumerate(zip(above_pcts_list, below_pcts_list), 2):
                            result_dict.update({
                                f"pct_gt_p{i}std": above,
                                f"pct_gt_m{i}std": below
                            })
                        
                        # Add secondary mode details
                        for i, mode in enumerate(mode_analysis['left_modes']):
                            prefix = f'left_mode_{i+1}'
                            result_dict.update({
                                f'{prefix}_location': mode['location'],
                                f'{prefix}_height': mode['relative_height'],
                                f'{prefix}_spread': mode['spread'],
                                f'{prefix}_distance': mode['distance_from_main']
                            })
                        
                        for i, mode in enumerate(mode_analysis['right_modes']):
                            prefix = f'right_mode_{i+1}'
                            result_dict.update({
                                f'{prefix}_location': mode['location'],
                                f'{prefix}_height': mode['relative_height'],
                                f'{prefix}_spread': mode['spread'],
                                f'{prefix}_distance': mode['distance_from_main']
                            })
                        
                        dist_res.append(result_dict)

                        pbar_inner.update(1)
                        pbar_inner.set_description(f"Processing {hook_name}")                        
                        
                        # Clear GPU memory
                        del combined_activations_gpu, combined_activations
                        gc.collect()
                        torch.cuda.empty_cache()
                
                pbar_outer.update(1)
                pbar_outer.set_description(f"Completed {outer_key}:{inner_key}")
                
                del hook_activations
                gc.collect()
                torch.cuda.empty_cache()
                
                logger.info(get_gpu_memory_str())
    # Create columns list
    base_columns = ['dist_name', 'params', 'sse', 'subject', 'attribute', 'n_sentences',
                   'hook', 'layer_num', 'alpha_hat', 'beta_hat', 'gamma_hat']
    
    fit_columns = [f"{p}_fit_{d}{i}" for p in ['alpha', 'beta', 'gamma'] 
                  for d in ['p', 'm'] for i in range(2, 5)]
    
    stat_columns = ['mu_hat', 'std_hat'] + \
                  [f"pct_gt_{d}{i}std" for d in ['p', 'm'] for i in range(2, 11)]
    
    mode_columns = ['main_mode_location', 'main_mode_spread', 'main_mode_density',
                   'n_left_modes', 'n_right_modes']
    
    # Add dynamic columns for modes (up to max observed)
    max_left_modes = max(res['n_left_modes'] for res in dist_res)
    max_right_modes = max(res['n_right_modes'] for res in dist_res)
    
    for i in range(max_left_modes):
        prefix = f'left_mode_{i+1}'
        mode_columns.extend([f'{prefix}_location', f'{prefix}_height', 
                           f'{prefix}_spread', f'{prefix}_distance'])
    
    for i in range(max_right_modes):
        prefix = f'right_mode_{i+1}'
        mode_columns.extend([f'{prefix}_location', f'{prefix}_height', 
                           f'{prefix}_spread', f'{prefix}_distance'])

    # Data columns
    data_columns = ['x_kde', 'density']
    columns = base_columns + fit_columns + stat_columns + mode_columns + data_columns
    
    return pd.DataFrame(dist_res, columns=columns)


# Examine distributions to get > 90% of top ranked
def get_top_dists(by_dist : pd.DataFrame,
                  min_proportion : float = 0.9):
    bd_srt = by_dist.sort_values(['n_top_rank', 'n_top_five', 'dist_name'], ascending=[False,False,True])
    n_ranked = by_dist['n_top_rank'].sum()
    rolling_sum = 0
    ii = 0
    while (rolling_sum / n_ranked) < min_proportion and ii < bd_srt.shape[0]:
        rolling_sum += bd_srt.iloc[ii]['n_top_rank']
        ii += 1
    return bd_srt.iloc[:ii]


def filter_top_contributing_rows(group):
    sorted_group = group.sort_values('n_top_rank', ascending=False)
    total_rank = sorted_group['n_top_rank'].sum()
    target_rank = 0.8 * total_rank
    cumsum = sorted_group['n_top_rank'].cumsum()
    # argmax is like first for boolean lists. Returns first instance of max. False == 0, True == 1
    i = np.argmax(cumsum >= target_rank)
    return sorted_group[:(i + 1)]



# Approximate empirical tail weight by fitting gamma * exp(-beta * x^alpha) to the right end of either a fitted
# function or a dataset.
def get_tail_weight(dist      : st._continuous_distns.rv_continuous,
                    params    : Tuple[float],  # Params list, shape params first, then loc, then scale
                    x_values  : List[float],  # Can send in dist function or x and y directly
                    y_values  : List[float],
                    x_min_max : Tuple[float, float] = None,
                    fig_fname : str = None,
                    n_sd_plus_left : float = 2.0,
                    n_sd_plus_right : float = 5.0):
    try:
        # Generate sample data from the distribution
        if dist is not None and isinstance(dist, st._continuous_distns.rv_continuous):
            #print("Doing fxn fit")
            if x_min_max is None:
                mean = dist.mean(*params)
                std = dist.std(*params)
                if np.isnan(mean) or np.isnan(std) or not np.isfinite(mean) or not np.isfinite(std):
                    # Fall back to reasonable defaults if mean/std are undefined
                    x_min_max = (1.5, 6)
                else:
                    x_min_max = (mean + n_sd_plus_left * std, mean + n_sd_plus_right * std)
        
            x_values = np.linspace(x_min_max[0], x_min_max[1], 1000)
            y_values = dist.pdf(x_values, *params)
            # Filter out any infinite or NaN values
            mask = np.isfinite(y_values)
            if not np.any(mask):
                return (np.nan, np.nan, np.nan, x_min_max[0], x_min_max[1])  # Return NaN if no valid points
            
            x_values = x_values[mask]
            y_values = y_values[mask]
            
            # Only proceed with curve fitting if we have enough valid points
            if len(x_values) < 10:  # arbitrary threshold
                return (np.nan, np.nan, np.nan, x_min_max[0], x_min_max[1])
            
        elif ((isinstance(x_values, list) and all(isinstance(x, (int, float)) for x in x_values) and
               isinstance(y_values, list) and all(isinstance(y, (int, float)) for y in y_values)) or
              (isinstance(x_values, np.ndarray) and np.issubdtype(x_values.dtype, np.floating) and
               isinstance(y_values, np.ndarray) and np.issubdtype(y_values.dtype, np.floating)) and
              len(x_values) == len(y_values)):
            #print("Doing data fit")
            x_min_max = (min(x_values), max(x_values))
            mean = np.mean(x_values)
            std = np.std(x_values)
        else:
            raise ValueError("Must send either dist:st._continuous_distns.rv_continuous or x_values and y_values of List[float, ...]")
        
            
        # Fit the model function to the filtered data
        initial_guess = [1.0, 1.0, 1.0]  # Initial guesses for beta, alpha, gamma
        try:
            #print("Trying fit")
            params_hat, covar_hat = curve_fit(model_function, x_values, y_values, 
                                              p0=initial_guess, maxfev=2000,
                                              bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]))
        except:
            return (np.nan, np.nan, np.nan, x_min_max[0], x_min_max[1])
        
        # Extract fitted parameters
        beta_fit, alpha_fit, gamma_fit = params_hat
        
        if fig_fname is not None:
            try:
                # Plot the results
                plt.figure(figsize=(10, 6))
                step_inds = range(0, len(x_values), max(1, len(x_values) // 25))
                markerline, stemlines, baseline = plt.stem(x_values[step_inds], y_values[step_inds], 
                                                         label=dist.name.capitalize() + ' PDF',
                                                         basefmt=' ', linefmt='r-', markerfmt='ro')
                plt.setp(stemlines, 'alpha', 0.6)
                plt.setp(markerline, 'alpha', 0.6)
                
                plt.plot(x_values, model_function(x_values, *params_hat), 
                        label=f'Fitted ${gamma_fit:.3g} \\exp(-{beta_fit:.3g} x^{{{alpha_fit:.4g}}})$', 
                        color='blue', ls='--')
                plt.xlabel('X')
                plt.ylabel('Density')
                plt.title('Fitting $f(x) = \\gamma e^{- \\beta x^{\\alpha}}$ to ' + 
                          dist.name.capitalize() + ' Distribution')
                plt.legend()
                plt.grid()
                plt.xlim(0.95 * x_values[0], 1.05 * x_values[-1])
                plt.savefig(fig_fname, format='png')
                plt.close()
            except:
                plt.close()  # Ensure figure is closed even if plotting fails
        
        return (beta_fit, alpha_fit, gamma_fit, x_min_max[0], x_min_max[1])
    
    except Exception as e:
        print(e)
        return (np.nan, np.nan, np.nan, x_min_max[0], x_min_max[1])

# For debugging and monitoring memory, not used in this code.
def print_tensor_sizes():
    """
    Prints all tensors in memory sorted by size.
    """
    for obj in gc.get_objects():
        try:
            # Check if object is a tensor
            if torch.is_tensor(obj):
                tensor = obj
            # Check if object has .data attribute (like Parameters)
            elif hasattr(obj, 'data') and isinstance(obj.data, torch.Tensor):
                tensor = obj.data
            else:
                continue
                
            if tensor.is_cuda:  # Only show GPU tensors
                size_mb = tensor.element_size() * tensor.nelement() / 1024**2
                print(f"Type: {type(obj).__name__:20} "
                      f"Size: {list(tensor.size()):20} "
                      f"Dtype: {tensor.dtype:10} "
                      f"Device: {tensor.device:12} "
                      f"Memory: {size_mb:.2f} MB")
        except Exception as e:
            continue


def is_notebook():
    """Check if we're running in a Jupyter notebook"""
    try:
        shell = get_ipython().__class__.__name__
        return shell == 'ZMQInteractiveShell'
    except:
        return False

def setup_tqdm_logging():
    """Configure logging to work with tqdm progress bars in any environment"""
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.in_notebook = is_notebook()

        def emit(self, record):
            try:
                msg = self.format(record)
                # In notebooks, we need to ensure proper line breaks
                if self.in_notebook:
                    msg = f"{msg}\n"
                tqdm.write(msg, file=sys.stdout)
                self.flush()
            except Exception:
                self.handleError(record)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[TqdmLoggingHandler()]
    )
    return logging.getLogger()



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# In[18]:


sentences_dir = './data'
sf = open(sentences_dir + "/sentences_20241018_00.json", 'r')
sentences = json.load(sf)
# remove the last word so it's more in context?
sentences = {k1 : {k2: [snt.rsplit(' ', 1)[0] for snt in sentences[k1][k2]] for k2 in sentences[k1].keys()} for k1 in sentences.keys()}
subjects = list(sentences.keys())
attributes = list(set().union(*[set(d.keys()) for d in sentences.values()]))

# Significantly cut down from the distributions in scipy.stats for ones that included negative values and ever even
# vaguely competed to be the best fitting in the data analyzed. For other use cases could expand back out and refilter.
dnames = ['crystalball', 'genlogistic', 'gennorm', 'hypsecant', 'johnsonsb', 'johnsonsu', 'laplace',
          'laplace_asymmetric', 'loggamma', 'logistic', 'norm', 'norminvgauss', 'pearson3', 'powernorm',
          'skewnorm', 't', 'truncnorm', 'vonmises', 'vonmises_line']

load_all_sim_from_file = True
output_base_dir = '.'
fig_outdir = output_base_dir + '/figs'
data_outdir = output_base_dir + '/raw_output' 
n_bins = 150
n_bins_kde = 1000
kernel_width_scalar=4.5

MODEL_NAMES = [
    "pythia-410m",
    "pythia-160m",
    "gpt2-small",
    "gpt2-medium",
    "pythia-1b",
    "gpt2-large",
]
batch_size = 20
torch.set_grad_enabled(False)
DEVICE='cuda'
for model_name in MODEL_NAMES:
    print(f"Processing {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
    model_n_layers = model.cfg.n_layers
    hook_names = [f'blocks.{i}.hook_resid_post' for i in range(model_n_layers)]
    
    # Modify your output paths to include model name
    model_fig_outdir = f"{fig_outdir}/{model_name}"
    model_data_outdir  = f"{data_outdir}/{model_name}"
    os.makedirs(model_fig_outdir, exist_ok=True)
    os.makedirs(model_data_outdir, exist_ok=True)
    
    results_df = do_simulations_all(
        model=model,
        sentences=sentences,
        hook_names=hook_names,
        dnames=dnames,
        fig_outdir=model_fig_outdir,
        n_bins=n_bins,
        n_bins_kde=n_bins_kde,
        kernel_width_scalar=kernel_width_scalar,
        n_sd_plus_left=2,
        n_sd_plus_right=10,
        batch_size=batch_size,
        device=DEVICE
    )
    
    # Save results with model name
    results_df.to_csv(f"{model_data_outdir}/results_{model_name}.csv")
    
    # Clear GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

