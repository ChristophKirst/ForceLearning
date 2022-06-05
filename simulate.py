#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FORCE Learning

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/
"""
__author__  = "Christoph Kirst, University of California, San Francisco"
__license__ = "MIT License"


#%% Train nework

import numpy as np
import matplotlib.pyplot as plt
from force import Network


#%% Target function
#def f(t, freq = 2, freq2 = 3):
#    return 0.2*np.sin(2*np.pi*freq*t/1000) + 0.5*np.sin(2*np.pi*freq2*t/1000)

def f(t, freq = 2):
    return 1.5 * np.sin(2*np.pi*freq*t/1000)

plt.figure(10); plt.clf();
t = np.arange(30000) * 0.1 
plt.plot(t, f(t))


#%% Training 

net = Network(alpha = 10, n=500)
net.run(30000, save=True, train = f)
plt.figure(1); plt.clf();
net.plot(f = f)

#%% Generation

net.run(5000, save=True)
plt.figure(2); plt.clf();
net.plot()


