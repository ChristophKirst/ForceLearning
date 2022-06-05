#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FORCE Learning

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2756108/
"""
__author__ = "Christoph Kirst, University of California, San Francisco"
__license__ = "MIT License"


import numpy as np
import matplotlib.pyplot as plt


class Network(object):
    
    def __init__(self, x = None, Jx = None, Jx0 = None, Jz = None, Wz = None, Wz0 = None, P = None, n = 1000, gx = 1.5, px = 0.1, pz = 1.0, alpha = 1.0 ):
        
        if x is None:
            x = np.random.uniform(0,1, size=n);
        else:
            n = x.shape[0];
        self.n = n;
        
        self.x = x;
        self.r = np.tanh(x);
        
        if Jx is None:
            sigma_x = np.sqrt(1.0 / (px * n));
            Jx = gx * np.random.normal(0, sigma_x, size=(n,n));
        self.Jx = Jx;
        if Jx0 is None:     
            Jx0 = np.random.rand(n,n) > px;
        self.Jx[Jx0] = 0;
        self.Jx0 = Jx0;
           
        if Jz is None:
            Jz = np.random.uniform(-1,1,n);
        self.Jz = Jz;
        
        if Wz is None:
            sigma_z = np.sqrt(1.0 / (pz * n));
            Wz = np.random.normal(0, sigma_z, size = n);
        self.Wz = Wz;
        
        if Wz0 is None:
            Wz0 = np.random.rand(n) > pz;
        self.Wz[Wz0] = 0;
        self.Wz0 = Wz0;
        
        self.z = np.dot(Wz, self.r);
        
        if P is None:
            P = np.identity(n) / alpha;
        self.P = P;
        
        self.save = None;
    
 
    def step(self, t, dt = 1, tau = 10, train = False, update = True):
      x, r, z, Jx, Jz, Wz, P = self.x, self.r, self.z, self.Jx, self.Jz, self.Wz, self.P;
      if train is not False:
          f = train;
                
      # network dynamics
      dxdt = (-x + np.dot(Jx, r) + np.dot(Jz, z)) / tau; 
      x_new = x + dxdt * dt;
      r_new = np.tanh(x_new);
 
      if update:
         self.x = x_new;
         self.r = r_new;   
 
      if train is False:
          if update:
              z_new = np.dot(Wz, r_new);
              self.z = z_new;
          return x_new;
      
      # FORCE learning 
      # e- 
      e_minus = np.dot(Wz, r_new) - f(t) 
        
      # P
      Pr = np.dot(P, r_new);
      P_new = P - np.outer(Pr, Pr.T) / (1 + np.dot(r_new, Pr));
      
      # Wz
      Wz_new = Wz - e_minus * np.dot(P_new, r_new);
      if self.Wz0 is not None:
          Wz_new[self.Wz0] = 0;
      
      # e+
      z_new = np.dot(Wz_new, r_new)
      e_plus = z_new - f(t)
      #print(e_minus, e_plus, np.abs(e_plus/(e_minus + 10e-16)))
    
      if update:
          self.z = z_new;
          self.Wz = Wz_new;
          self.P = P_new;
          self.error = (e_minus, e_plus)

      return x_new, Wz_new, P_new;
  
    
    def run(self, steps, t0 = 0, dt = 1, save = False, train = False, verbose = True):
        if save:
            x_save = np.zeros((steps, self.n));
            t_save = np.zeros(steps);
            Wz_save = np.zeros((steps, self.n));
            z_save = np.zeros(steps);
            error_save = np.zeros((steps,2));
        
        if verbose is True:
            verbose = 100;
            
        t = t0;
        for step in range(steps):
            self.step(t, dt=dt, train=train, update=True)
            t += dt;
            if save:
                t_save[step] = t;
                x_save[step] = self.x;
                Wz_save[step] = self.Wz;
                z_save[step] = self.z
                error_save[step] = self.error;
                
            if verbose and (step % verbose == 0 or step==0 or step == steps-1):
                print('Step %d/%d' % (step, steps));
          
        if save:
            self.save = dict(t = t_save, x = x_save, Wz = Wz_save, z = z_save, error = error_save);
        
    def plot(self, f = None):     
        if self.save is not None:
            t, x, Wz, z, e = self.save['t'], self.save['x'], self.save['Wz'], self.save['z'], self.save['error'];     
            ax = plt.subplot(2,2,1);
            plt.plot(t, x);
            plt.title('x')
            
            plt.subplot(2,2,2, sharex=ax);
            plt.plot(t, Wz);
            plt.title('Wz');
            
            plt.subplot(2,2,3, sharex=ax);
            plt.plot(t, z, label ='z(t)');
            if f is not None:
              plt.plot(t, f(t), label = 'f(t)');
            plt.legend()
            plt.title('z');
            
            plt.subplot(2,2,4, sharex=ax);
            plt.plot(t, e);
            plt.plot(t, np.abs(e[:,1]/e[:,0]))
            plt.title('error')
            
            
    def plot_weights(self):
        plt.imshow(self.Wx, origin='lower')

    def __str__(self):
        return 'Network[%d]' % self.n
    
    def __repr__(self):
        return self.__str__();







