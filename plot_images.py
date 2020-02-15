#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:30:32 2020

@author: krutikapatil
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_pairwise_joint_pdf():
    data_ = np.load('out.npy') # 10,000 X 300 X 3
    gammas = [0.5, 0.3, 0.1]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    T=150 #time step number ... the simulation was run for a total of 300 time steps
    X_axis=1 # y 
    Y_axis=2 # z
    axes[0, 0].set_title('Linear normalization')
    axes[0, 0].hist2d(data_[:, T,X_axis], data_[:, T,Y_axis], bins=100)
    for ax, gamma in zip(axes.flat[1:], gammas):
        ax.set_title(r'Power law $(\gamma=%1.1f)$' % gamma)
        ax.hist2d(data_[:, T,X_axis], data_[:, T,Y_axis],
                bins=100, norm=mcolors.PowerNorm(gamma))
    fig.text(0.5, 0.04, ' Y', ha='center', va='center')
    fig.text(0.06, 0.5, ' Z', ha='center', va='center', rotation='vertical')
        
    fig.tight_layout()
    fig.text(0.02,0.01,'Y vs Z at T= 7.5 seconds')
    fig.savefig('Y vs Z at T= 7.5 seconds.png', dpi=660)
    plt.show()
    


if __name__ =='__main__':

    plot_pairwise_joint_pdf()    
