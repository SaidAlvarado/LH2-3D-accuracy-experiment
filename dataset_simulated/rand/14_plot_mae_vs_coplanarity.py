# This code grabs 8 or more (defined in code) random points from the grid position of the dataset and attempts to solve the scene with them.
# it does this several time to get statistics of how efective is to add more points to the scene reconstruction.
#
#

import pandas as pd
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

from skspatial.objects import Plane, Points

#############################################################################
###                                Options                                ###
#############################################################################

data_file = 'dataset_simulated/rand/mae_v_npoints.csv'


def plot_acc_vs_mad(df_plot):
    
    # Find out how many unique N_Points are available in this experiment
    MAD = df_plot['MAD'].to_numpy()
    MAE = df_plot['MAE'].to_numpy()

    # prepare the plot
    fig = plt.figure(layout="constrained")
    gs = GridSpec(3, 3, figure = fig)
    error_ax    = fig.add_subplot(gs[0:3, 0:3])
    axs = (error_ax,)

    # # Plot Y = MAE, X = MAD
    # for i in unique_MAD:
    #     mae = df_plot.loc[(df_plot['MAD'] == i), 'MAE'].values
    #     error_ax.scatter([i]*mae.shape[0], mae, color='xkcd:blue', alpha=0.3)
    #     error_ax.scatter([i], mae.min(), color='xkcd:red', alpha=1)
    # error_ax.scatter([i]*mae.shape[0], mae, color='xkcd:blue', alpha=0.3, label='Reconstruction Error')
    # error_ax.scatter([i], mae.min(), color='xkcd:red', alpha=1, label='Best Reconstruction')
    error_ax.scatter(MAD, MAE, color='xkcd:blue', alpha=0.3, label='Reconstruction Error')
    # error_ax.scatter(MAD, MAE, color='xkcd:red', alpha=1, label='Best Reconstruction')


    # Plot Y = MAE, X = N_points
    # error_ax.plot(mae_std[:,0], mae_std[:,1], 'xkcd:blue')
    # Add and area with 2 std deviation
    # error_ax.fill_between(mae_std[:,0], mae_std[:,1] - mae_std[:,2], mae_std[:,1] + mae_std[:,2], alpha=0.2, edgecolor='xkcd:indigo', facecolor='lightblue', linestyle='dashed', antialiased=True)

    for ax in axs:
        ax.grid()
        ax.legend()
    
    error_ax.set_xlabel('Median Average Deviation [mm]')
    error_ax.set_ylabel('Medium Average Reconstruction Error [mm]')

    print(df_plot)
    plt.show()

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    df=pd.read_csv(data_file, index_col=0)

    df = df.loc[ (df['Coplanar'] > 30) & (df['MAE'] < 200)]

    plot_acc_vs_mad(df)


    
