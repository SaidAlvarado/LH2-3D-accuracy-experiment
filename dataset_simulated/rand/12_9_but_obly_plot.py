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

data_file = 'dataset_simulated/rand/mae_v_npoints.csv.bak'


def plot_acc_vs_npoints(df_plot):
    
    # Find out how many unique N_Points are available in this experiment
    unique_n_points = np.unique(df_plot['n_points'].to_numpy().astype(int), axis=0)

    # Go through all the available N_point experiments
    mae_std = np.empty((1+unique_n_points.max()-8,3))
    for i in unique_n_points:
        # Get the mean of the MAE, and the STD of the MAE
        mae = df_plot.loc[(df_plot['n_points'] == i), 'MAE'].values.mean(axis=0)    
        std = df_plot.loc[(df_plot['n_points'] == i), 'MAE'].values.std(axis=0)
        # Add it to our empty array for plotting later
        mae_std[int(i)-8] = np.array([i, mae, std])    

    # prepare the plot
    fig = plt.figure(layout="constrained", figsize=(5,4))
    gs = GridSpec(3, 3, figure = fig)
    error_ax    = fig.add_subplot(gs[0:3, 0:3])
    axs = (error_ax,)

    # Plot Y = MAE, X = N_points
    error_ax.plot(mae_std[:,0], mae_std[:,1], 'xkcd:blue')
    # Add and area with 2 std deviation
    error_ax.fill_between(mae_std[:,0], np.clip(mae_std[:,1] - 2*mae_std[:,2], 0.0, 1e10), mae_std[:,1] + 2*mae_std[:,2], alpha=0.2, edgecolor='xkcd:indigo', facecolor='lightblue', linestyle='dashed', antialiased=True)

    for ax in axs:
        ax.grid()
        # ax.legend()
    
    error_ax.set_xlabel('Number of points')
    error_ax.set_ylabel('Mean Average Error [mm]')

    error_ax.set_xlim((8, 100))
    error_ax.set_ylim((0, 50))

    plt.savefig('Result-G-2lh_3d-pufpr.pdf')

    print(mae_std)
    plt.show()


#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    df=pd.read_csv(data_file, index_col=0)

    df = df.loc[ (df['Coplanar'] > 30) & (df['MAE'] < 200)]

    plot_acc_vs_npoints(df)


    
