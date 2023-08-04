import pandas as pd
from datetime import datetime
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

#############################################################################
###                                Options                                ###
#############################################################################

# Select which file to analyze
# data_file = 'dataset_experimental/data_1point.csv'        # Only has a single LH-reading per position in the grid.
data_file = 'dataset_experimental/data_all.csv'             # It has all data captured in the experiment

#############################################################################
###                                Functions                              ###
#############################################################################

# All functions used in the main, are defined in functions.py
from functions import *

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    df = pd.read_csv(data_file, index_col=0)

    # Transform LH counts to sweep angles, and then
    # Project sweep angles on to the z=1 image plane
    pts_lighthouse_A = LH2_count_to_pixels(df['LHA_count_1'].values, df['LHA_count_2'].values, 0)
    pts_lighthouse_B = LH2_count_to_pixels(df['LHB_count_1'].values, df['LHB_count_2'].values, 1)    

    # Add the LH2 projected points into the dataframe that holds the info about which-point-is-where in real life.
    df['LHA_proj_x'] = pts_lighthouse_A[:,0]
    df['LHA_proj_y'] = pts_lighthouse_A[:,1]
    df['LHB_proj_x'] = pts_lighthouse_B[:,0]
    df['LHB_proj_y'] = pts_lighthouse_B[:,1]

    # Solve the 3D scene with recoverPose and Triangulate points
    point3D, t_star, R_star = solve_3d_scene(pts_lighthouse_A, pts_lighthouse_B)

    # Add The 3D point to the Dataframe that has the real coordinates, timestamps etc.
    # This will help correlate which point are supposed to go where.
    df['LH_x'] = point3D[:,0]
    df['LH_y'] = point3D[:,2]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
    df['LH_z'] = point3D[:,1]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height

    # Scale the scene to real size, using the 40mm distance between each grid position as a reference
    if 'experimental' in data_file:
        df = scale_scene_to_real_size(df)

    # Compute distances between gridpoints, to later plot the error histograms.
    x_dist, y_dist, z_dist = compute_distance_between_grid_points(df)

    # Bring scaled & triangulated 3D points to the origin (0,0,0), for easier comparison with the ground truth. 
    df = correct_perspective(df)


    #############################################################################
    ###                             Plotting                                  ###
    #############################################################################
    # Plot projected views of the lighthouse
    plot_projected_LH_views(pts_lighthouse_A, pts_lighthouse_B)

    # Plot superimposed "captured data" vs. "ground truth",
    plot_transformed_3D_data(df)
    # Plot the error histogram from that comparison
    plot_error_histogram(df)























    #############################################################################
    ###                      How to play with the data                        ###
    #############################################################################

    # All the data in the script is stored in the df dataframe.
    print(df)

    # If you want a numpy matrix with the projected image point from each LH:
    pts_lighthouse_A = df[['LHA_proj_x', 'LHA_proj_y']].values
    pts_lighthouse_B = df[['LHB_proj_x', 'LHB_proj_y']].values

    # If you want the real "ground truth" position of where each point should be in the grid, (in mm)
    # The frame of reference is the one explained in the rainbow diagram here: 
    # https://crystalfree.atlassian.net/wiki/spaces/~105637413/pages/2313420801/LH2+-+3D+accuracy+experiment+-+with+3D+printer+-+v2#Sensor-Movement-Pattern
    ground_truth_mm = df[['real_x_mm', 'real_y_mm', 'real_z_mm']].values

    # These two numpy arrays are index-synchronized.
    # These two indices correspond to the same data point
    print(ground_truth_mm[18])
    print(pts_lighthouse_A[18])

    # The other available columns are:
    # [LH_x, LH_y, LH_z] - are the 3d points, triangulated from the LH views, and scaled so that each grid point is 40mm apart. It's in milimeters.
    # [Rt_x, Rt_y, Rt_z] - Same as above, but the points have been rotated and translated so that the [0,0,0] point of the grid, rests at the origin, and the grid is aligned to the coordinate axes.

    # If you want to get all the triangulated 3D points corresponding to the grid position (40, 120, 80) mm
    specific_point = df.loc[(df['real_x_mm'] == 40)  & (df['real_y_mm'] == 120) & (df['real_z_mm'] == 80), ['LH_x', 'LH_y', 'LH_z']].values