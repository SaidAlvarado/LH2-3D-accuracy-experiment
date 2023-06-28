import pandas as pd
from datetime import datetime
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

#############################################################################
###                                Options                                ###
#############################################################################
## To use all the data (instead of a measurement per grip point), set this to true
ALL_DATA = True
# Base station calibration parameters
fcal = {'phase1-B': -0.005335,
        'phase1-C': -0.004791}


#############################################################################
###                                Code                                   ###
#############################################################################

## Read the Timestamps to get which point correspond to which positions
with open("Experiments/2 - 230504 - full grid - covering LH2/timestamps.json", "r") as file:
    time_data = json.load(file)

# This is the important number
z_lvl = ['z=0cm', 'z=4cm', 'z=8cm', 'z=12cm']
exp_grid = ["(1,1)", "(1,3)", "(1,5)", "(1,7)", "(1,9)", "(1,11)", "(1,13)",
            "(3,1)", "(3,3)", "(3,5)", "(3,7)", "(3,9)", "(3,11)", "(3,13)",
            "(5,1)", "(5,3)", "(5,5)", "(5,7)", "(5,9)", "(5,11)", "(5,13)",
            "(7,1)", "(7,3)", "(7,5)", "(7,7)", "(7,9)", "(7,11)", "(7,13)",
            "(9,1)", "(9,3)", "(9,5)", "(9,7)", "(9,9)", "(9,11)", "(9,13)"]


## Read the struct log with the information
# Define a regular expression pattern to extract timestamp and source from log lines
log_pattern = re.compile(r'timestamp=(?P<timestamp>.*?) .*? sweep_0_poly=(?P<sweep_0_poly>\d+) sweep_0_off=(?P<sweep_0_off>\d+) sweep_0_bits=(?P<sweep_0_bits>\d+) sweep_1_poly=(?P<sweep_1_poly>\d+) sweep_1_off=(?P<sweep_1_off>\d+) sweep_1_bits=(?P<sweep_1_bits>\d+) sweep_2_poly=(?P<sweep_2_poly>\d+) sweep_2_off=(?P<sweep_2_off>\d+) sweep_2_bits=(?P<sweep_2_bits>\d+) sweep_3_poly=(?P<sweep_3_poly>\d+) sweep_3_off=(?P<sweep_3_off>\d+) sweep_3_bits=(?P<sweep_3_bits>\d+)')

# Create an empty list to store the extracted data
data = []

# Open the log file and iterate over each line
with open("Experiments/2 - 230504 - full grid - covering LH2/pydotbot.log", "r") as log_file:
    for line in log_file:
        # Extract timestamp and source from the line
        match = log_pattern.search(line)
        if match and "lh2-4" in line:
            # Append the extracted data to the list
            data.append({
                "timestamp": datetime.strptime(match.group("timestamp"), "%Y-%m-%dT%H:%M:%S.%fZ"),
                "poly_0": int(match.group("sweep_0_poly")),
                "off_0":  int(match.group("sweep_0_off")),
                "bits_0": int(match.group("sweep_0_bits")),
                "poly_1": int(match.group("sweep_1_poly")),
                "off_1":  int(match.group("sweep_1_off")),
                "bits_1": int(match.group("sweep_1_bits")),
                "poly_2": int(match.group("sweep_2_poly")),
                "off_2":  int(match.group("sweep_2_off")),
                "bits_2": int(match.group("sweep_2_bits")),
                "poly_3": int(match.group("sweep_3_poly")),
                "off_3":  int(match.group("sweep_3_off")),
                "bits_3": int(match.group("sweep_3_bits")),
            })
# Create a pandas DataFrame from the extracted data
df = pd.DataFrame(data)

## Remove lines that don't have the data from both lighthouses
# Define the conditions
cond1 = df[['poly_0', 'poly_1', 'poly_2', 'poly_3']].isin([0, 1]).sum(axis=1) == 2
cond2 = df[['poly_0', 'poly_1', 'poly_2', 'poly_3']].isin([2, 3]).sum(axis=1) == 2
cond = cond1 & cond2
# Filter the rows that meet the condition
df = df.loc[cond].reset_index(drop=True)

# Decide which data to use.
if (ALL_DATA):
    ip_df = pd.DataFrame(columns=df.columns)
    for z in z_lvl:
        for gp in exp_grid:
            start = datetime.strptime(time_data[z][gp]['start'], "%Y-%m-%dT%H:%M:%S.%fZ")
            end = datetime.strptime(time_data[z][gp]['end'], "%Y-%m-%dT%H:%M:%S.%fZ")
            ip_df = pd.concat([ip_df, df.loc[ (df['timestamp'] >= start) & (df['timestamp'] <= end)]])
    df = ip_df.reset_index(drop=True)

    # Extract a continuous slice of data.
    # start = datetime.strptime('2023-05-04T14:51:36.530023Z', "%Y-%m-%dT%H:%M:%S.%fZ")
    # end   = datetime.strptime('2023-05-04T17:50:48.181728Z', "%Y-%m-%dT%H:%M:%S.%fZ")
    # df = df.loc[ (df['timestamp'] >= start) & (df['timestamp'] <= end)].reset_index(drop=True)

# Only one measurement per datapoint
else:
    # Get one point from each important experiment position.
    ip_df = pd.DataFrame(columns=df.columns)
    for z in z_lvl:
        for gp in exp_grid:
            start = datetime.strptime(time_data[z][gp]['start'], "%Y-%m-%dT%H:%M:%S.%fZ")
            end = datetime.strptime(time_data[z][gp]['end'], "%Y-%m-%dT%H:%M:%S.%fZ")
            timestamp = start + (end - start)/2
            closest_idx = np.argmin(np.abs(df['timestamp'] - timestamp))
            row = df.iloc[closest_idx]
            # ip_df.append(row)
            ip_df = pd.concat([ip_df, row.to_frame().T])
    df = ip_df.reset_index(drop=True)



## Convert the data to a numpy a array and sort them to make them compatible with Cristobal's code
poly_array = df[["bits_0", "bits_1", "bits_2", "bits_3"]].to_numpy()
sorted_indices = np.argsort(df[['poly_0','poly_1','poly_2','poly_3']].values,axis=1)
bits_df = df[['bits_0','bits_1','bits_2','bits_3']]
sorted_bits = np.empty_like(bits_df)
for i, row in enumerate(sorted_indices):
    sorted_bits[i] = bits_df.values[i, row]


## Sort the columns for LH2-A and LH2-B separatedly.
c01 = np.sort(sorted_bits[:,0:2], axis=1).astype(int)
c23 = np.sort(sorted_bits[:,2:4], axis=1).astype(int)
# Re-join the columns and separate them into the variables used by cristobals code.
c0123 = np.hstack([c01, c23])
c0123 = np.sort(sorted_bits, axis=1).astype(int)
# This weird order to asign the columns is because there was an issue with the dataset, and the data order got jumbled.
c1A = c0123[:,0] 
c2A = c0123[:,2]
c1B = c0123[:,1]
c2B = c0123[:,3]


#############################################################################
###                           Save reordered data                         ###
#############################################################################

sorted_df = pd.DataFrame({
                          'timestamp' : df['timestamp'],

                          'LHA_1': c0123[:,0],

                          'LHA_2': c0123[:,2],

                          'LHB_1': c0123[:,1],

                          'LHB_2': c0123[:,3]},
                          index = df.index
                          )

sorted_df['real_x_mm'] = -1
sorted_df['real_y_mm'] = -1
sorted_df['real_z_mm'] = -1

for depth in z_lvl:
   if depth == 'z=0cm':  y = 0.0
   if depth == 'z=4cm':  y = 40.0
   if depth == 'z=8cm':  y = 80.0
   if depth == 'z=12cm': y = 120.0
   for coord in exp_grid:
      # Add the real x,y,z coordinate of each point
        sorted_df.loc[(df['timestamp'] >= datetime.strptime(time_data[depth][coord]['start'], "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] <= datetime.strptime(time_data[depth][coord]['end'], "%Y-%m-%dT%H:%M:%S.%fZ")), 'real_x_mm'] = time_data["point_positions"][coord]['x']
        sorted_df.loc[(df['timestamp'] >= datetime.strptime(time_data[depth][coord]['start'], "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] <= datetime.strptime(time_data[depth][coord]['end'], "%Y-%m-%dT%H:%M:%S.%fZ")), 'real_z_mm'] = time_data["point_positions"][coord]['z'] - 40  # Make the lower left corner the (0,0,0) of the cube
        sorted_df.loc[(df['timestamp'] >= datetime.strptime(time_data[depth][coord]['start'], "%Y-%m-%dT%H:%M:%S.%fZ")) & (df['timestamp'] <= datetime.strptime(time_data[depth][coord]['end'], "%Y-%m-%dT%H:%M:%S.%fZ")), 'real_y_mm'] = y

# Change the format of the timestamp column
# sorted_df['timestamp'] = sorted_df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
sorted_df['timestamp'].apply(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

# Clear all point for which you don't know the corresponding 3D coordinate, and print.
# sorted_df = sorted_df[sorted_df['real_z_mm'] != -1]  # This was moved down into the analysis part of the code. To avoid 
# sorted_df.to_csv('output.csv', index=True)

#############################################################################
###                           Clear Outliers                         ###
#############################################################################
# This goes grid point by grid point and removes datapoints who are too far away from mean.

filter_df = pd.DataFrame()
for z in z_lvl:
    for gp in exp_grid:
        start = datetime.strptime(time_data[z][gp]['start'], "%Y-%m-%dT%H:%M:%S.%fZ")
        end = datetime.strptime(time_data[z][gp]['end'], "%Y-%m-%dT%H:%M:%S.%fZ")
        # Find outliers by lookingvery strong sudden jumps the measurements of each gridpoints.
        prev_diff_df = abs(sorted_df.loc[ (sorted_df['timestamp'] >= start) & (sorted_df['timestamp'] <= end),['LHA_1', 'LHA_2', 'LHB_1', 'LHB_2']].diff().fillna(0).shift(1))
        next_diff_df = abs(sorted_df.loc[ (sorted_df['timestamp'] >= start) & (sorted_df['timestamp'] <= end),['LHA_1', 'LHA_2', 'LHB_1', 'LHB_2']].diff().fillna(0).shift(-1))
        # Get a boolean dataframe with indexes of the good measurement-s
        filter_df = pd.concat([filter_df, (prev_diff_df['LHA_1'] <= 20 ) & (next_diff_df['LHA_1'] <= 20 ) & (prev_diff_df['LHA_2'] <= 20 ) & (next_diff_df['LHA_2'] <= 20 ) & (prev_diff_df['LHB_1'] <= 20 ) & (next_diff_df['LHB_1'] <= 20 ) & (prev_diff_df['LHB_2'] <= 20 ) & (next_diff_df['LHB_2'] <= 20 )])
        # filter_df = pd.concat([filter_df, diff_df.le(20).all(axis=1)])

# Apply the filter that removes the outliers
sorted_df_bak = sorted_df
sorted_df = sorted_df.iloc[filter_df.index[filter_df[0] == True]].reset_index(drop=True)
# Get the cleaned values back on the variables needed for the next part of the code.
c1A = sorted_df['LHA_1'].values 
c2A = sorted_df['LHA_2'].values
c1B = sorted_df['LHB_1'].values
c2B = sorted_df['LHB_2'].values




#############################################################################
###                             Cristobal Code                            ###
#############################################################################

periods = [959000, 957000]   # These are the max counts for the LH2 mode 1 and 2 respecively

# Translate points into position from each camera
a1A = (c1A*8/periods[0])*2*np.pi  # Convert counts to angles traveled in the weird 40deg planes, in radians
a2A = (c2A*8/periods[0])*2*np.pi    + fcal['phase1-C']  # This is a calibration parameter from the LH2. More information, here: https://github.com/cntools/libsurvive/wiki/BSD-Calibration-Values
a1B = (c1B*8/periods[1])*2*np.pi
a2B = (c2B*8/periods[1])*2*np.pi    + fcal['phase1-B']

# Calculate the Horizontal (Azimuth) angle from the Lighthouse
azimuthA = (a1A+a2A)/2        
azimuthB = (a1B+a2B)/2
# Calulate the Vertical (Elevation, lower number, lower height) angle
elevationA = np.pi/2 - np.arctan2(np.sin(a2A/2-a1A/2-60*np.pi/180),np.tan(np.pi/6))
elevationB = np.pi/2 - np.arctan2(np.sin(a2B/2-a1B/2-60*np.pi/180),np.tan(np.pi/6))

pts_lighthouse_A = np.zeros((len(c1A),2))
pts_lighthouse_B = np.zeros((len(c1B),2))

# Project points into the unit plane (double check this equations.... somewhere.)
for i in range(len(c1A)):
  pts_lighthouse_A[i,0] = -np.tan(azimuthA[i])
  pts_lighthouse_A[i,1] = -np.sin(a2A[i]/2-a1A[i]/2-60*np.pi/180)/np.tan(np.pi/6)
for i in range(len(c1B)):
  pts_lighthouse_B[i,0] = -np.tan(azimuthB[i])
  pts_lighthouse_B[i,1] = -np.sin(a2B[i]/2-a1B[i]/2-60*np.pi/180)/np.tan(np.pi/6)

# Undistort points
dist_a = np.array([[1.0409, -2.4617, 0.1403, -0.0466, 2.6625]])
dist_b = np.array([[0.0924, -0.6955, -0.0070, 0.0334, 0.7030]])

# Define the intrinsic camera matrices of the LH2 basestations
Mat_A = np.array([[ 1.0282    ,  0.0396    , 0.0447],
                  [ 0.        ,  0.8734    ,-0.0117],
                  [ 0.        ,  0.        ,  1.  ]])

Mat_B = np.array([[ 1.0296    ,-0.0110     ,-0.0672],
                  [ 0.        , 0.9175,     0.0158],
                  [ 0.        ,  0.        ,  1.        ]])

pts_lighthouse_A_undist = cv2.undistortPoints(pts_lighthouse_A, Mat_A, dist_a).reshape((-1,2))
pts_lighthouse_B_undist = cv2.undistortPoints(pts_lighthouse_B, Mat_B, dist_b).reshape((-1,2))
# Back up the normal distorted points
pts_lighthouse_A_bak = pts_lighthouse_A
pts_lighthouse_B_bak = pts_lighthouse_B
pts_lighthouse_A = pts_lighthouse_A_undist
pts_lighthouse_B = pts_lighthouse_B_undist

# Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
sorted_df['LHA_proj_x'] = pts_lighthouse_A[:,0]
sorted_df['LHA_proj_y'] = pts_lighthouse_A[:,1]
sorted_df['LHB_proj_x'] = pts_lighthouse_B[:,0]
sorted_df['LHB_proj_y'] = pts_lighthouse_B[:,1]

## 3D SCENE SOLVING

# Obtain translation and rotation vectors
F, mask = cv2.findEssentialMat(pts_lighthouse_A, pts_lighthouse_B, method=cv2.FM_LMEDS)
# F, mask = cv2.findFundamentalMat(pts_lighthouse_A, pts_lighthouse_B, cv2.FM_LMEDS)
# points, R_star, t_star, mask = cv2.recoverPose(Mat_A.T @ F @ Mat_B, pts_lighthouse_A, pts_lighthouse_B)
points, R_star, t_star, mask = cv2.recoverPose( F, pts_lighthouse_A, pts_lighthouse_B)

# Triangulate the points
R_1 = np.eye(3,dtype='float64')
t_1 = np.zeros((3,1),dtype='float64')

# Calculate the Projection Matrices P
# The weird transpose everywhere is  because cv2.recover pose gives you the Camera 2 to Camera 1 transformation (which is the backwards of what you want.)
# To get the Cam1 -> Cam2 transformation, we need to invert this.
# R^-1 => R.T  (because rotation matrices are orthogonal)
# inv(t) => -t 
# That's where all the transpositions and negatives come from.
# Source: https://stackoverflow.com/a/45722936
P1 = Mat_A @ np.hstack([R_1.T, -R_1.T.dot(t_1)])
P2 = Mat_B @ np.hstack([R_star.T, -R_star.T.dot(t_star)])  
# The projection matrix is the intrisic matrix of the camera multiplied by the extrinsic matrix.
# When the intrinsic matrix is the identity.
# The results is the [ Rotation | translation ] in a 3x4 matrix


point3D = cv2.triangulatePoints(P1,P2,pts_lighthouse_B.T, pts_lighthouse_A.T).T
point3D = point3D[:, :3] / point3D[:, 3:4]

#############################################################################
###                             SolvePNP                            ###
#############################################################################

## LHA
obj_points = sorted_df[['real_x_mm', 'real_y_mm', 'real_z_mm']].values.astype(np.float32) # The calibration function ONLY likes float32, and wants the vector inside a list
img_points = sorted_df[['LHA_proj_x', 'LHA_proj_y']].values.astype(np.float32)
retval, r_a, t_a = cv2.solvePnP(obj_points, img_points, Mat_A, dist_a)
R_a, _jac = cv2.Rodrigues(r_a) # convert the rotation vecotr to a rotation matrix

## LHA
obj_points = sorted_df[['real_x_mm', 'real_y_mm', 'real_z_mm']].values.astype(np.float32) # The calibration function ONLY likes float32, and wants the vector inside a list
img_points = sorted_df[['LHB_proj_x', 'LHB_proj_y']].values.astype(np.float32)
retval, r_b, t_b = cv2.solvePnP(obj_points, img_points, Mat_B, dist_b)
R_b, _jac = cv2.Rodrigues(r_b) # convert the rotation vecotr to a rotation matrix

P1_pnp = Mat_A @ np.hstack([R_a, t_a])
P2_pnp = Mat_B @ np.hstack([R_b, t_b])  

point3D_pnp = cv2.triangulatePoints(P1_pnp,P2_pnp,pts_lighthouse_B.T, pts_lighthouse_A.T).T
point3D_pnp = point3D_pnp[:, :3] / point3D_pnp[:, 3:4] * 0.01

#############################################################################
###                       Analyze distances                               ###
#############################################################################
# Add The 3D point to the Dataframe that has the real coordinates, timestamps etc.
# This will help correlate which point are supposed to go where.
sorted_df['LH_x'] = point3D[:,0]
sorted_df['LH_y'] = point3D[:,2]   # We need to invert 2 of the axis because the LH2 frame Z == depth and Y == Height
sorted_df['LH_z'] = point3D[:,1]   # But the dataset assumes X = Horizontal, Y = Depth, Z = Height

# Remove the points that we don't know what their real world correspondance is.
sorted_df = sorted_df[sorted_df['real_z_mm'] != -1]

# Grab the point at (0,0,0) mm and (40,0,0) mm and use them to calibrate/scale the system.
scale_p1 = sorted_df.loc[(sorted_df['real_x_mm'] == 0)  & (sorted_df['real_y_mm'] == 0) & (sorted_df['real_z_mm'] == 0), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)
scale_p2 = sorted_df.loc[(sorted_df['real_x_mm'] == 40) & (sorted_df['real_y_mm'] == 0) & (sorted_df['real_z_mm'] == 0), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)
scale = 40 / np.linalg.norm(scale_p2 - scale_p1)
# Scale all the points
sorted_df['LH_x'] *= scale
sorted_df['LH_y'] *= scale
sorted_df['LH_z'] *= scale

##################### GET X AXIS DISTANCES
x_dist = []
for y in [0, 40, 80, 120]:
    for z in [0, 40, 80, 120, 160]:
        for x in [0, 40, 80, 160, 200]:  # We are missing x=240 because we only want the distance between the points, not the actual points.
            # Grab all the points
            p1 = sorted_df.loc[(sorted_df['real_x_mm'] == x)  & (sorted_df['real_y_mm'] == y) & (sorted_df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
            p2 = sorted_df.loc[(sorted_df['real_x_mm'] == x+40)  & (sorted_df['real_y_mm'] == y) & (sorted_df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values

            # Now permute all the distances between all the points in each position
            for v in p1:
                for w in p2:
                    x_dist.append(np.linalg.norm(v-w))

##################### GET Y AXIS DISTANCES
y_dist = []
for y in [0, 40, 80]:        # We are missing y=120 because we only want the distance between the points, not the actual points.
    for z in [0, 40, 80, 120, 160]:
        for x in [0, 40, 80, 160, 200, 240]: 
            # Grab all the points
            p1 = sorted_df.loc[(sorted_df['real_x_mm'] == x)  & (sorted_df['real_y_mm'] == y) & (sorted_df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
            p2 = sorted_df.loc[(sorted_df['real_x_mm'] == x)  & (sorted_df['real_y_mm'] == y+40) & (sorted_df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values

            # Now permute all the distances between all the points in each position
            for v in p1:
                for w in p2:
                    y_dist.append(np.linalg.norm(v-w))

##################### GET Z AXIS DISTANCES
z_dist = []
for y in [0, 40, 80, 120]:
    for z in [0, 40, 80, 120]:       # We are missing z=160 because we only want the distance between the points, not the actual points.
        for x in [0, 40, 80, 160, 200, 240]: 
            # Grab all the points
            p1 = sorted_df.loc[(sorted_df['real_x_mm'] == x)  & (sorted_df['real_y_mm'] == y) & (sorted_df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
            p2 = sorted_df.loc[(sorted_df['real_x_mm'] == x)  & (sorted_df['real_y_mm'] == y) & (sorted_df['real_z_mm'] == z+40), ['LH_x', 'LH_y', 'LH_z']].values

            # Now permute all the distances between all the points in each position
            for v in p1:
                for w in p2:
                    z_dist.append(np.linalg.norm(v-w))

# At the end, put all the distances together in an array and calculate mean and std
x_dist = np.array(x_dist)
y_dist = np.array(y_dist)
z_dist = np.array(z_dist)
# Remove ouliers, anything bigger than 1 meters gets removed.
x_dist = x_dist[x_dist <= 500]
y_dist = y_dist[y_dist <= 500]
z_dist = z_dist[z_dist <= 500]
print(f"X mean = {x_dist.mean() - 40}")
print(f"X std = {x_dist.std()}")
print(f"Y mean = {y_dist.mean() - 40}")
print(f"Y std = {y_dist.std()}")
print(f"Z mean = {z_dist.mean() - 40}")
print(f"Z std = {z_dist.std()}")

#############################################################################
###                             Plotting                                  ###
#############################################################################


######################################### Error Histograms ####################################### 

# Plot the results
fig = plt.figure(layout="constrained")
gs = GridSpec(2, 6, figure = fig)
# Define individual subplots
hist_x_ax    = fig.add_subplot(gs[0:2, 0:2])
hist_y_ax    = fig.add_subplot(gs[0:2, 2:4])
hist_z_ax    = fig.add_subplot(gs[0:2, 4:6])
axs = (hist_x_ax, hist_y_ax, hist_z_ax)

# X histogram
n, bins, patches = hist_x_ax.hist(x_dist, 50, density=False)
hist_x_ax.axvline(x=x_dist.mean(), color='red', label="Mean")
# Y histogram
n, bins, patches = hist_y_ax.hist(y_dist, 50, density=False)
hist_y_ax.axvline(x=y_dist.mean(), color='red', label="Mean")
# Z histogram
n, bins, patches = hist_z_ax.hist(z_dist, 50, density=False)
hist_z_ax.axvline(x=z_dist.mean(), color='red', label="Mean")

# Add labels and grids
for ax in axs:
    ax.grid()
    ax.legend()
# 
hist_x_ax.set_xlabel('Distance X axis [mm]')
hist_y_ax.set_xlabel('Distance Y axis [mm]')
hist_z_ax.set_xlabel('Distance Z axis [mm]')
hist_x_ax.set_ylabel('Measurements')
fig.suptitle('Distance Between Grid points\n(Should be 40mm)')

plt.show()


######################################### 3D Plotting #######################################  

## Plot the two coordinate systems
#  x is blue, y is red, z is green

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# First lighthouse:
ax.quiver(0,0,0,0.1,0,0,color='xkcd:blue',lw=3)
ax.quiver(0,0,0,0,0,-0.1,color='xkcd:red',lw=3)
ax.quiver(0,0,0,0,0.1,0,color='xkcd:green',lw=3)

# Second lighthouse:
t_star_rotated = np.array([t_star.item(0), t_star.item(1), t_star.item(2)])
print(R_star)
print(t_star_rotated)
x_axis = np.array([0.1,0,0])@np.linalg.inv(R_star)
y_axis = np.array([0,0.1,0])@np.linalg.inv(R_star)
z_axis = np.array([0,0,0.1])@np.linalg.inv(R_star)
ax.quiver(t_star_rotated[0],t_star_rotated[2],-t_star_rotated[1],x_axis[0],x_axis[2],-x_axis[1],color='xkcd:blue',lw=3)
ax.quiver(t_star_rotated[0],t_star_rotated[2],-t_star_rotated[1],y_axis[0],y_axis[2],-y_axis[1],color='xkcd:red',lw=3)
ax.quiver(t_star_rotated[0],t_star_rotated[2],-t_star_rotated[1],z_axis[0],z_axis[2],-z_axis[1],color='xkcd:green',lw=3)
ax.scatter(point3D[:,0],point3D[:,2],-point3D[:,1])

ax.text(-0.18,-0.1,0,s='LHA')
ax.text(t_star_rotated[0], t_star_rotated[2], -t_star_rotated[1],s='LHB')

# Set axis limits
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

plt.show()

######################################### 3D Plottingn - Solve_PNP #######################################  

## Plot the two coordinate systems
#  x is blue, y is red, z is green

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# First lighthouse:
ax.quiver(0,0,0,0.1,0,0,color='xkcd:blue',lw=3)
ax.quiver(0,0,0,0,0,-0.1,color='xkcd:red',lw=3)
ax.quiver(0,0,0,0,0.1,0,color='xkcd:green',lw=3)

# Second lighthouse:
t_b_rotated = np.array([t_b.item(0), t_b.item(1), t_b.item(2)])
print(R_b)
print(t_b_rotated)
x_axis = np.array([0.1,0,0])@np.linalg.inv(R_b)
y_axis = np.array([0,0.1,0])@np.linalg.inv(R_b)
z_axis = np.array([0,0,0.1])@np.linalg.inv(R_b)
ax.quiver(t_b_rotated[0],t_b_rotated[2],-t_b_rotated[1],x_axis[0],x_axis[2],-x_axis[1],color='xkcd:blue',lw=3)
ax.quiver(t_b_rotated[0],t_b_rotated[2],-t_b_rotated[1],y_axis[0],y_axis[2],-y_axis[1],color='xkcd:red',lw=3)
ax.quiver(t_b_rotated[0],t_b_rotated[2],-t_b_rotated[1],z_axis[0],z_axis[2],-z_axis[1],color='xkcd:green',lw=3)
ax.scatter(point3D_pnp[:,0],point3D_pnp[:,2],-point3D_pnp[:,1])

ax.text(-0.18,-0.1,0,s='LHA')
ax.text(t_star_rotated[0], t_star_rotated[2], -t_star_rotated[1],s='LHB')

# Set axis limits
ax.set_xlim3d(-1,1)
ax.set_ylim3d(-1,1)
ax.set_zlim3d(-1,1)

plt.show()

######################################### 2D projection #######################################  
# Plot the results
fig = plt.figure(layout="constrained")
gs = GridSpec(6, 3, figure = fig)
lh1_ax    = fig.add_subplot(gs[0:3, 0:3])
lh2_ax = fig.add_subplot(gs[3:6, 0:3])
# theta_ax = fig.add_subplot(gs[0, 3:])
# gyro_ax = fig.add_subplot(gs[1, 3:])
# v_ax = fig.add_subplot(gs[2, 3:])
# Join them together to iterate over them
# axs = (lh1_ax, theta_ax, gyro_ax, v_ax)
axs = (lh1_ax, lh2_ax)

# 2D plots - LH2 perspective
lh1_ax.scatter(pts_lighthouse_A_bak[:,0], pts_lighthouse_A_bak[:,1], edgecolor='r', facecolor='red', lw=1, label="LH1")
lh2_ax.scatter(pts_lighthouse_B_bak[:,0], pts_lighthouse_B_bak[:,1], edgecolor='r', facecolor='red', lw=1, label="LH2")
lh1_ax.scatter(pts_lighthouse_A[:,0], pts_lighthouse_A[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH1")
lh2_ax.scatter(pts_lighthouse_B[:,0], pts_lighthouse_B[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH2")
# Plot one synchronized point to check for a delay.

# Add labels and grids
for ax in axs:
    ax.grid()
    ax.legend()
lh1_ax.axis('equal')
lh2_ax.axis('equal')
# 
lh1_ax.set_xlabel('U [px]')
lh1_ax.set_ylabel('V [px]')
#
lh2_ax.set_xlabel('U [px]')
lh2_ax.set_ylabel('V [px]')
#
plt.show()
