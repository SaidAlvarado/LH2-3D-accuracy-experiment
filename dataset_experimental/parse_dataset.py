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
ALL_DATA = False
# Base station calibration parameters
fcal = {'phase1-B': -0.005335,
        'phase1-C': -0.004791}


#############################################################################
###                                Code                                   ###
#############################################################################

## Read the Timestamps to get which point correspond to which positions
with open("dataset_experimental/raw data/timestamps.json", "r") as file:
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
with open("dataset_experimental/raw data/pydotbot.log", "r") as log_file:
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

                          'LHA_count_1': c0123[:,0],

                          'LHA_count_2': c0123[:,2],

                          'LHB_count_1': c0123[:,1],

                          'LHB_count_2': c0123[:,3]},
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

# # Clear all point for which you don't know the corresponding 3D coordinate, and print.
# sorted_df = sorted_df[sorted_df['real_z_mm'] != -1]  # This was moved down into the analysis part of the code. To avoid 
# sorted_df.to_csv('output.csv', index=True)

#############################################################################
###                           Clear Outliers                         ###
#############################################################################
# This goes grid point by grid point and removes datapoints who are too far away from mean.
if ALL_DATA:
    filter_df = pd.DataFrame()
    for z in z_lvl:
        for gp in exp_grid:
            start = datetime.strptime(time_data[z][gp]['start'], "%Y-%m-%dT%H:%M:%S.%fZ")
            end = datetime.strptime(time_data[z][gp]['end'], "%Y-%m-%dT%H:%M:%S.%fZ")
            # Find outliers by lookingvery strong sudden jumps the measurements of each gridpoints.
            prev_diff_df = abs(sorted_df.loc[ (sorted_df['timestamp'] >= start) & (sorted_df['timestamp'] <= end),['LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2']].diff().fillna(0).shift(1))
            next_diff_df = abs(sorted_df.loc[ (sorted_df['timestamp'] >= start) & (sorted_df['timestamp'] <= end),['LHA_count_1', 'LHA_count_2', 'LHB_count_1', 'LHB_count_2']].diff().fillna(0).shift(-1))
            # Get a boolean dataframe with indexes of the good measurement-s
            filter_df = pd.concat([filter_df, (prev_diff_df['LHA_count_1'] <= 20 ) & (next_diff_df['LHA_count_1'] <= 20 ) & (prev_diff_df['LHA_count_2'] <= 20 ) & (next_diff_df['LHA_count_2'] <= 20 ) & (prev_diff_df['LHB_count_1'] <= 20 ) & (next_diff_df['LHB_count_1'] <= 20 ) & (prev_diff_df['LHB_count_2'] <= 20 ) & (next_diff_df['LHB_count_2'] <= 20 )])
            # filter_df = pd.concat([filter_df, diff_df.le(20).all(axis=1)])

    # Apply the filter that removes the outliers
    sorted_df_bak = sorted_df
    sorted_df = sorted_df.iloc[filter_df.index[filter_df[0] == True]].reset_index(drop=True)
# Get the cleaned values back on the variables needed for the next part of the code.
c1A = sorted_df['LHA_count_1'].values 
c2A = sorted_df['LHA_count_2'].values
c1B = sorted_df['LHB_count_1'].values
c2B = sorted_df['LHB_count_2'].values


# Clear all point for which you don't know the corresponding 3D coordinate, and print.
sorted_df = sorted_df[sorted_df['real_z_mm'] != -1]  # This was moved down into the analysis part of the code. To avoid 
if ALL_DATA:
    sorted_df.to_csv('dataset_experimental/data_all.csv', index=True)
else:
    sorted_df.to_csv('dataset_experimental/data_1point.csv', index=True)


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

