import pandas as pd
from datetime import datetime
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2

import random
import math
import cv2
np.set_printoptions(formatter={'float': "{0:.4f}".format})

#############################################################################
###                                Options                                ###
#############################################################################
## To use all the data (instead of a measurement per grip point), set this to true
ALL_DATA = True


##################################################################################################################
###                                     Non Planar Calibration Functions                                       ###
###        https://github.com/chandnii7/Camera-Calibration/blob/main/src/Program2_A4_Chandni_Patel.ipynb       ###
##################################################################################################################

#input point pairs
def GetFilePoint(filename):    
    point_3D, point_2D = [], []
    with open(filename) as f:
        point_pairs = f.readlines()
        for i in point_pairs:
            pt = i.split()
            point_3D.append([float(p) for p in pt[:3]])
            point_2D.append([float(p) for p in pt[3:]])
    print("\nTotal Point Pairs = ", len(point_3D))
    return point_3D, point_2D

#required matrix A
def GetMatrixA(point_3D, point_2D):
    A = []
    array_0 = np.zeros(4)
    for i, j in zip(point_3D, point_2D):        
        #convert 3D to 3DH
        pi = np.concatenate([np.array(i), [1]])
        #row 1 for point i
        xipi = j[0] * pi
        r1 = np.concatenate([pi, array_0, -xipi])
        A.append(r1)
        #row 2 for point i
        yipi = j[1] * pi
        r2 = np.concatenate([array_0, pi, -yipi])        
        A.append(r2)
    return A

#projection matrix M
def GetProjectionMatrixM(A):    
    M = []
    u, d, vT = np.linalg.svd(A, full_matrices = True)
    #get M from s i.e. vector with the singular values
    M = vT[-1].reshape(3, 4)   
    for i in range(len(d)):
        if (round(d[i],1) == 0):
            M = vT[i].reshape(3, 4)
            break
    return M

#mean square error
def GetMSE(point_3D, point_2D, M):    
    sum_error = 0
    m1, m2, m3 = M[0][:4], M[1][:4], M[2][:4] 
    for i, j in zip(point_3D, point_2D):
        #convert 3D to 3DH
        pi = np.concatenate([np.array(i), [1]])
        #compute xi & yi using M
        computed_xi = (m1.T.dot(pi)) / (m3.T.dot(pi))
        computed_yi = (m2.T.dot(pi)) / (m3.T.dot(pi))
        #sum of all errors
        sum_error += ((j[0] - computed_xi) ** 2 + (j[1] - computed_yi) ** 2)
    #E = (sum of errors) / m
    mse = sum_error / len(point_3D)
    print("\nMean Square Error = ", round(mse, 4))

##########################################  RANSAC Functions ####################################################

def GetDistance(M, point_3D, point_2D):
    d = []
    m1, m2, m3 = M[0][:4], M[1][:4], M[2][:4]    
    for i, j in zip(point_3D, point_2D):
        #convert 3D to 3DH
        pi = np.concatenate([np.array(i), [1]])
        #compute xi & yi using M
        computed_xi = (m1.T.dot(pi)) / (m3.T.dot(pi))
        computed_yi = (m2.T.dot(pi)) / (m3.T.dot(pi))
        d.append(np.sqrt((j[0] - computed_xi) ** 2 + (j[1] - computed_yi) ** 2))
    return d

def ApplyRANSAC(point_3D, point_2D):
    w = 0.5    
    count = 0
    num_inliner = 6
    np.random.seed(0)
    prob, kmax, nmin, nmax = 0.99, 10e3, 20, 500
    # with open('RANSAC.config', 'r') as conf:
    #     #probability that at least one of the draws is free from outlier
    #     prob = float(conf.readline().split()[0])
    #     #maximum number of draws that can be performed
    #     kmax = int(conf.readline().split()[0])
    #     #minimum number of points needed to fit model
    #     nmin = int(conf.readline().split()[0])
    #     #maximun number of points needed to fit model
    #     nmax = int(conf.readline().split()[0])     
    
    A = GetMatrixA(point_3D, point_2D)
    M = GetProjectionMatrixM(A) 
    
    d = GetDistance(M, point_3D, point_2D)
    t = 1.5 * np.median(d)
    
    while(count < kmax):        
        i = np.random.choice(len(point_3D), nmax)
        random_3D_points, random_2D_points = np.array(point_3D)[i], np.array(point_2D)[i]
        
        A = GetMatrixA(random_3D_points, random_2D_points)
        M = GetProjectionMatrixM(A)     
        
        d = GetDistance(M, point_3D, point_2D)
        inliner_points = []
        for i, d in enumerate(d):
            if d < t:
                inliner_points.append(i)
        
        if len(inliner_points) >= num_inliner:
            num_inliner = len(inliner_points)
            inliner_3D_points, inliner_2D_points = np.array(point_3D)[inliner_points], np.array(point_2D)[inliner_points]            
            A = GetMatrixA(inliner_3D_points, inliner_2D_points)
            M = GetProjectionMatrixM(A)  
            d = GetDistance(M, point_3D, point_2D)
            
        w = float(len(inliner_points))/float(len(point_2D))
        kmax = float(math.log(1 - prob)) / np.absolute(math.log(1 - (w ** nmax)))
        t = 1.5 * np.median(d)
        count += 1;
        
    print("\nNumber of inliers", num_inliner)
    return A, M

def Non_Planar_Calibration(point_3D, point_2D, WithRANSAC = False): 
    print("\n\n*************************************************************")
    
    
    A, M = [], []    
    if WithRANSAC:
        print("\nCamera Parameters from Non-Planar Camera Calibration with RANSAC: ")
        A, M = ApplyRANSAC(point_3D, point_2D)        
    else:        
        print("\nCamera Parameters from Non-Planar Camera Calibration: ")
        A = GetMatrixA(point_3D, point_2D)
        M = GetProjectionMatrixM(A)        
    
    #additional variables
    a1 = M[0][:3] #vector a1
    a2 = M[1][:3] #vector a2
    a3 = M[2][:3] #vector a3
    b = [] #vector b
    for i in range(len(M)):
        b.append(M[i][3])       
    print("\nM^ = ", M) 

    #|p| = 1 / |a3|
    norm_rho = 1 / np.linalg.norm(a3)
    #u0 = |p|^2 a1.a3    
    u0 = norm_rho ** 2 * (a1.dot(a3.T))    
    #v0 = |p|^2 a2.a3
    v0 = norm_rho ** 2 * (a2.dot(a3.T))
    print("\nu0 = ", round(u0,4), "\t\tv0 = ", round(v0,4))
    
    #av = (|p|^2 a2.a2 - v0^2)^1/2
    av = np.sqrt(norm_rho ** 2 * a2.dot(a2.T) - v0 ** 2)    
    #s = (|p|^4 / av)[(a1 x a3).(a2 x a3)]
    s = (norm_rho ** 4) / av * np.cross(a1, a3).dot(np.cross(a2, a3))
    #au = (|p|^2 a1.a1 - u0^2 - s^2)^1/2
    au = np.sqrt(norm_rho ** 2 * a1.dot(a1.T) - u0 ** 2 - s ** 2)
    print("\nalpha u = ", round(au,4), "\talpha v = ", round(av,4))
    
    #sign of rho
    sign_rho = np.sign(b[2])
    rho = sign_rho * norm_rho
    print("\ns = ", round(s,4), "\t\tρ = ", round(rho,4))
    
    #get K*
    K_star = np.array([[au, s, u0],[0, av, v0],[0, 0, 1]])
    print("\nK* = ", K_star)    
    
    #T* = sign_rho |p| (K*)^-1 b
    T_star = rho * np.linalg.inv(K_star).dot(b).T
    print("\nT* = ", T_star)
    
    #r3 = sign_rho |p| a3
    r3 = rho * a3.T
    #r1 = (|p|^2 / av) (a2 x a3)
    r1 = norm_rho ** 2 / av * np.cross(a2, a3)
    #r2 = r3 x r1
    r2 = np.cross(r3, r1)
    #R* = [r1^T r2^T r3^T]^T
    R_star = np.array([r1.T, r2.T, r3.T])
    print("\nR* = ", R_star)
    
    #projection matrix M = pM^
    M = rho * M
    print("\nM = ", M)
    
    GetMSE(point_3D, point_2D, M)

#############################################################################
###                              My Code                                  ###
#############################################################################

## Read the Timestamps to get which point correspond to which positions
with open("dataset_experimental/raw_data/timestamps.json", "r") as file:
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
with open("dataset_experimental/raw_data/pydotbot.log", "r") as log_file:
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

# Get rid of extreme outliers
# threshold = 15000
# for i in range(c1A.shape[0]):    
#     if (abs(c1A[i]-c1A[i-1])>threshold):
#         c1A[i] = c1A[i-1]
#     if (abs(c2A[i]-c2A[i-1])>threshold):
#         c2A[i] = c2A[i-1]
#     if (abs(c1B[i]-c1B[i-1])>threshold):
#         c1B[i] = c1B[i-1]
#     if (abs(c2B[i]-c2B[i-1])>threshold):
#         c2B[i] = c2B[i-1]

periods = [959000, 957000]   # These are the max counts for the LH2 mode 1 and 2 respecively

# Translate points into position from each camera
a1A = (c1A*8/periods[0])*2*np.pi  # Convert counts to angles traveled in the weird 40deg planes, in radians
a2A = (c2A*8/periods[0])*2*np.pi
a1B = (c1B*8/periods[1])*2*np.pi
a2B = (c2B*8/periods[1])*2*np.pi

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
  pts_lighthouse_A[i,1] = -np.sin(a2A[i]/2-a1A[i]/2-60*np.pi/180)/np.tan(np.pi/6)  * 1/np.cos(azimuthA[i])
for i in range(len(c1B)):
  pts_lighthouse_B[i,0] = -np.tan(azimuthB[i])
  pts_lighthouse_B[i,1] = -np.sin(a2B[i]/2-a1B[i]/2-60*np.pi/180)/np.tan(np.pi/6)  * 1/np.cos(azimuthB[i])


#############################################################################
###                           3D CALIBRATION                              ###
#############################################################################

# Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
sorted_df['LHA_proj_x'] = pts_lighthouse_A[:,0]
sorted_df['LHA_proj_y'] = pts_lighthouse_A[:,1]
sorted_df['LHB_proj_x'] = pts_lighthouse_B[:,0]
sorted_df['LHB_proj_y'] = pts_lighthouse_B[:,1]



# Now, run the actual calibration.
point_3D = sorted_df[['real_x_mm', 'real_y_mm', 'real_z_mm']].values.astype(np.float32) # The calibration function ONLY likes float32, and wants the vector inside a list
point_2D = sorted_df[['LHB_proj_x', 'LHB_proj_y']].values.astype(np.float32)
Non_Planar_Calibration(point_3D, point_2D, WithRANSAC = False)




#############################################################################
###                       OPENCV CALIBRATION                              ###
#############################################################################

# Get the front planar surface to get an idea of the correct matrix.
# objpoints = [sorted_df.loc[sorted_df['real_y_mm'] == 0, ['real_x_mm', 'real_z_mm', 'real_y_mm']].values.astype(np.float32) ] # The calibration function ONLY likes float32, and wants the vector inside a list. ALSO, OpenCV assumes Z-axis == Depth, so we need to invert our axis, because for us Y == Depth
# imgpoints = [sorted_df.loc[sorted_df['real_y_mm'] == 0, ['LHB_proj_x', 'LHB_proj_y']].values.astype(np.float32)]
# ret, mtx_guess, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, [1, 1], None, None)    # 12,4 is the size in meters of the image plane of the Lighthouse. Calculated from basic trigonometry. and this FOV numbers (source: https://www.reddit.com/r/ValveIndex/comments/emed7l/how_wide_is_the_index_base_station_fov/)


# Now, run the actual calibration.
objpoints = [sorted_df[['real_x_mm', 'real_y_mm', 'real_z_mm']].values.astype(np.float32) ] # The calibration function ONLY likes float32, and wants the vector inside a list
imgpoints = [sorted_df[['LHB_proj_x', 'LHB_proj_y']].values.astype(np.float32)]
# Note OpenCV is not designed to calibrate from non-planar geometries.
# If you do, you have to provide a guess for the camera matrix. We use a previously calibrated coplanar face as the guess
# source: https://stackoverflow.com/a/68799098
mtx_guess = np.array([[ 1.0296    ,  0.0    , 0.0],
                  [ 0.        ,  0.9175    ,  0.0158],
                  [ 0.        ,  0.        ,  1.  ]])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, [12, 4], mtx_guess, None,flags=cv2.CALIB_USE_INTRINSIC_GUESS)  

print('boop')
#############################################################################
###                             Plotting                                  ###
#############################################################################


######################################### Error Histograms ####################################### 

######################################### 3D Plotting #######################################  


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
lh1_ax.scatter(pts_lighthouse_A[:,0], pts_lighthouse_A[:,1], edgecolor='r', facecolor='red', lw=1, label="LH1")
lh2_ax.scatter(pts_lighthouse_B[:,0], pts_lighthouse_B[:,1], edgecolor='r', facecolor='red', lw=1, label="LH2")
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
