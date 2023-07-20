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

USE_CAMERA_MATRIX = False    

# file with the data to analyze
# data_file = 'dataset_experimental/data_1point.csv'
data_file = 'dataset_experimental/data_all.csv'
# data_file = 'dataset_simulated/data.csv'

# Base station calibration parameters
fcal = {'phase1-B': -0.005335,
        'phase1-C': -0.004791}

# Undistort points
dist_a = np.array([[1.0409, -2.4617, 0.1403, -0.0466, 2.6625]])
dist_b = np.array([[0.0924, -0.6955, -0.0070, 0.0334, 0.7030]])

# Define the intrinsic camera matrices of the LH2 basestations
Mat_A = np.array([[ 0.9646    ,  0.0017    , -0.0169],
                  [ 0.        ,  0.8706    ,0.0050],
                  [ 0.        ,  0.        ,  1.  ]])

Mat_B = np.array([[ 1.0256    , 0.0033     ,-0.0474],
                  [ 0.        , 0.9174,     -0.0129],
                  [ 0.        ,  0.        ,  1.        ]])

#############################################################################
###                                Functions                              ###
#############################################################################

def LH2_count_to_pixels(count_1, count_2, mode):
    """
    Convert the sweep count from a single lighthouse into pixel projected onto the LH2 image plane
    ---
    count_1 - int - polinomial count of the first sweep of the lighthouse
    count_2 - int - polinomial count of the second sweep of the lighthouse
    mode - int [0,1] - mode of the LH2, let's you know which polynomials are used for the LSFR. and at which speed the LH2 is rotating.
    """
    periods = [959000, 957000]

    # Translate points into position from each camera
    a1 = (count_1*8/periods[mode])*2*np.pi  # Convert counts to angles traveled in the weird 40deg planes, in radians
    a2 = (count_2*8/periods[mode])*2*np.pi   

    # Transfor sweep angles to azimuth and elevation coordinates
    azimuth   = (a1+a2)/2 
    elevation = np.pi/2 - np.arctan2(np.sin(a2/2-a1/2-60*np.pi/180),np.tan(np.pi/6)) 

    # Project the angles into the z=1 image plane
    pts_lighthouse = np.zeros((len(count_1),2))
    for i in range(len(count_1)):
        pts_lighthouse[i,0] = -np.tan(azimuth[i])
        pts_lighthouse[i,1] = -np.sin(a2[i]/2-a1[i]/2-60*np.pi/180)/np.tan(np.pi/6) * 1/np.cos(azimuth[i])

    # Return the projected points
    return pts_lighthouse

def LH2_angles_to_pixels(azimuth, elevation):
    """
    Project the Azimuth and Elevation angles of a LH2 basestation into the unit image plane.
    """
    pts_lighthouse = np.array([np.tan(azimuth),         # horizontal pixel  
                               np.tan(elevation) * 1/np.cos(azimuth)]).T    # vertical   pixel 
    return pts_lighthouse

def solve_3d_scene(pts_a, pts_b):
    """
    Use the projected LH2-camera points to triangulate the position of the LH2  basestations and of the LH2 receiver
    """
    # Obtain translation and rotation vectors
    F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_LMEDS)
    if USE_CAMERA_MATRIX:
        points, R_star, t_star, mask = cv2.recoverPose(Mat_A @ F @ Mat_B, pts_a, pts_b)
    else:
        points, R_star, t_star, mask = cv2.recoverPose(F, pts_a, pts_b)

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
    if USE_CAMERA_MATRIX:
        P1 = Mat_A @ np.hstack([R_1.T, -R_1.T.dot(t_1)])
        P2 = Mat_B @ np.hstack([R_star.T, -R_star.T.dot(t_star)])  
    else:
        P1 = np.hstack([R_1.T, -R_1.T.dot(t_1)])
        P2 = np.hstack([R_star.T, -R_star.T.dot(t_star)])  
    # The projection matrix is the intrisic matrix of the camera multiplied by the extrinsic matrix.
    # When the intrinsic matrix is the identity.
    # The results is the [ Rotation | translation ] in a 3x4 matrix

    point3D = cv2.triangulatePoints(P1, P2, pts_b.T, pts_a.T).T
    point3D = point3D[:, :3] / point3D[:, 3:4]

    # Return the triangulated 3D points
    # Return the position and orientation of the LH2-B wrt LH2-A
    return point3D, t_star, R_star

def scale_scene_to_real_size(df):
    """
    Code takes the solved 3D scene and scales the scene so that the distance between the gridpoints is indeed 40mm

    --- Input
    df: dataframe with the triangulated position of the grid-points and the real position of the grid-points
    --- Output
    df: dataframe with the updated scaled-up scene
    """
    # Grab the point at (0,0,0) mm and (40,0,0) mm and use them to calibrate/scale the system.
    scale_p1 = df.loc[(df['real_x_mm'] == 0)  & (df['real_y_mm'] == 0) & (df['real_z_mm'] == 0), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)
    scale_p2 = df.loc[(df['real_x_mm'] == 40) & (df['real_y_mm'] == 0) & (df['real_z_mm'] == 0), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)
    scale = 40 / np.linalg.norm(scale_p2 - scale_p1)
    # Scale all the points
    df['LH_x'] *= scale
    df['LH_y'] *= scale
    df['LH_z'] *= scale

    # Return scaled up scene
    return df

def compute_distance_between_grid_points(df):
    """
    Code that calculates the mean error and std deviation of the distance between grid points.

    --- Input
    df: dataframe with the scaled triangulated position of the grid-points and the real position of the grid-points
    --- Output
    x_dist: array float - X-axis distances between adjacent grid-points 
    y_dist: array float - Y-axis distances between adjacent grid-points 
    z_dist: array float - Z-axis distances between adjacent grid-points 
    """

    ##################### GET X AXIS DISTANCES
    x_dist = []
    for y in [0, 40, 80, 120]:
        for z in [0, 40, 80, 120, 160]:
            for x in [0, 40, 80, 160, 200]:  # We are missing x=240 because we only want the distance between the points, not the actual points.
                # Grab all the points
                p1 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
                p2 = df.loc[(df['real_x_mm'] == x+40)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values

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
                p1 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
                p2 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y+40) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values

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
                p1 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z), ['LH_x', 'LH_y', 'LH_z']].values
                p2 = df.loc[(df['real_x_mm'] == x)  & (df['real_y_mm'] == y) & (df['real_z_mm'] == z+40), ['LH_x', 'LH_y', 'LH_z']].values

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

    return x_dist, y_dist, z_dist

def correct_perspective(df):
    """
    Create a rotation and translation vector to move the reconstructed grid onto the origin for better comparison.
    Using an SVD, according to: https://nghiaho.com/?page_id=671
    """
    
    B = np.unique(df[['real_x_mm','real_y_mm','real_z_mm']].to_numpy(), axis=0)
    A = np.empty_like(B, dtype=float)
    for i in range(B.shape[0]):
        A[i] = df.loc[(df['real_x_mm'] == B[i,0])  & (df['real_y_mm'] == B[i,1]) & (df['real_z_mm'] == B[i,2]), ['LH_x', 'LH_y', 'LH_z']].values.mean(axis=0)

    # Get  all the reconstructed points
    A2 = df[['LH_x','LH_y','LH_z']].to_numpy().T

    # Convert the point to column vectors,
    # to match twhat the SVD algorithm expects
    A = A.T
    B = B.T

    # Get the centroids
    A_centroid = A.mean(axis=1).reshape((-1,1))
    B_centroid = B.mean(axis=1).reshape((-1,1))

    # Get H
    H = (A - A_centroid) @ (B - B_centroid).T

    # Do the SVD
    U, S, V = np.linalg.svd(H)

    # Get the rotation matrix
    R = V @ U.T

    if np.linalg.det(R) < 0:
        assert False, "R determinant is negative"

    # Get the ideal translation
    t = B_centroid - R @ A_centroid

    correct_points = (R@A2 + t)
    correct_points = correct_points.T

    # Update dataframe
    df['Rt_x'] = correct_points[:,0]
    df['Rt_y'] = correct_points[:,1]
    df['Rt_z'] = correct_points[:,2]
    return df

def plot_distance_histograms(x_dist, y_dist, z_dist):
    """
    Plot a histogram of the distance between grid-points in the X, Y and Z direction.
    Also prints the mean and std deviation of this distances.
    """
    # Print mean and std deviation of the distance in all 3 x_axis
    print(f"X mean = {x_dist.mean()}")
    print(f"X std = {x_dist.std()}")
    print(f"Y mean = {y_dist.mean()}")
    print(f"Y std = {y_dist.std()}")
    print(f"Z mean = {z_dist.mean()}")
    print(f"Z std = {z_dist.std()}")
    
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

def plot_reconstructed_3D_scene(point3D, t_star, R_star, df=None):
    """
    Plot a 3D scene with the traingulated points previously calculated
    ---
    input:
    point3D - array [3,N] - triangulated points of the positions of the LH2 receveier
    t_star  - array [3,1] - Translation vector between the first and the second lighthouse basestation
    R_star  - array [3,3] - Rotation matrix between the first and the second lighthouse basestation
    df      - dataframe   - dataframe holding the real positions of the gridpoints
    """
    ## Plot the two coordinate systems
    #  x is blue, y is red, z is green

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
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
    ax.quiver(t_star_rotated[0],t_star_rotated[2],-t_star_rotated[1],x_axis[0],x_axis[2],-x_axis[1], color='xkcd:blue',lw=3)
    ax.quiver(t_star_rotated[0],t_star_rotated[2],-t_star_rotated[1],y_axis[0],y_axis[2],-y_axis[1],color='xkcd:red',lw=3)
    ax.quiver(t_star_rotated[0],t_star_rotated[2],-t_star_rotated[1],z_axis[0],z_axis[2],-z_axis[1],color='xkcd:green',lw=3)
    ax.scatter(point3D[:,0],point3D[:,2],-point3D[:,1], alpha=0.1)

    # Check if this is the simulated dataset. If yes, plot the correct basestation pose and points
    if 'simulated' in data_file:
        with open("dataset_simulated/basestation.json", "r") as json_file:
            lh2_pose = json.load(json_file)

        lha_t = np.array(lh2_pose['lha_t'])
        lha_R = np.array(lh2_pose['lha_R'])
        lhb_t = np.array(lh2_pose['lhb_t'])
        lhb_R = np.array(lh2_pose['lhb_R'])

        if df is not None:
            real_point3D = df[['real_x_mm','real_y_mm','real_z_mm']].to_numpy()
            ax.scatter(real_point3D[:,0],real_point3D[:,1],real_point3D[:,2], color='xkcd:green', label='ground truth')

        arrow = np.array([1,0,0]).reshape((-1,1))
        ax.quiver(lha_t[0],lha_t[1],lha_t[2], (lha_R @ arrow)[0], (lha_R @ arrow)[1], (lha_R @ arrow)[2], length=0.2, color='xkcd:red' )
        ax.quiver(lhb_t[0],lhb_t[1],lhb_t[2], (lhb_R @ arrow)[0], (lhb_R @ arrow)[1], (lhb_R @ arrow)[2], length=0.2, color='xkcd:red' )
        ax.scatter(lha_t[0],lha_t[1],lha_t[2], color='xkcd:red', label='LH1')
        ax.scatter(lhb_t[0],lhb_t[1],lhb_t[2], color='xkcd:red', label='LH2')

    # Plot the real 
    ax.text(-0.18,-0.1,0,s='LHA')
    ax.text(t_star_rotated[0], t_star_rotated[2], -t_star_rotated[1],s='LHB')

    ax.axis('equal')
    ax.set_title('Corrected - Elevation Angle')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')


    plt.show()

def plot_projected_LH_views(pts_a, pts_b):
    """
    Plot the projected views from each of the lighthouse
    """
    fig = plt.figure(layout="constrained")
    gs = GridSpec(6, 3, figure = fig)
    lh1_ax    = fig.add_subplot(gs[0:3, 0:3])
    lh2_ax = fig.add_subplot(gs[3:6, 0:3])
    axs = (lh1_ax, lh2_ax)

    # 2D plots - LH2 perspective
    lh1_ax.scatter(pts_a[:,0], pts_a[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH1")
    lh2_ax.scatter(pts_b[:,0], pts_b[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH2")
    # Plot one synchronized point to check for a delay.

    # Add labels and grids
    for ax in axs:
        ax.grid()
        ax.legend()
        ax.axis('equal')
        ax.set_xlabel('U [px]')
        ax.set_ylabel('V [px]')
        ax.invert_yaxis()

    plt.show()

def plot_transformed_3D_data(df):
    """
    Plot the difference between the ground truth and the reconstructed data.
    This will make it easy to plot the error. 
    """
    # Create a figure to plot
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_proj_type('ortho')

    # Plot the ground truth points
    real_points = np.unique(df[['real_x_mm','real_y_mm','real_z_mm']].to_numpy(), axis=0)  # Plot just a single point per grid position to save on computational power.
    ax2.scatter(real_points[:,0], real_points[:,1], real_points[:,2], alpha=0.5 ,color='xkcd:green', label="Ground Truth")
    # Plot real dataset points
    points = df[['Rt_x','Rt_y','Rt_z']].to_numpy()
    ax2.scatter(points[:,0], points[:,1], points[:,2], alpha=0.05, color='xkcd:blue', label="Real Data")
 
    ax2.axis('equal')
    ax2.legend()

    # ax2.set_title('Tangent projection')
    ax2.set_xlabel('X [mm]')
    ax2.set_ylabel('Y [mm]')
    ax2.set_zlabel('Z [mm]')

    plt.show()

    # ax.scatter(p000[:,0],p000[:,2],-p000[:,1], color='green')

def plot_error_histogram(df):
    """ 
    Calculate and plot a histogram  of the error of the reconstructed points, vs. 
    the ground truth.
    """
    # Extract needed data from the main dataframe
    points = df[['Rt_x','Rt_y','Rt_z']].to_numpy()
    ground_truth = df[['real_x_mm','real_y_mm','real_z_mm']].to_numpy()

    # Calculate distance between points and their ground truth
    errors =  np.linalg.norm(ground_truth - points, axis=1)
    # print the mean and standard deviation
    print(f"Mean Absolute Error = {errors.mean()} mm")
    print(f"Root Mean Square Error = {np.sqrt((errors**2).mean())} mm")
    print(f"Error Standard Deviation = {errors.std()} mm")

    # prepare the plot
    fig = plt.figure(layout="constrained")
    gs = GridSpec(3, 3, figure = fig)
    hist_ax    = fig.add_subplot(gs[0:3, 0:3])
    axs = (hist_ax,)

    # Plot the error histogram
    n, bins, patches = hist_ax.hist(errors, 50, density=False)
    hist_ax.axvline(x=errors.mean(), color='red', label="Mean")

    for ax in axs:
        ax.grid()
        ax.legend()
    
    hist_ax.set_xlabel('Distance Error [mm]')
    hist_ax.set_ylabel('Measurements')

    plt.show()

    return

#############################################################################
###                                  Main                                 ###
#############################################################################

if __name__ == "__main__":

    # Import data
    df=pd.read_csv(data_file, index_col=0)

    # Project sweep angles on to the z=1 image plane
    if 'azimuth_A' not in df.columns:   
        # Use real dataset directly from lfsr counts
        pts_lighthouse_A = LH2_count_to_pixels(df['LHA_count_1'].values, df['LHA_count_2'].values, 0)
        pts_lighthouse_B = LH2_count_to_pixels(df['LHB_count_1'].values, df['LHB_count_2'].values, 1)
    else: 
        # Use simulated dataset from azimuth and elevation angles
        pts_lighthouse_A = LH2_angles_to_pixels(df['azimuth_A'].values, df['elevation_A'].values)
        pts_lighthouse_B = LH2_angles_to_pixels(df['azimuth_B'].values, df['elevation_B'].values)      

    # Add the LH2 projected matrix into the dataframe that holds the info about what point is where in real life.
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

    # Scale the scene to real size
    if 'experimental' in data_file:
        df = scale_scene_to_real_size(df)

    # Compute distances between gridpoints
    x_dist, y_dist, z_dist = compute_distance_between_grid_points(df)

    # Bring reconstructed data to the origin for easier comparison
    df = correct_perspective(df)


    #############################################################################
    ###                             Plotting                                  ###
    #############################################################################
    # Plot X,Y,Z gridpoint distance histograms. If the dataset is real.
    # if 'experimental' in data_file:
    #     plot_distance_histograms(x_dist, y_dist, z_dist)

    # Plot 3D reconstructed scene
    # plot_reconstructed_3D_scene(point3D, t_star, R_star, df)

    # Plot projected views of the lighthouse
    # plot_projected_LH_views(pts_lighthouse_A, pts_lighthouse_B)

    # Plot superimposed "captured data" vs. "ground truth", and error histogram
    plot_transformed_3D_data(df)
    plot_error_histogram(df)
