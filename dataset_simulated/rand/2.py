import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import json
import pandas as pd
from scipy.spatial.transform import Rotation

#############################################################################
###                Define LH2 and point positions                         ###
#############################################################################
# LH2 A, pos and rotation (this is a LH -> world convertion)
lha_t = np.array([0,0,0])
lha_R, _ = cv2.Rodrigues(np.array([0, np.pi/4, 0]))    # tilted right (towards X+)
# LH2 B, pos and rotation
lhb_t = np.array([3,0,0])
lhb_R, _ = cv2.Rodrigues(np.array([0, -np.pi/4, 0 ]))  # tilted left (towards X-)
points = np.array([[1,0,1],
                   [2,0,1],
                   [1,0,2],
                   [2,0,2],
                   [1,0,3],
                   [2,0,3],
                   [1,-1,1],
                   [2,-1,1],
                   [1,-1,2],
                   [2,-1,2],
                   [1,-1,3],
                   [2,-1,3]], dtype=float)


obj_points = points - np.array([1,0,1])

#############################################################################
###                   Elevation and Azimuth angle                         ###
#############################################################################

# lhx_R.T is the inverse rotation matrix
# (points - lha_t).T is just for making them column vectors for correctly multiplying witht the rotation matrix.
p_a = lha_R.T @ (points - lha_t).T
p_b = lhb_R.T @ (points - lhb_t).T

elevation_a = np.arctan2( p_a[1], np.sqrt(p_a[0]**2 + p_a[2]**2))
elevation_b = np.arctan2( p_b[1], np.sqrt(p_b[0]**2 + p_b[2]**2))

azimuth_a = np.arctan2(p_a[0], p_a[2]) # XZ plan angle, 0 == +Z, positive numbers goes to +X
azimuth_b = np.arctan2(p_b[0], p_b[2])

#############################################################################
###                             Cristobal Code                            ###
#############################################################################

pts_lighthouse_A = np.array([np.tan(azimuth_a),       # horizontal pixel 
                             np.tan(elevation_a)]).T  # vertical   pixel 

pts_lighthouse_B = np.array([np.tan(azimuth_b),       # horizontal pixel 
                             np.tan(elevation_b)]).T  # vertical   pixel

## 3D SCENE SOLVING

# Obtain translation and rotation vectors
E, mask = cv2.findFundamentalMat(pts_lighthouse_A, pts_lighthouse_B, cv2.FM_LMEDS)
_, R_star, t_star, mask = cv2.recoverPose(E, pts_lighthouse_A, pts_lighthouse_B)

# Triangulate the points
R_1 = np.eye(3,dtype='float64')
t_1 = np.zeros((3,1),dtype='float64')

# P1 = np.hstack([R_1.T, -R_1.T.dot(t_1)])
# P2 = np.hstack([R_star.T, -R_star.T.dot(t_star)])  
P1 = np.hstack([R_1.T, -R_1.T.dot(t_1)])    # http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0053.html this is how you invert a projection matrix
P2 = np.hstack([R_star.T, -R_star.T.dot(t_star)])  

point3D = cv2.triangulatePoints(P1,P2,pts_lighthouse_B.T, pts_lighthouse_A.T).T
point3D = point3D[:, :3] / point3D[:, 3:4]

# Normalize the size of the triangulation to the real size of the dataset, to better compare.
# scale = np.linalg.norm(lhb_t - lha_t) / np.linalg.norm( t_star - t_1) 
# t_star *= scale
# point3D *= scale

# Rotate the dataset to superimpose the triangulation, and the ground truth.
# rot, _ = Rotation.align_vectors(lhb_t.reshape((1,-1)), t_star.T)  # align_vectors expects 2 (N,3), so we need to reshape and transform htem a bit.
# t_star = rot.as_matrix() @ t_star
# point3D = (rot.as_matrix() @ point3D.T).T

#############################################################################
###                          SolvePnP                                 ###
#############################################################################
# obj_points = points.copy()
# obj_points[:,0] -= 1
# obj_points[:,1] -= 1
img_points = pts_lighthouse_B
retval, r_pnp, t_pnp = cv2.solvePnP(obj_points, img_points, np.eye(3), np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
# R_a, _jac = cv2.Rodrigues(r_pnp) # convert the rotation vecotr to a rotation matrix

#############################################################################
###                          Save dataset                                 ###
#############################################################################

# Create a pandas DataFrame from the extracted data
df = pd.DataFrame({ 'azimuth_A':   azimuth_a,
                    'elevation_A': elevation_a,
                    'azimuth_B':   azimuth_b,
                    'elevation_B': elevation_b,
                    'real_x_mm':   points[:,0],
                    'real_y_mm':   points[:,1],
                    'real_z_mm':   points[:,2]})

df.to_csv('dataset_simulated/data.csv', index=True)


# Also save the canonical position of the lighthouse 
matrix_list = { "lha_t": lha_t.tolist(),
                "lha_R": lha_R.tolist(),
                "lhb_t": lhb_t.tolist(),
                "lhb_R": lhb_R.tolist()}

with open('dataset_simulated/basestation.json', 'w') as file:
    json.dump(matrix_list, file)

#############################################################################
###                             Plotting                                  ###
#############################################################################
############################# 2D projection #################################  
# Plot the results
fig = plt.figure(layout="constrained")
gs = GridSpec(6, 3, figure = fig)
lh1_ax    = fig.add_subplot(gs[0:3, 0:3])
lh2_ax = fig.add_subplot(gs[3:6, 0:3])
axs = (lh1_ax, lh2_ax)

# 2D plots - LH2 perspective
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
lh1_ax.invert_yaxis()
lh2_ax.invert_yaxis()
# plt.show()

######################################### 3D Plotting #######################################  

## Plot the two coordinate systems
#  x is blue, y is red, z is green

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

# Plot the lighthouse orientation
arrow = np.array([0,0,1]).reshape((-1,1))
ax2.quiver(lha_t[0],lha_t[1],lha_t[2], (lha_R @ arrow)[0], (lha_R @ arrow)[1], (lha_R @ arrow)[2], length=0.2, color='xkcd:red' )
ax2.quiver(lhb_t[0],lhb_t[1],lhb_t[2], (lhb_R @ arrow)[0], (lhb_R @ arrow)[1], (lhb_R @ arrow)[2], length=0.2, color='xkcd:red' )
ax2.quiver(t_1[0],t_1[1],t_1[2], (R_1 @ arrow)[0], (R_1 @ arrow)[1], (R_1 @ arrow)[2], length=0.2, color='xkcd:orange' )
ax2.quiver(t_star[0],t_star[1],t_star[2], (R_star @ arrow)[0], (R_star @ arrow)[1], (R_star @ arrow)[2], length=0.2, color='xkcd:orange' )

ax2.scatter(points[:,0],points[:,1],points[:,2], color='xkcd:blue', label='ground truth')
ax2.scatter(point3D[:,0],point3D[:,1],point3D[:,2], color='xkcd:green', alpha=0.5, label='triangulated')
ax2.scatter(lha_t[0],lha_t[1],lha_t[2], color='xkcd:red', label='LH1')
ax2.scatter(lhb_t[0],lhb_t[1],lhb_t[2], color='xkcd:red', label='LH2')
ax2.scatter(t_1[0],t_1[1],t_1[2], color='xkcd:orange', label='triang LH1')
ax2.scatter(t_star[0],t_star[1],t_star[2], color='xkcd:orange', label='triang LH2')

ax2.set_xlabel('X [mm]')
ax2.set_ylabel('Y [mm]')
ax2.set_zlabel('Z [mm]')

ax2.axis('equal')
ax2.legend()

# Set axis limits
# ax.set_xlim3d(-1,1)
# ax.set_ylim3d(-1,1)
# ax.set_zlim3d(-1,1)

plt.show()
a=1