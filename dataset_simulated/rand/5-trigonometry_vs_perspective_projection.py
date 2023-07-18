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
lha_t = np.array([0,0,0], dtype=float)
lha_R, _ = cv2.Rodrigues(np.array([0, 0, 0], dtype=float))    # tilted right (towards X+)
z=2
points = np.array([[0,0,z],
                   [1,0,z], # first round
                   [1,1,z],
                   [0,1,z],
                   [-1,1,z],
                   [-1,0,z],
                   [-1,-1,z],
                   [0,-1,z],
                   [1,-1,z],
                   [2,0,z], #second perimeter
                   [2,1,z],
                   [2,2,z],
                   [1,2,z],
                   [0,2,z],
                   [-1,2,z],
                   [-2,2,z],
                   [-2,1,z],
                   [-2,0,z],
                   [-2,-1,z],
                   [-2,-2,z],
                   [-1,-2,z],
                   [0,-2,z],
                   [1,-2,z],
                   [2,-2,z],
                   [2,-1,z]], dtype=float)


#############################################################################
###                   Elevation and Azimuth angle                         ###
#############################################################################

# lhx_R.T is the inverse rotation matrix
# (points - lha_t).T is just for making them column vectors for correctly multiplying witht the rotation matrix.
p_a = lha_R.T @ (points - lha_t).T

elevation_a = np.arctan2( p_a[1], np.sqrt(p_a[0]**2 + p_a[2]**2))
# elevation_a = np.arctan2( p_a[1], p_a[2])

azimuth_a = np.arctan2(p_a[0], p_a[2]) # XZ plan angle, 0 == +Z, positive numbers goes to +X

#############################################################################
###                       Projection to LH2 plane                         ###
#############################################################################

pts_a_lh = np.array([np.tan(azimuth_a),                             # horizontal pixel  
                     np.tan(elevation_a) * 1/np.cos(azimuth_a)]).T  # vertical   pixel 

# pts_a_lh = np.array([np.tan(azimuth_a),                             # horizontal pixel  
#                      np.tan(elevation_a)]).T  # vertical   pixel 

#############################################################################
###                    Project points with OpenCV                         ###
#############################################################################

rvec, _ = cv2.Rodrigues(lha_R.T)
tvec = -lha_R.T @ lha_t
pts_a_pp, _ = cv2.projectPoints(points, rvec, tvec, np.eye(3), np.zeros((5,1)))
pts_a_pp = pts_a_pp.reshape((-1,2))


#############################################################################
###                             Plotting                                  ###
#############################################################################
############################# 2D projection #################################  
# Plot the results
fig = plt.figure(layout="constrained")
gs = GridSpec(6, 3, figure = fig)
lh1_ax    = fig.add_subplot(gs[0:6, 0:6])
axs = (lh1_ax, )

# 2D plots - LH2 perspective
lh1_ax.scatter(pts_a_lh[:,0], pts_a_lh[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="Tangent projection")
lh1_ax.scatter(pts_a_pp[:,0], pts_a_pp[:,1], edgecolor='red', facecolor='red', alpha=0.5, lw=1, label="OpenCV projection")

# Add labels and grids
for ax in axs:
    ax.grid()
    ax.legend()
    ax.axis('equal')
    ax.set_xlabel('U [px]')
    ax.set_ylabel('V [px]')
# 
lh1_ax.invert_yaxis()
# plt.show()

######################################### 3D Plotting TAN projection #######################################  

## Plot the two coordinate systems
#  x is blue, y is red, z is green

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_proj_type('ortho')

# Plot the lighthouse orientation
arrow = np.array([0,0,1]).reshape((-1,1))
ax2.quiver(lha_t[0],lha_t[1],lha_t[2], (lha_R @ arrow)[0], (lha_R @ arrow)[1], (lha_R @ arrow)[2], length=0.2, color='xkcd:red' )
#
ax2.scatter(points[:,0],points[:,1],points[:,2], color='xkcd:blue', label='ground truth')
ax2.scatter(lha_t[0],lha_t[1],lha_t[2], color='xkcd:red', label='LH1')


ax2.axis('equal')
ax2.legend()

ax2.set_xlabel('X [mm]')
ax2.set_ylabel('Y [mm]')
ax2.set_zlabel('Z [mm]')


plt.show()
a=1 # does nothing, just to place a brealpoint