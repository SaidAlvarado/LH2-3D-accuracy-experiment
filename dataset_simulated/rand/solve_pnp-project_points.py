import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from scipy.spatial.transform import Rotation

#############################################################################
###                Define LH2 and point positions                         ###
#############################################################################
# LH2 A, pos and rotation
lha_t = np.array([0,0,0])
lha_R, _ = cv2.Rodrigues(np.array([0, np.pi/4, 0]))    # tilted right (towards X+)
# LH2 B, pos and rotation
lhb_t = np.array([6,0,0])
lhb_R, _ = cv2.Rodrigues(np.array([0, -np.pi/4, 0 ]))  # tilted left (towards X-)
points = np.array([[2,0,2],
                   [2,2,2],
                   [3,-2,2],
                   [3,-2,2],
                   [3,2,2],
                   [3,3,2],
                   [4,0,2],
                   [4,2,2],
                   [4,-2,2]], dtype=float)


obj_points = points - np.array([2,0,2])

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
###                       Projection to LH2 plane                         ###
#############################################################################

pts_lighthouse_A = np.array([np.tan(azimuth_a),       # horizontal pixel  
                             np.tan(elevation_a)]).T  # vertical   pixel 

pts_lighthouse_B = np.array([np.tan(azimuth_b),       # horizontal pixel 
                             np.tan(elevation_b)]).T  # vertical   pixel

#############################################################################
###                    Project points with OpenCV                         ###
#############################################################################
rvec, _ = cv2.Rodrigues(lha_R.T)
tvec = -lha_R.T @ lha_t
proj_pts_A, _ = cv2.projectPoints(points, rvec, tvec, np.eye(3), np.zeros((5,1)))
proj_pts_A = proj_pts_A.reshape(pts_lighthouse_A.shape)

rvec, _ = cv2.Rodrigues(lhb_R.T)
tvec = -lhb_R.T @ lhb_t
proj_pts_B, _ = cv2.projectPoints(points, rvec, tvec, np.eye(3), np.zeros((5,1)))
proj_pts_B = proj_pts_B.reshape(pts_lighthouse_B.shape)

#############################################################################
###                          SolvePnP                                 ###
#############################################################################

img_points = pts_lighthouse_B
retval, r_pnp, t_pnp = cv2.solvePnP(obj_points, img_points, np.eye(3), np.zeros((5,1)), flags=cv2.SOLVEPNP_ITERATIVE)


img_points = proj_pts_B
retval, r_pnp_pp, t_pnp_pp = cv2.solvePnP(obj_points, img_points, np.eye(3), np.zeros((5,1)), flags=cv2.SOLVEPNP_ITERATIVE)


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
lh1_ax.scatter(pts_lighthouse_A[:,0], pts_lighthouse_A[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH1 - original")
lh2_ax.scatter(pts_lighthouse_B[:,0], pts_lighthouse_B[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH2 - original")
# Plot the cv2.projectpoints points
lh1_ax.scatter(proj_pts_A[:,0], proj_pts_A[:,1], edgecolor='red', facecolor='red', alpha=0.3, lw=1, label="LH1 - projectPoint")
lh2_ax.scatter(proj_pts_B[:,0], proj_pts_B[:,1], edgecolor='red', facecolor='red', alpha=0.3, lw=1, label="LH2 - projectPoint")

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
# plt.show()

######################################### 3D Plotting #######################################  

## Plot the two coordinate systems
#  x is blue, y is red, z is green

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_proj_type('ortho')

# Plot the lighthouse orientation
arrow = np.array([0,0,1]).reshape((-1,1))
ax2.quiver(lha_t[0],lha_t[1],lha_t[2], (lha_R @ arrow)[0], (lha_R @ arrow)[1], (lha_R @ arrow)[2], length=0.4, color='xkcd:red')
ax2.quiver(lhb_t[0],lhb_t[1],lhb_t[2], (lhb_R @ arrow)[0], (lhb_R @ arrow)[1], (lhb_R @ arrow)[2], length=0.4, color='xkcd:red')

ax2.scatter(points[:,0],points[:,1],points[:,2], color='xkcd:blue', label='ground truth', s=50)
ax2.scatter(lha_t[0],lha_t[1],lha_t[2], color='xkcd:red', label='LH1', s=50)
ax2.scatter(lhb_t[0],lhb_t[1],lhb_t[2], color='xkcd:red', label='LH2', s=50)

ax2.axis('equal')
ax2.legend()

plt.show()