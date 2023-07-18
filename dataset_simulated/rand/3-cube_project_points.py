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


#############################################################################
###                    Project points with OpenCV                         ###
#############################################################################
rvec, _ = cv2.Rodrigues(lha_R.T)
tvec = -lha_R.T @ lha_t
pts_a, _ = cv2.projectPoints(points, rvec, tvec, np.eye(3), np.zeros((5,1)))
pts_a = pts_a.reshape((-1,2))

rvec, _ = cv2.Rodrigues(lhb_R.T)
tvec = -lhb_R.T @ lhb_t
pts_b, _ = cv2.projectPoints(points, rvec, tvec, np.eye(3), np.zeros((5,1)))
pts_b = pts_b.reshape((-1,2))


#############################################################################
###                          3D SCENE SOLVING                             ###
#############################################################################
K = np.eye(3, dtype=float)

# Obtain translation and rotation vectors.
E, mask  =  cv2.findEssentialMat(pts_a, pts_b, K, cv2.RANSAC, 0.99999, 0.1)
_, R, t, mask = cv2.recoverPose(E, pts_a, pts_b)

# Calculate Projection matrix, from the canonical camera pose. (0,0,0) with no rotation
# P_a = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
# P_b = np.hstack((R, t))

# Calculate the projection using the real Camera A pose.
Pw_a = np.vstack([np.hstack([lha_R.T, -lha_R.T @ lha_t.reshape((-1,1))]), [0,0,0,1]])   # Pw_a - transformation matrix: World -> Camera A
Pa_b = np.vstack([np.hstack([R, t]), [0,0,0,1]])                                        # Pa_b - transformation matrix: Camera A -> Camera B
Pw_b = Pa_b @ Pw_a                                                                      # Pw_b - transformation matrix: World -> Camera B

# # Remove the last row to get the projection matrix
P_b = Pw_b[0:3,:]
P_a = Pw_a[0:3,:]

point3D = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T).T
point3D = point3D[:, :3] / point3D[:, 3:4]

# Scale the size of the dataset to the real size, for comparison
# scale = np.linalg.norm(lhb_t - lha_t) / np.linalg.norm( P_b[:,3] - P_a[:,3]) 
# P_b[:,3] *= scale
# point3D *= scale


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
lh1_ax.scatter(pts_a[:,0], pts_a[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH1")
lh2_ax.scatter(pts_b[:,0], pts_b[:,1], edgecolor='blue', facecolor='blue', alpha=0.5, lw=1, label="LH2")

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

######################################### 3D Plotting TAN projection #######################################  

## Plot the two coordinate systems
#  x is blue, y is red, z is green

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_proj_type('ortho')

# Plot the lighthouse orientation
arrow = np.array([0,0,1]).reshape((-1,1))
ax2.quiver(lha_t[0],lha_t[1],lha_t[2], (lha_R @ arrow)[0], (lha_R @ arrow)[1], (lha_R @ arrow)[2], length=0.2, color='xkcd:red' )
ax2.quiver(lhb_t[0],lhb_t[1],lhb_t[2], (lhb_R @ arrow)[0], (lhb_R @ arrow)[1], (lhb_R @ arrow)[2], length=0.2, color='xkcd:red' )
#
ax2.scatter(points[:,0],points[:,1],points[:,2], color='xkcd:blue', label='ground truth')
ax2.scatter(lha_t[0],lha_t[1],lha_t[2], color='xkcd:red', label='LH1')
ax2.scatter(lhb_t[0],lhb_t[1],lhb_t[2], color='xkcd:red', label='LH2')


# Plot triangulated points TAN projection
# Calculate the Camera location to plot them
# t_1 = np.array([0,0,0])
# R_1 = np.eye(3)
# Get the Camera A -> world transformation by inverting P_a == Pw_a
t_1 = - P_a[:,0:3].T @ P_a[:,3]
R_1 = P_a[:,0:3].T
# second camera
# R_2 = R.T
# t_2 = - R.T @ t
R_2 = P_b[:,0:3].T
t_2 = - P_b[:,0:3].T @ P_b[:,3]

ax2.quiver(t_1[0],t_1[1],t_1[2], (R_1 @ arrow)[0], (R_1 @ arrow)[1], (R_1 @ arrow)[2], length=0.2, color='xkcd:orange' )
ax2.quiver(t_2[0],t_2[1],t_2[2], (R_2 @ arrow)[0], (R_2 @ arrow)[1], (R_2 @ arrow)[2], length=0.2, color='xkcd:orange' )
ax2.scatter(point3D[:,0],point3D[:,1],point3D[:,2], color='xkcd:green', alpha=0.5, label='triangulated')
ax2.scatter(t_1[0],t_1[1],t_1[2], color='xkcd:orange', label='triang LH1')
ax2.scatter(t_2[0],t_2[1],t_2[2], color='xkcd:orange', label='triang LH2')

ax2.axis('equal')
ax2.legend()

ax2.set_title('Tangent projection')
ax2.set_xlabel('X [mm]')
ax2.set_ylabel('Y [mm]')
ax2.set_zlabel('Z [mm]')


plt.show()
a=1