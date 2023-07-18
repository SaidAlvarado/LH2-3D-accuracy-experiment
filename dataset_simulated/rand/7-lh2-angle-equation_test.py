from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Plot the results
fig = plt.figure(layout="constrained")
gs = GridSpec(3, 6, figure = fig)
azi_ax = fig.add_subplot(gs[:, 0:3], projection='3d')
elev_ax    = fig.add_subplot(gs[:, 3:6], projection='3d')
axs = (elev_ax, azi_ax)


# Grab some test data.
a1 = np.linspace(0,np.pi*2, 200)
a2 = np.linspace(0,np.pi*2, 200)
a1, a2 = np.meshgrid(a1, a2)
azimuth = np.rad2deg(  (a1+a2)/2  )
elevation = np.rad2deg( np.pi/2 -  np.arctan2(np.sin(a2/2-a1/2-60*np.pi/180),np.tan(np.pi/6))  )

a1 = np.rad2deg(a1)
a2 = np.rad2deg(a2)

# Plot a basic wireframe.
azi_ax.plot_wireframe(a1, a2, azimuth, rstride=10, cstride=10)
elev_ax.plot_wireframe(a1, a2, elevation, rstride=10, cstride=10)

for ax in axs:
    ax.grid()
    ax.legend()
    ax.axis('equal')
    ax.set_xlabel('A1 [deg]')
    ax.set_ylabel('A2 [deg]')
    ax.set_proj_type('ortho')

azi_ax.set_zlabel('Azimuth [deg]')
elev_ax.set_zlabel('Elevation [deg]')
azi_ax.set_title('Azimuth')
elev_ax.set_title('Elevation')

plt.show()
b=1