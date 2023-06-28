# LH2-3D-accuracy-experiment
 Repository to hold the dataset and processing algorithm for the 3D experiments of the LH2 paper

## Repository Organization

### 3d printer pattern
- Here is the gcode that you give to the 3d printer to move in the pattern used for the experiment

### experiment data
- `pydotbot.log`   - This is the raw data comming from the Pydotbot-controller log files.
- `timestamp.json` - This file holds which timestamps correspond to  which real positions of the LH2 receiver.
- `results_plotter.py` - main script, processes `pydotbot.log` and plots the triangulated 3D plot of the dataset.

### simulated data
- `sim.py` - this file simulates LH2 data and attemps to do a triangulation, using openCV functions
