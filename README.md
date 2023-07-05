# LH2-3D-accuracy-experiment
Repository to hold the dataset and processing algorithm for the 3D experiments of the LH2 paper.


## Getting Started

### Requirements
To run this code, your Python installation must have the following modules installed.
- Numpy
- Pandas
- Matplotlib
- OpenCV

### Selecting the dataset to use

[line 16](https://github.com/SaidAlvarado/LH2-3D-accuracy-experiment/blob/ec05885f698933a4b7f4334eb4ed30c83dc28873/lh2_3d_analysis.py#L16) of the lh2_3d_analysis allows you to select which dataset to process.
```python
# data_file = 'dataset_experimental/data_1point.csv'
# data_file = 'dataset_experimental/data_all.csv'
data_file = 'dataset_simulated/data.csv'
```

there are 3 datasets available
- `dataset_simulated/data.csv` Is a simulated dataset created by the `dataset_simulated/simulate_dataset.py` script.
- `dataset_experimental/data_all.csv` Is the experimental dataset. It uses every single data point captured in the 3D printer eperiment. This file was generated by the `dataset_experimental/parse_dataset.py` script.
- `dataset_experimental/data_1point.csv` Is the experimental dataset. It has only 1 data point per grid position. This file was generated by the `dataset_experimental/parse_dataset.py` script.

### Run the code

1. Install the required dependencies.
2. Select the dataset you want to process.
3. Run the following command from the root folder.
`$ python lh2_3d_analysis.py`

A few plots should appear with the results of the analysis


## Repository Organization

The repository has the following files:

- `3d_printer_gcode`
    - `LH2_4cm_calibration_grid.gcode` 3D printer movement pattern that generates the dataset.
- `dataset_experimental`
    - `raw_data` 
        - `pydotbot.log` Raw logfile generated from [PyDotBot](https://github.com/DotBots/PyDotBot) with the LH2 data.
        - `timestamps.json` This file has timestamps that correlate the raw data from the above logfile, to the groundtruth positions of the LH2 receiver.
    - `calibrate_lh2.py` WIP.
    - `parse_dataset.py` Script that parses the raw log file and creates a distilled CSV of data, ready to be processed.
    - `data_all.csv` Dataset with all the data captured in the experiment
    - `data_1point.csv` Simplfied dataset with a single data-point per grid position.
- `dataset_simulated`
    - `simulate_dataset.py` Script that creates artificial dataset based on a toy configuration of basestations and LH2 receivers. This is for testing against a known dataset.
    - `data.csv` Simulated dataset.
    - `basestation.json` This has the pose of all the simulated basestations (generated by `simulate_dataset.py`)
- `lh2_3d_analysis.py` Main file. Process the datasets and generates the plots.
