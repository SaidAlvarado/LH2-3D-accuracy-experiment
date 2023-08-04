# LH2-3D-accuracy-experiment
Repository to hold the dataset and processing algorithm for the 3D experiments of the LH2 paper.


## Getting Started

### Requirements
To install the required modules, run the following command from the root folder.

`$ pip install -r requirements.txt`

### Selecting which dataset to use

[line 15](https://github.com/SaidAlvarado/LH2-3D-accuracy-experiment/blob/2f47a34116a0649291dec6418c86256989b11059/lh2_3d_analysis.py#L15) of the `lh2_3d_analysis.py` allows you to select which dataset to process. Uncomment the desired dataset.
```python
# data_file = 'dataset_experimental/data_1point.csv'
data_file = 'dataset_experimental/data_all.csv'
```

there are 2 datasets available
- `data_all.csv`  It uses every single data point captured in the 3D printer eperiment.
- `data_1point.csv`  It has only 1 data point per grid position. 

### Run the code

1. Install the required dependencies.
2. Select the dataset you want to process.
3. Run the following command from the root folder.

- `$ python lh2_3d_analysis.py`

A few plots should appear with the results of the analysis

### Change the code

At the end of the `lh2_3d_analysis.py` file, you'll find detailed explanations for each variable used in the script. 
Additionally, the end of the script includes examples demonstrating how to generate NumPy arrays with the projected LH views.
