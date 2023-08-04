# LH2-3D-accuracy-experiment
Repository to hold the dataset and processing algorithm for the 3D experiments of the LH2 paper.


## Getting Started

### Requirements
To install the required modules, run the following command from the root folder.

`$ pip install -r requirements.txt`

### Selecting the dataset to use

[line 15](https://github.com/SaidAlvarado/LH2-3D-accuracy-experiment/blob/ec05885f698933a4b7f4334eb4ed30c83dc28873/lh2_3d_analysis.py#L16) of the lh2_3d_analysis allows you to select which dataset to process. Uncomment the desired dataset.
```python
# data_file = 'dataset_experimental/data_1point.csv'
data_file = 'dataset_experimental/data_all.csv'
```

there are 2 datasets available
- `dataset_experimental/data_all.csv` Is the experimental dataset. It uses every single data point captured in the 3D printer eperiment.
- `dataset_experimental/data_1point.csv` Is the experimental dataset. It has only 1 data point per grid position. 

### Run the code

1. Install the required dependencies.
2. Select the dataset you want to process.
3. Run the following command from the root folder.

- `$ python lh2_3d_analysis.py`

A few plots should appear with the results of the analysis

### Change the code

At the bottom of the `$ python lh2_3d_analysis.py` are instructions on what each variable does.
And there are code examples on how to get numpy arrays with the projected LH views.
