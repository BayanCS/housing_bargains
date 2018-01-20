# Description
A command-line application that lets one train a model on the King County House Sales dataset,
and obtain predictions from it.

Code can be found under `src`, inclunding the script using for training, `src/train.py`, and the one used for
obtaining predictions `src/predict.py`.
In addition to that, there is a IPython Notebook, `housing_bargains.ipynb`, which was as a development sandbox.

The data used for the project can be found under the `data` directory.
The complete dataset is contained in the `data/kc_house_data.csv` file.
While the `train_data.csv` and `test_data.csv`, contain the train/test split data for use by the command-line 
scripts.

# Instructions
In order to train a model, use `src/train.py`. Example:

    src/train.py --csv_file=data/train_data.csv --model_path=model.pkl --param-search=True


For obtaining predictions, `src/predict.py` should be used. Example:

    src/predict.py --csv_file=data/test_data.csv --model_path=model.pkl --index=3

For more detailed information either of the two scripts can be executed with the `--help` flag.

# Software Requirements
* Python 2.7 or higher, or 3.3 or higher
* scikit-learn>=0.19.1
* numpy>=1.14.0
* scipy>=1.0.0
* pandas>=0.22.0

The required packages can be easily installed through pip by running `pip -r src/requrirements.txt`.
