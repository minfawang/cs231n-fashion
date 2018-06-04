# CS231N Final Project

We are using the dataset from the following Kaggle competition: [iMaterialist Challenge (Fashion) at FGVC5](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/).

The project proposal is available at [CS231N_Project_Proposal.pdf](./CS231N_Project_Proposal.pdf).

# Setup

#### Prerequisites

* **Python3**: we use Python version 3+ for this project.
* [Pipenv](https://github.com/pypa/pipenv): Python package manager and virtual environment. Can be installed with command `pip install pipenv`.

#### Initial Setup

At the first time, run the following commands:

```bash
git clone git@github.com:minfawang/cs231n-fashion.git  # Clones repo.
cd cs231n-fashion  # Changes your directory to the root of the repo.
# If you use a conda custom Python binary, then you may use the
# command in the comment below:
# pipenv --python /usr/local/bin/python3 install
pipenv --three install  # Create a virtual env using Python3.

# Enter virtual env.
pipenv shell

# Set up custom python kernel with correct binary and dependency.
# https://stackoverflow.com/a/47296960
python -m ipykernel install --user --name=cs231n-fashion
```

For running the cs231n pre-defined image on VM instance on Google Cloud, you need to also run this comamnd per [instructions from the course page](http://cs231n.github.io/gce-tutorial/):

```bash
/home/shared/setup.sh && source ~/.bashrc
```

#### Download data

First, download the `json` files from the Kaggle [data page](https://www.kaggle.com/c/imaterialist-challenge-fashion-2018/data).

Please download the files into **data/** directory and then unzip all of them. Then download the images using the script below:

```bash
# Change max_download parameters in the file.
python utils/downloader.py
```

#### Each run

Everytime you need to update the project or run the scripts:

```bash
pipenv shell  # Enter the virtual env.
# Make updates.
exit
```


#### Useful commands

##### Training


```bash
# run training.
python code/keras_model_runner.py --mode=train --fine_tune --reg=0.00001 --steps_per_epoch=2000 --batch_size=64 --initial_epoch=0 --model_dir=model_dir/keras_xception/
```

**Additional flags:**

* `--generator_use_weight=1`: Assign per-calss weights in training time.
* `--generator_use_wad=1`: Generate wide-and-deep features.

##### Testing
```bash
# run test, generate submission file. If set pred_threshold to a filename, then use per class threshold.
python code/model_runner.py --mode=test --model_dir=/home/shared/cs231n-fashion/model_dir/baseline2/ --pred_threshold=0.8
```

##### Eval
```bash
# run eval.
python code/model_runner.py --mode=eval --model_dir=/home/shared/cs231n-fashion/model_dir/baseline2/ --eval_thresholds=0.3;0.5;0.7;0.75;0.8;0.85;0.9
```

##### Print debug dump
```bash
# Print debug dump. Check the Threshold Selection part in binbin_playground for reference.
# By default this prints the output of validation set. You can change this behavior in model_runner.py
python code/model_runner.py --mode=debug --model_dir=/home/shared/cs231n-fashion/model_dir/baseline2/ --debug_dump_file=model_dir/baseline2/debug_dump.csv
```

##### Print debug test dump
Similar as above, just replace **debug** with **debug_test**. It could be used to create model ensemble.
```
python code/model_runner.py --mode=debug_test --model_dir=/home/shared/cs231n-fashion/model_dir/baseline2/ --debug_dump_file=model_dir/baseline2/debug_test_dump.csv
```

##### Threshold selection
Check binbin_playground for reference. This could give extra 3% boost for single model.

##### Other useful commands
```bash
# after logging in, run the following command to monitor memory usage
sh /home/binbinx/memusg.sh
```

```bash
# this will download the test_prediction to local
gcloud compute scp binbinx@cs231n-fashion-ssd:/home/shared/cs231n-fashion/submission/test_prediction.csv .
```

#### Model Ensemble

- For each model, run the `debug_test` command and generate a csv file.
- Put all csv files into a single folder.

```bash
python code/ensemble.py --pred_threshold=0.2 --ensemble_dir=/home/shared/ensemble_dir --ensemble_output=/home/shared/ensemble_output.csv --output_type='prob' --mode='validate'
```

#### Scratch Pad
```bash
# Model from scratch:
python code/keras_model_runner.py --mode=train --model_dir=model_dir/keras_xception/retrain/ --drop_out_rate=0.0 --reg=0.00001 --gpu_id=0 --batch_size=32 --steps_per_epoch=1600 --epochs=1000 --fine_tune

# Model from scratch with sample weighting:
python code/keras_model_runner.py --mode=train --model_dir=model_dir/keras_xception/retrain_weight/ --drop_out_rate=0.0 --reg=0.00001 --gpu_id=0 --batch_size=32 --steps_per_epoch=1600 --epochs=1000 --fine_tune --generator_use_weight

```
