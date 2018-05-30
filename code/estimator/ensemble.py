"""
Example command:
python code/estimator/model_runner.py --mode=test --ensemble_model_dir=/home/shared/cs231n-fashion/model_dir/ --pred_threshold=0.8 --gpu_id=1 --test_prediction=/home/minfa/test_ensemble_prediction.csv --batch_size=256
"""

import importlib
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import re
import pandas as pd

FLAGS = tf.app.flags.FLAGS
NUM_TEST = 39706


def write_predictions(probs, output_file):
    """
    Inputs:
        probs: np.array with shape [num_examples, num_classes].
        output_file: str.
    
    Globals:
        FLAGS.test_prediction
        FLAGS.pred_threshold
    """
    print("Saving test data to: ", output_file)
    with open(output_file, "w") as f:
        f.write("image_id,label_id\n")
        img_id = 1

        # deal with unified threshold or per class thresholding
        thresholds = []
        if re.match("^\d+?\.\d+?$", FLAGS.pred_threshold) is None:
            print("Use per class thresholding.")
            thresholds = pd.read_csv(FLAGS.pred_threshold)['thresholds'].values
        else:
            th = float(FLAGS.pred_threshold)
            thresholds = [th for i in range(228)]

        with tqdm(total=NUM_TEST) as progress_bar:
            for prob in probs:
                labels=" ".join([
                    str(i+1)
                    for i in range(len(prob))
                    if prob[i] >= thresholds[i]
                ])
                f.write("%d,%s\n"%(img_id, labels))
                img_id += 1
                progress_bar.update(1)
        print("Processed %d examples. Good Luck! :)"%(img_id))
    

def predict(ensemble_label_to_model_meta, run_config, params, test_input_fn):
    """
    Inputs:
        ensemble_label_to_model_meta: dict. KV is in the following format:
            'label': ('model_file', 'exp_name', weight).
            The model_file should export model_fn(features, labels, mode, params).
            The exp_name should be the folder name that stores the weights of the model.
    
    Globals:
        FLAGS.model_dir
        
    Returns:
        probs produced by the ensemble model.
    """
    # Validate existence of exp_names.
    
    total_weight = 0.0
    agg_probs = None

    for label in ensemble_label_to_model_meta:
        model_filename, exp_name, weight = ensemble_label_to_model_meta[label]
        weight = np.float32(weight)
        model_dir = os.path.join(FLAGS.ensemble_model_dir, exp_name)

        # Create model from model_file.
        cur_file_dir = os.path.dirname(os.path.realpath(__file__))
        model_file = os.path.join(cur_file_dir, "{}.py".format(model_filename))
        model_fn = importlib.import_module(model_filename).model_fn
        print('Load model_fn from: {}'.format(model_file))

        # Initialize classifier with weights from exp_name.
        classifier = tf.estimator.Estimator(
            config=run_config,
            model_fn=model_fn,
            model_dir=model_dir,
            params=params
        )

        # Make predictions.
        probs = np.array([
            pred['probs']
            for pred in classifier.predict(test_input_fn)
        ])
        
        # Aggregate weighted probs.
        if agg_probs is None:
            agg_probs = weight * probs
        else:
            agg_probs += weight * probs
        total_weight += weight
    
    agg_probs /= total_weight
    return agg_probs