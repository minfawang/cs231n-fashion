"""
Example command:
python ensemble.py --pred_threshold=0.2 --ensemble_dir=/home/shared/ensemble_dir --ensemble_output=/home/shared/ensemble_output.csv
"""

import importlib
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import re
import pandas as pd
import csv

NUM_TEST = 39706
NUM_VALID = 9897


tf.app.flags.DEFINE_string("ensemble_dir", '', "Ensemble directory.")
tf.app.flags.DEFINE_string("ensemble_output", '', "Output path.")
tf.app.flags.DEFINE_string("pred_threshold", '', "the threshold for prediction")
tf.app.flags.DEFINE_string("output_type", 'prob', "[prob|pred]")
tf.app.flags.DEFINE_string("mode", 'test', "[test|validate]")
FLAGS = tf.app.flags.FLAGS


def write_predictions(probs, output_file):
    """
    Inputs:
        probs: np.array with shape [num_examples, num_classes].
        output_file: str.
    
    Globals:
        FLAGS.test_prediction
        FLAGS.pred_threshold
    """
    total_count = NUM_TEST if FLAGS.mode == 'test' else NUM_VALID
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

        with tqdm(total=total_count) as progress_bar:
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


def write_probs(probs, output_file):
    total_count = NUM_TEST if FLAGS.mode == 'test' else NUM_VALID
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['image_id', 'label_prob'])
        
        with tqdm(total=total_count) as progress_bar:
            for img_id, prob in enumerate(probs, start=1):
                prob_str = ' '.join(["%.2f"%(p) for p in prob])
                writer.writerow([img_id, prob_str])

                progress_bar.update(1)

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


if __name__ == '__main__':
    assert FLAGS.pred_threshold or FLAGS.output_type == 'prob', 'FLAGS.pred_threshold is required.'
    assert FLAGS.ensemble_dir, 'FLAGS.ensemble_dir is required.'
    assert FLAGS.ensemble_output, 'FLAGS.ensemble_output is required.'
    assert FLAGS.mode in ['test', 'validate']
    assert FLAGS.output_type in ['prob', 'pred']
    
    ensemble_dir = FLAGS.ensemble_dir
    output_file = FLAGS.ensemble_output
    
    prob_files = os.listdir(ensemble_dir)
    agg_probs = None

    for prob_file in prob_files:
        if not prob_file.endswith('.csv'):
            print('[WARNING] Skip non-csv file: {}'.format(prob_file))
            continue
            
        prob_file_path = os.path.join(ensemble_dir, prob_file)
        print("Reading data from: {}".format(prob_file_path))

        probs = pd.read_csv(prob_file_path)['label_prob'].map(lambda x: np.array([float(v) for v in x.split(' ')])).values
        probs = np.array(probs.tolist())
        if agg_probs is None:
            agg_probs = probs
        else:
            agg_probs += probs

    agg_probs /= len(prob_files)
    
    write_fn = write_probs if FLAGS.output_type == 'prob' else write_predictions
    write_fn(agg_probs, output_file)