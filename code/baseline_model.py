# define the model_fn for baseline model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import input_fn

num_classes=228
learning_rate=3e-4

def model_fn(features, labels, mode):
    """Model function for fashion classes predictions.
    
    Inputs:
        features: dict.
            Required key: "image". The value needs to have shape (batch_size, 299, 299, 3),
                and each value needs to be in range [0, 1].
        labels: shape (batch_size, num_classes)
        mode: tf.estimator.ModeKeys.(PREDICT|TRAIN|EVAL)
    
    Returns:
        estimator_spec.
    """
    # class_prob > threshold will be outputted.
    threshold = 0.5
    
    # Input layer.
    images = features
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1")
    features = module(images)  # (batch_size, D)
    
    # Create multi-head sigmoid outputs.
    # It measures the independent probability of a class showing in the image.
    raw_logits = tf.contrib.layers.fully_connected(
        inputs=features,
        num_outputs=num_classes,
        activation_fn=None)  # (batch_size, num_classes)
    
    raw_probs = tf.sigmoid(raw_logits)  # (batch_size, num_classes)
    
#     # RNN layer.
#     gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
#     outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=)
    
    predictions = {
        'pred_3': (raw_probs > 0.3),
        'pred_5': (raw_probs > 0.5),
        'pred_7': (raw_probs > 0.7),
        'probs': raw_probs,
    }
    
    # PREDICT mode.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate loss (for both TRAIN and EVAL mode).
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels,
        logits=raw_logits)
    
    # Add evalutaion metrics (for EVAL mode).
    auc = tf.metrics.auc(
        labels=labels,
        predictions=predictions['probs'])
    
    precisions_3 = tf.metrics.precision(
        labels=labels, predictions=predictions['pred_3'])
    recalls_3 = tf.metrics.recall(
        labels=labels, predictions=predictions['pred_3'])
    mean_f1_3 = (2*precisions_3[1]*recalls_3[1]/(precisions_3[1]+recalls_3[1]), precisions_3[1])    
    
    precisions_5 = tf.metrics.precision(
        labels=labels, predictions=predictions['pred_5'])
    recalls_5 = tf.metrics.recall(
        labels=labels, predictions=predictions['pred_5'])   
    mean_f1_5 = (2*precisions_5[1]*recalls_5[1]/(precisions_5[1]+recalls_5[1]), precisions_5[1])
    
    precisions_7 = tf.metrics.precision(
        labels=labels, predictions=predictions['pred_7'])
    recalls_7 = tf.metrics.recall(
        labels=labels, predictions=predictions['pred_7'])  
    mean_f1_7 = (2*precisions_7[1]*recalls_7[1]/(precisions_7[1]+recalls_7[1]), precisions_7[1]) 
    
    eval_metric_ops = {
        'precisions_0.3': precisions_3,
        'recalls_0.3': recalls_3,
        'mean_f1_0.3': mean_f1_3,
        'precisions_0.5': precisions_5,
        'recalls_0.5': recalls_5,
        'mean_f1_0.5': mean_f1_5,
        'precisions_0.7': precisions_7,
        'recalls_0.7': recalls_7,
        'mean_f1_0.7': mean_f1_7,
        'auc': auc,
    }

    # the following is used or metric loging
    #auc_log = tf.identity(auc[0], name='auc')
    #f1_log_3 = tf.identity(mean_f1_3[0], name='f1_3')
    #f1_log_5 = tf.identity(mean_f1_5[0], name='f1_5')
    #f1_log_7 = tf.identity(mean_f1_7[0], name='f1_7')

    tf.summary.scalar('precisions_0.3', precisions_3[1])
    tf.summary.scalar('precisions_0.5', precisions_5[1])
    tf.summary.scalar('precisions_0.7', precisions_7[1])
    tf.summary.scalar('recalls_0.3', recalls_3[1])
    tf.summary.scalar('recalls_0.5', recalls_5[1])
    tf.summary.scalar('recalls_0.7', recalls_7[1])
    tf.summary.scalar('f1_0.3', mean_f1_3[0])
    tf.summary.scalar('f1_0.5', mean_f1_5[0])
    tf.summary.scalar('f1_0.7', mean_f1_7[0])

    # TRAIN mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)
      
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)


