# define the model_fn for baseline model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import input_fn

def build_graph(images, labels, params):
    """Build the computation graph, when we have a new model, simply modify this function.
    
    Returns:
        raw_probs: multiclass prediction prob
        loss: loss
    """
    num_classes=params['num_classes']
    learning_rate=params['learning_rate']
    module_trainable=params['module_trainable']
    
    # Input layer.
    module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1",
                       trainable=module_trainable)
    features = module(images)  # (batch_size, D)

    #     # RNN layer.
#     gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
#     outputs, _ = tf.nn.dynamic_rnn(cell=gru_cell, inputs=)

    # Create multi-head sigmoid outputs.
    # It measures the independent probability of a class showing in the image.
    raw_logits = tf.contrib.layers.fully_connected(
        inputs=features,
        num_outputs=num_classes,
        activation_fn=None)  # (batch_size, num_classes)
    
    raw_probs = tf.sigmoid(raw_logits)  # (batch_size, num_classes)
    
    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels,
        logits=raw_logits)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    return raw_probs, loss, optimizer


def model_fn(features, labels, mode, params):
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
    thresholds=params['eval_thresholds']
    
    # build graph, get raw_prob logits
    raw_probs, loss, optimizer = build_graph(features, labels, params)

    predictions = {
        'probs': raw_probs,
    }
    
    # PREDICT mode.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Add evalutaion metrics (for EVAL mode).
    auc = tf.metrics.auc(
        labels=labels,
        predictions=predictions['probs'])
    
    eval_metric_ops = {'auc': auc}
    precisions, precisions_op = tf.metrics.precision_at_thresholds(labels, predictions['probs'], thresholds)
    recalls, recalls_op = tf.metrics.recall_at_thresholds(labels, predictions['probs'], thresholds)
    
    summary_p=[]
    summary_r=[]
    summary_f1=[]
    for i in range(len(thresholds)):
        tag=str(thresholds[i])
        p_tag='precisions_'+tag
        r_tag='recalls_'+tag
        f1_tag='f1_'+tag
        
        eval_metric_ops[p_tag]=(precisions[i], precisions_op[i])
        eval_metric_ops[r_tag]=(recalls[i], recalls_op[i])
        eval_metric_ops[f1_tag]=(2*precisions[i]*recalls[i]/(precisions[i]+recalls[i]+1e-10),
                                 tf.group(precisions_op[i], recalls_op[i]))
        
        summary_p.append(tf.summary.scalar(p_tag, precisions_op[i], family='precisions'))
        summary_r.append(tf.summary.scalar(r_tag, recalls_op[i], family='recalls'))
        summary_f1.append(tf.summary.scalar(f1_tag, eval_metric_ops[f1_tag][0], family='f1_scores'))
    
    tf.summary.scalar('auc', auc[1])
    tf.summary.merge_all()

    # TRAIN mode.
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          train_op=train_op)
    # EVAL mode.
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metric_ops)


