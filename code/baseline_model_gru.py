# define the model_fn for baseline model
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import input_fn
import os

def build_graph(images, labels, is_training, params):
    """Build the computation graph, when we have a new model, simply modify this function.
    
    Returns:
        raw_probs: multiclass prediction prob
        loss: loss
    """
    num_classes=params['num_classes']
    learning_rate=params['learning_rate']
    module_trainable=params['module_trainable']
    reg=params['reg']
    
    # Input layer.
    if module_trainable:
        module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1",
                            trainable=True, tags={"train"})
    else:
        module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/1")
    features = module(images)  # (batch_size, D)
    
    hidden_size=1536
    # Re-represent the input feature
    fc1 = tf.contrib.layers.fully_connected(
        inputs=features,
        num_outputs=hidden_size,
        activation_fn=tf.nn.relu)
    
    # Pass through bidirectional GRU layer
    gru_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    gru_fw, gru_bw = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_cell,
                                                     cell_bw=gru_cell,
                                                     inputs=features)
    gru_out = tf.concatenate([gru_fw, gru_bw], axis=1)
    
    # Re-represent
    fc2 = tf.contrib.layers.fully_connected(
        inputs=gru_out,
        num_outputs=tf.shape(gru_out)[1],
        activation_fn=tf.nn.relu)
        
    fc2 = tf.layers.dropout(
        inputs=fc2,
        rate=0.15,
        training=is_training,
        activation_fn=tf.nn.relu)
    
    # Output
    raw_logits = tf.contrib.layers.fully_connected(
        inputs=fc2,
        num_outputs=num_classes,
        activation_fn=None)  # (batch_size, num_classes)
    
    raw_probs = tf.sigmoid(raw_logits)  # (batch_size, num_classes)
    
    loss, optimizer = None, None
    if labels is not None:
        # in test mode, don't build loss graph
        loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=labels,
            logits=raw_logits)
    
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        if module_trainable:
            loss += reg*tf.losses.get_regularization_loss()
    
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
    model_dir=params['model_dir']
    
    # build graph, get raw_prob logits
    raw_probs, loss, optimizer = build_graph(features, labels, mode==tf.estimator.ModeKeys.TRAIN, params)

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
        
        f1=2*precisions[i]*recalls[i]/(precisions[i]+recalls[i]+1e-10)
        summary_p.append(tf.summary.scalar(p_tag, precisions_op[i], family='precisions'))
        summary_r.append(tf.summary.scalar(r_tag, recalls_op[i], family='recalls'))
        summary_f1.append(tf.summary.scalar(f1_tag, f1, family='f1_scores'))
    
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
#         summary_hook=tf.train.SummarySaverHook(save_steps=100,
#                                                output_dir=os.path.join(model_dir,'eval/'), 
#                                                summary_op=sum_all)
#         logging_hook=tf.train.LoggingTensorHook(every_n_iter=100,
#                                                 at_end=True,
#                                                 tensors=f1_value)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
#                                           evaluation_hooks=[summary_hook, logging_hook],
                                          eval_metric_ops=eval_metric_ops)