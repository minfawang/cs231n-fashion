import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import Lambda
from sklearn.metrics import precision_recall_fscore_support

class FMetrics:
    def __init__(self, thresholds=None):
        if thresholds is None:
            thresholds = [i/10.0 for i in range(1, 10)]
        self.thresholds = thresholds
        self.precision_metric_fns=[]
        self.recall_metric_fns=[]
        self.f1_metric_fns=[]
    
    def __metric_fn(self, y_true, y_pred, threshold):
        y_true = tf.cast(y_true, "int32")
        y_pred = tf.minimum(tf.maximum(0, y_pred+0.5-threshold), 1)
        y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
        y_correct = y_true * y_pred
        sum_true = tf.reduce_sum(y_true)
        sum_pred = tf.reduce_sum(y_pred)
        sum_correct = tf.reduce_sum(y_correct)
        precision = sum_correct / sum_pred
        recall = sum_correct / sum_true
        f_score = 2 * precision * recall / (precision + recall)
        f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
        return f_score, precision, recall
    
    def get_fscores(self):
        """
        The lambda function won't work .. so i hard code all metrics.
        """
#         f1=Lambda(lambda x,y: self.__metric_fn(x,y,0.1)[0])
#         f2=Lambda(lambda x,y: self.__metric_fn(x,y,0.2)[0])
#         f3=Lambda(lambda x,y: self.__metric_fn(x,y,0.3)[0])
#         f4=Lambda(lambda x,y: self.__metric_fn(x,y,0.4)[0])
#         f5=Lambda(lambda x,y: self.__metric_fn(x,y,0.5)[0])
#         f6=Lambda(lambda x,y: self.__metric_fn(x,y,0.6)[0])
#         f7=Lambda(lambda x,y: self.__metric_fn(x,y,0.7)[0])
#         f8=Lambda(lambda x,y: self.__metric_fn(x,y,0.8)[0])                        
#         f9=Lambda(lambda x,y: self.__metric_fn(x,y,0.9)[0])
        
#         return [f1, f2, f3, f4, f5, f6, f7, f8, f9]
        return [self.fscore2, self.fscore5, self.fscore7]

    def get_precisions(self):
        f1=lambda x,y: self.__metric_fn(x,y,0.1)[1]
        f2=lambda x,y: self.__metric_fn(x,y,0.2)[1]
        f3=lambda x,y: self.__metric_fn(x,y,0.3)[1]
        f4=lambda x,y: self.__metric_fn(x,y,0.4)[1]
        f5=lambda x,y: self.__metric_fn(x,y,0.5)[1]
        f6=lambda x,y: self.__metric_fn(x,y,0.6)[1]
        f7=lambda x,y: self.__metric_fn(x,y,0.7)[1]
        f8=lambda x,y: self.__metric_fn(x,y,0.8)[1]                         
        f9=lambda x,y: self.__metric_fn(x,y,0.9)[1]
        
        return [f1, f2, f3, f4, f5, f6, f7, f8, f9]
                         
    def get_recalls(self):
        f1=lambda x,y: self.__metric_fn(x,y,0.1)[2]
        f2=lambda x,y: self.__metric_fn(x,y,0.2)[2]
        f3=lambda x,y: self.__metric_fn(x,y,0.3)[2]
        f4=lambda x,y: self.__metric_fn(x,y,0.4)[2]
        f5=lambda x,y: self.__metric_fn(x,y,0.5)[2]
        f6=lambda x,y: self.__metric_fn(x,y,0.6)[2]
        f7=lambda x,y: self.__metric_fn(x,y,0.7)[2]
        f8=lambda x,y: self.__metric_fn(x,y,0.8)[2]                         
        f9=lambda x,y: self.__metric_fn(x,y,0.9)[2]
        
        return [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    
    def fscore2(self, y_true, y_pred):
        y_true = tf.cast(y_true, "int32")
        y_pred = tf.minimum(tf.maximum(0.0, y_pred+0.5-0.2), 1.0)
        y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
        y_correct = y_true * y_pred
        sum_true = tf.reduce_sum(y_true)
        sum_pred = tf.reduce_sum(y_pred)
        sum_correct = tf.reduce_sum(y_correct)
        precision = sum_correct / sum_pred
        recall = sum_correct / sum_true
        f_score = 2 * precision * recall / (precision + recall)
        f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
        return f_score
    def fscore3(self, y_true, y_pred):
        y_true = tf.cast(y_true, "int32")
        y_pred = tf.minimum(tf.maximum(0.0, y_pred+0.5-0.3), 1.0)
        y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
        y_correct = y_true * y_pred
        sum_true = tf.reduce_sum(y_true)
        sum_pred = tf.reduce_sum(y_pred)
        sum_correct = tf.reduce_sum(y_correct)
        precision = sum_correct / sum_pred
        recall = sum_correct / sum_true
        f_score = 2 * precision * recall / (precision + recall)
        f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
        return f_score
    def fscore5(self, y_true, y_pred):
        y_true = tf.cast(y_true, "int32")
        y_pred = tf.minimum(tf.maximum(0.0, y_pred+0.5-0.5), 1.0)
        y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
        y_correct = y_true * y_pred
        sum_true = tf.reduce_sum(y_true)
        sum_pred = tf.reduce_sum(y_pred)
        sum_correct = tf.reduce_sum(y_correct)
        precision = sum_correct / sum_pred
        recall = sum_correct / sum_true
        f_score = 2 * precision * recall / (precision + recall)
        f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
        return f_score
    def fscore7(self, y_true, y_pred):
        y_true = tf.cast(y_true, "int32")
        y_pred = tf.minimum(tf.maximum(0.0, y_pred+0.5-0.7), 1.0)
        y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
        y_correct = y_true * y_pred
        sum_true = tf.reduce_sum(y_true)
        sum_pred = tf.reduce_sum(y_pred)
        sum_correct = tf.reduce_sum(y_correct)
        precision = sum_correct / sum_pred
        recall = sum_correct / sum_true
        f_score = 2 * precision * recall / (precision + recall)
        f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
        return f_score
  

class FMetricsCallback(Callback):
    def on_train_begin(self, logs={}):
        self.thresholds = [i/10.0 for i in range(1, 10)]
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        print('\n')
        for i, th in enumerate(self.thresholds):
            val_predict = np.asarray(self.model.predict(self.model.validation_data[0])>=th).astype(int)
            val_targ = self.model.validation_data[1]
            p, r, f, _ = precision_recall_fscore_support(val_targ, val_predict)
            self.val_f1s.append(f)
            self.val_recalls.append(r)
            self.val_precisions.append(p)
            print("— val_f1: %f — val_precision: %f — val_recall %f" %(f, p, r))
            return
