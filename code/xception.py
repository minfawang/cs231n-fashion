import os
import numpy as np
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras import metrics
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from utils.custom_metrics import FMetrics, FMetricsCallback
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Nadam
from keras import regularizers

MODEL_BEST_NAME = 'top_model_weights.h5'
MODEL_CHECKPOINT_NAME = 'model_weights-{epoch:02d}-{val_acc:.2f}.hdf5'

class KerasXception:
    
    def __init__(self, params):
        self.params=params
        
        # get useful params, keep as private field here.
        self.model_dir = self.params['model_dir']
        self.num_classes = self.params['num_classes']
        self.fine_tune = self.params['fine_tune']
        self.reg = self.params['reg']
        self.drop_out_rate = self.params['drop_out_rate']

        self.model_file = os.path.join(self.model_dir, MODEL_BEST_NAME)
        self.model_checkpoint = os.path.join(self.model_dir, MODEL_CHECKPOINT_NAME)
        
        self.model = self.__build_graph(self.fine_tune)
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        elif os.path.exists(self.model_file):
            # load the model weights
            print ("Load model weights from %s"%(self.model_file))
            self.model.load_weights(self.model_file)
            
        
    def __build_graph(self, enable_fine_tune):
        # create the base pre-trained model
        base_model = Xception(weights='imagenet', include_top=False)

        # add a global spatial max pooling layer
        x = base_model.output #(?,3,3,2048)
        x = GlobalAveragePooling2D()(x) #(?, 2048)

#         # let's add a fully-connected layer
#         x = Dense(2048, activation='relu')(x)
        x = Dropout(rate=self.drop_out_rate)(x)

        # 228 classes 
        predictions = Dense(self.num_classes, activation='sigmoid')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        f_metrics = FMetrics()
        f_scores = f_metrics.get_fscores()
        
        if not enable_fine_tune:
            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False

            # compile the model (should be done *after* setting layers to non-trainable)
            model.compile(optimizer='rmsprop', 
                          loss='binary_crossentropy',
                          metrics=['accuracy']+f_scores)
        else:
            print("@@@@@Fine tune enabled.@@@@@")
            print("Fine tune the last feature flow and the entire exit flow")
            for layer in model.layers: # change from 116 to 36 to ALL
                layer.trainable = True
                layer.kernel_regularizer = regularizers.l2(self.reg)
                
#             for layer in model.layers[36:]:
#                 layer.trainable = True
#                 layer.kernel_regularizer = regularizers.l2(self.reg)
        
            # compile the model with a SGD/momentum optimizer
            # and a very slow learning rate.
            optimizer = Nadam(lr=5e-4, schedule_decay=0.001)
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy']+f_scores)
        
        print (model.summary())
        return model
    

    ###################################################
    def train(self, train_generator, validation_data=None, validation_steps=None,
              epochs=200, steps_per_epoch=5000, shuffle=True, max_queue_size=16, workers=12, initial_epoch=0):
        """
        """
        # Define callbacks list. This should be the same for all models.
        fmetric_callback = FMetricsCallback()
        callbacks_list = [
            # Save each model that improves validation accuracy
            ModelCheckpoint(self.model_checkpoint, monitor='val_acc', verbose=1, save_best_only=True),
            # Save model with best fscore5
            ModelCheckpoint(self.model_file, monitor='val_fscore5', verbose=1, mode='max', save_best_only=True),
            # EarlyStopping(monitor='val_acc', patience=5, verbose=1),
            TensorBoard(log_dir=self.model_dir, 
                        histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, 
                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        ]

        self.model.fit_generator(train_generator,
                                 use_multiprocessing=True,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 shuffle=shuffle,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 validation_data=validation_data,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks_list,
                                 initial_epoch=initial_epoch)
    
    def eval(self, eval_generator, steps=None, max_queue_size=16, workers=12, use_multiprocessing=True):
        self.model.evaluate_generator(eval_generator, steps=steps, 
                                      max_queue_size=max_queue_size, workers=workers, 
                                      use_multiprocessing=use_multiprocessing, verbose=1)
    
    def predict(self, test_generator, steps=None, max_queue_size=16):
        # Set multiprocessing to be false and worker to 1 to keep output order managed.
        return self.model.predict_generator(test_generator, steps=steps, workers=1,
                                            max_queue_size=max_queue_size, use_multiprocessing=False, verbose=1)
