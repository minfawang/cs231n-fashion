import os
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras import metrics
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

MODEL_FILE_NAME = 'top_model_weights.h5'

class KerasXception:
    
    def __init__(self, params):
        self.params=params
        
        # get useful params, keep as private field here.
        self.model_dir = self.params['model_dir']
        self.model_file = os.path.join(self.model_dir, MODEL_FILE_NAME)
        self.num_class = self.params['num_class']
        self.fine_tune = self.params['fine_tune']
        
        self.model = build_graph(self.fine_tune)
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        else if os.path.exists(self.model_file):
            # load the model weights
            print ("Load model weights from %d"%(self.model_file))
            self.model.load_weights(self.model_file)
            
        
    def __build_graph(self, enable_fine_tune):
        # create the base pre-trained model
        base_model = Xception(weights='imagenet', include_top=False)

        # add a global spatial max pooling layer
        x = base_model.output #(?,3,3,2048)
        x = GlobalAveragePooling2D()(x) #(?, 2048)

        # let's add a fully-connected layer
        x = Dense(2048, activation='relu')(x)
        # 228 classes 
        predictions = Dense(self.num_class, activation='sigmoid')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        
        if not enable_fine_tune:
            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional InceptionV3 layers
            for layer in base_model.layers:
                layer.trainable = False

            # compile the model (should be done *after* setting layers to non-trainable)
            model.compile(optimizer='rmsprop', 
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            # TODO(how many layers to fine tune?)
            for layer in self.model.layers[100:]:
            layer.trainable = True
        
        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(optimizer='nadam',
                      learning_rate=1e-4,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        print (model.summary())
    

    ###################################################
    def train(train_generator, validation_data=None, validation_steps=None,
              epochs=200, steps_per_epoch=5000, shuffle=True, max_queue_size=512, workers=12):
        """
        """
        # Define callbacks list. This should be the same for all models.
        callbacks_list = [
            ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
            EarlyStopping(monitor='val_acc', patience=5, verbose=1),
            TensorBoard(log_dir=self.model_dir, 
                        histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, 
                        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        ]

        model.fit_generator(train_generator,
                            use_multiprocessing=True,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            shuffle=shuffle,
                            max_queue_size=max_queue_size,
                            worker=worker,
                            validation_data=validation_generator,
                            validation_steps=validation_steps,
                            callbacks=callbacks_list)
    
    def eval(eval_generator, steps=None, max_queue_size=512, workers=12, use_multiprocessing=True):
        model.evaluate_generator(eval_generator, steps=steps, 
                                 max_queue_size=max_queue_size, workers=workers, 
                                 use_multiprocessing=use_multiprocessing, verbose=1)
    
    def predict(test_generator, steps=None, max_queue_size=512):
        # Set multiprocessing to be false and worker to 1 to keep output order managed.
        return model.predict_generator(test_generator, steps=steps, workers=1,
                                       max_queue_size=max_queue_size, use_multiprocessing=False, verbose=1)
