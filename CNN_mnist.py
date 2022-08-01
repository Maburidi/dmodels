import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
from tensorflow.keras import models
import argparse



class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    '''
    Halts the training after reaching 60 percent accuracy

    Args:
      epoch (integer) - index of epoch (required but unused in the function definition below)
      logs (dict) - metric results from the training epoch
    '''

    # Check accuracy
    if(logs.get('loss') < 0.4):

      # Stop if threshold is met
      print("\nLoss is lower than 0.4 so cancelling training!")
      self.model.stop_training = True

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build 5 layers convolutional neural network" +
                                     "predict MNIST data")
    parser.add_argument('--vis', type=int, help='Plot figures 1 or 0',
                        default=0)
    args = parser.parse_args()

        
    ####################### Load Data and get it ready ###################
 
    # Load the Fashion MNIST dataset
    
    fmnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

    # Normalize the pixel values
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    
    training_images = training_images.reshape(-1, 28, 28, 1)
    test_images = test_images.reshape(-1, 28, 28, 1)
    
    #print(np.shape(training_images[0]))
    #plt.imshow(training_images[0])
    
    
    #============================ Build the Model=================================
    
    # Instantiate class
    callbacks = myCallback()

    
    # build the layers 
    model = tf.keras.models.Sequential([
        # Add convolutions and max pooling
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),     #64 layers of convolutions  
        tf.keras.layers.MaxPooling2D(2, 2),                                       #maxpooling like compress picture  
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Add the same layers as before
        tf.keras.layers.Flatten(),                                   
        tf.keras.layers.Dense(128, activation='relu'),   
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Print the model summar
    model.summary()
    # Compile 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    

    #=========================== Train the Model ==================================

    print(f'\nMODEL TRAINING:')    
    model.fit(training_images, training_labels, epochs=5)
    #model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    #=========================== Evaluate the Model ================================
    
    print(f'\nMODEL EVALUATION:')
    test_loss = model.evaluate(test_images, test_labels)
    
    
    #=========================== Predict and visualize =============================
    if args.vis ==1:
        f, axarr = plt.subplots(3,4)
        
        FIRST_IMAGE=0
        SECOND_IMAGE=23
        THIRD_IMAGE=28
        CONVOLUTION_NUMBER = 1     
        
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
        
        for x in range(0,4):
            f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
            axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
            axarr[0,x].grid(False)
            
            f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
            axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
            axarr[1,x].grid(False)
            
            f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
            axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
            axarr[2,x].grid(False)

        
    
    
    
    

