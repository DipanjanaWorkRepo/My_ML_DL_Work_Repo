import tensorflow as tf
import numpy as np
import os

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

from sklearn.utils.class_weight import compute_class_weight

# Image Path in Local
train_dir = '<path to training folder>/train'
test_dir = '<path to test folder>/test'

# Downloading pre-trained VGG-16 model using Keras API
# The VGG16 model contains a convolutional part and a fully-connected (or dense) part which is used for classification.
# If include_top=True then the whole VGG16 model is downloaded which is about 528 MB.
# If include_top=False then only the convolutional part of the VGG16 model is downloaded

model = VGG16(include_top=True, weights='imagenet')

#check the shape of the tensors expected as input by the pre-trained VGG16 model
input_shape = model.layers[0].output_shape[1:3]    #input shape[1:3] gives height & width of the tensor

# Making use of Keras's data-generator function to input data to the neural network
# Increasing the size of the input data set by data augmentation during the data input process

datagen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=180,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=[0.9, 1.5],
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')


#Only rescaling is done for test dataset as increasing dataset size is not required for testset
datagen_test = ImageDataGenerator(rescale=1./255)  # Rescaling the pixel-values so they are between 0.0 and 1.0 because this is expected by the VGG16 model



# ImageDataGenerator outputs images in batches
batch_size = <set some suitable number>

# Creating a placeholder where the overly distorted images(as a result of augmentation) could be saved(if any)
if True:
    save_to_dir = None
else:
    save_to_dir='augmented_images/'


# Creating the actual data-generator that will read files from disk, resize the images and return a random batch
generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

# Data-generators will loop for eternity, we need to specify the number of steps to perform during evaluation and prediction on the test-set

steps_test = generator_test.n / batch_size

# File-paths of images in training and test datasets
image_paths_train = [os.path.join(train_dir, filename) for filename in generator_train.filenames]
image_paths_test = [os.path.join(test_dir, filename) for filename in generator_test.filenames]

# Get the class-numbers for all the images in the training and test-sets
cls_train = generator_train.classes
cls_test = generator_test.classes

#Number of categories to classify
num_classes = generator_train.num_classes

# Sometimes dataset would have inbalance in number of different classes of images being used in training
# To straighten up the issue scikit-learn is being used to calculate weights that will properly balance the dataset.
# These weights are applied to the gradient for each image in the batch during training,
# so as to scale their influence on the overall gradient for the batch
class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)

# The lower layers of a Convolutional Neural Network can recognize many different shapes or features in an image.
# It is the last few fully-connected layers that combine these featuers into classification of a whole image.
# So we can try and re-route the output of the last convolutional layer of the VGG16 model to a new fully-connected neural network
# that we create for doing classification on our dataset.

# Getting the summary of the VGG-16 model will help to identify the name of the
# last convolutional layer of this model we would like to choose for our purpose
model.summary()

# We can see that the last convolutional layer is called 'block5_pool' so we use Keras to get a reference to that layer.
# We refer to this layer as the Transfer Layer because its output will be re-routed to our new fully-connected neural network
# which will do the classification for our dataset
transfer_layer = model.get_layer('block5_pool')
print(transfer_layer.output)


# Next is to create a new model using Keras API
# First we take the part of the VGG16 model from its input-layer to the output of the transfer-layer
conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

# Next we will build a new model on top of this
# Start a new Keras Sequential model.
new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
new_model.add(Flatten())

# Add a dense (i.e fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dropout-layer to prevent possible overfitting and
# improve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add the final layer for the actual classification.
new_model.add(Dense(num_classes, activation='softmax'))

# Defining the hyperparametres required to train our model


optimizer = Adam(lr=1e-5)

loss = 'categorical_crossentropy'  #as our dataset has more than 2 categories of objects we used categorical_crossentropy as loss function

metrics = ['categorical_accuracy']  #to validate classification accuracy of our model

# In Transfer Learning we intend to reuse the pre-trained VGG16 model as it is, so we will disable training for all its layers

conv_model.trainable = False

for layer in conv_model.layers:
    layer.trainable = False

# Next our new model is required to be compiled to have the changes made on it to take effect
new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# An epoch normally means one full processing of the training-set.
# But the data-generator that we created above, will produce batches of training-data for eternity.
# So we need to define the number of steps we want to run for each "epoch"
# and this number gets multiplied by the batch-size defined above.
# For example,if we have 100 steps per epoch and a batch-size of 20, so one "epoch" will cover of 2000 random images from the training-set

epochs = <set some suitable number>
steps_per_epoch = <set some suitable number>

# Train the model
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

# Verifying classification accuracy:
result = new_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))

#Fine Tuning

# Once the new classifier has been trained we can try and gently fine-tune some of the deeper layers in the VGG16 model through Fine Tuning
# We want to train the last two convolutional layers whose names contain 'block5' or 'block4'

conv_model.trainable = True

for layer in conv_model.layers:
    # Boolean whether this layer is trainable.
    trainable = ('block5' in layer.name or 'block4' in layer.name)

    # Set the layer's bool.
    layer.trainable = trainable

# Tuning up learning rate
optimizer_fine = Adam(lr=1e-7)

# Recompile the model so that changes can take effect before we continue training
new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

# Train the model
history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

result = new_model.evaluate_generator(generator_test, steps=steps_test)

print("Test-set classification accuracy: {0:.2%}".format(result[1]))
