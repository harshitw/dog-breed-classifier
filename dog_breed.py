# dog project

import tensroflow as tf  # importing files and libraries
import numpy as np
from sklern.datasets import load_files
from keras.utils import np_utils
from glob import glob


# function to load the train, test and validation data
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))


import random
random.seed(8675309)

# load filenames in a shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))


# Detecting humans
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# Human face detector
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


human_files_short = human_files[:100]
dog_files_short = train_files[:100]

for i in len(human_files_short):
    if face_detector(human_files_short[i]):
        human += 1
    if face_detector(dog_files_short[i]):
        dog += 1

print(human)
print(dog)


# Detect dogs
from keras.applications.resnet50 import ResNet50

# define resnet50 model
ResNet50_model = ResNet50(weights = 'imagenet')

# preprocessing the data
from keras.preprocessing import image
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# Making predictions with ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


# writing a dog detector
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


# Assess the dog detector
for i in len(human_files_short):
    if dog_detector(human_files_short[i]):
        human1 += 1
    if dog_detector(dog_files_short[i]):
        dog1 += 1

print(human1)
print(dog1)


# Create a CNN from scratch to classify dog breeds

# Preprocess the data
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

# Model Architecture
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization

model = Sequential()

# conv - max pool
model.add(Conv2D(16, (3, 3), padding = 'same', input_shape = (224, 224, 3)))
model.add(BatchNormalization(axis = 3, scale = False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (4, 4), strides = (2, 2), padding = 'same'))

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (224, 224, 3)))
model.add(BatchNormalization(axis = 3, scale = False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (4, 4), strides = (2, 2), padding = 'same'))

model.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (224, 224, 3)))
model.add(BatchNormalization(axis = 3, scale = False))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (4, 4), strides = (2, 2), padding = 'same'))

model.add(GlobalAveragePooling2D(input_shape = ))
model.add(Dense(133, activation = 'softmax'))

model.summary()


# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
from keras.callbacks import ModelCheckpoint
epochs = 20

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, validation_data=(valid_tensors, valid_targets), epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

model.load_weights('saved_models/weights.best.from_scratch.hdf5')


# Test the model
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# CNN to classify dog breeds using transfer learning using VGG16 model
# obtain bottleneck_features
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']

# model architecture
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()

# compile the model
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# train the model
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, validation_data=(valid_VGG16, valid_targets), epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

# load the model with best validation loss
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

# test the model
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


# Predict the dog breed with the model
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


# CNN to classify dog breed using transfer learning with ResNet50 bottleneck features

# Obtain bottleneck_features
bottleneck_features = np.load('bottleneck_features/Dog{ResNet-50}Data.npz')
train_ResNet-50 = bottleneck_features['train']
valid_ResNet-50 = bottleneck_features['valid']
valid_ResNet-50 = bottleneck_features['valid']
test_ResNet-50 = bottleneck_features['test']

# Model Architecture
ResNet50_model = Sequential()
ResNet50_model.add(GlobalAveragePooling2D(input_shape = train_ResNet-50.shape[:1]))
ResNet50_model.add(Dense(133, activation = 'softmax'))

ResNet50_model.summary()

# compile the model
ResNet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.ResNet50.hdf5',
                               verbose=1, save_best_only=True)

ResNet50_model.fit(train_ResNet-50, train_targets, validation_data=(valid_ResNet-50, valid_targets), epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

# load the model weights with best validation loss
ResNet50_model.load_weights('saved_models/weights.best.ResNet50.hdf5')

# Test the model
ResNet50_predictions = [np.argmax(ResNet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_ResNet-50]

# Predict the dog breed with the model
from extract_bottleneck_features import *

def ResNet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_ResNet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = ResNet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
