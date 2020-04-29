import csv
import cv2
from scipy import ndimage
from PIL import Image
import numpy as np

rows = []
with open('/opt/images/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)
        
car_images = []
steering_angles = []
for row in rows:
    # steering measurements and adjustments for the side camera images
    steering_center = float(row[3])
    steering_angles.append(steering_center)
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_angles.append(steering_left)
    steering_right = steering_center - correction
    steering_angles.append(steering_right)
    # flipped measurements for generalization
    steering_center_flip = -steering_center
    steering_angles.append(steering_center_flip)
    steering_left_flip = -steering_left
    steering_angles.append(steering_left_flip)
    steering_right_flip = -steering_right
    steering_angles.append(steering_right_flip)

    # read in images from center, left and right cameras
    path = "/opt/images/IMG/" # fill in the path to your training IMG directory
    img_center = ndimage.imread(row[0])
    car_images.append(img_center)
    img_left = ndimage.imread(row[1])
    car_images.append(img_left)
    img_right = ndimage.imread(row[2])
    car_images.append(img_right)
    # flipped images for generalization
    img_center_flip = np.fliplr(img_center)
    car_images.append(img_center_flip)
    img_left_flip = np.fliplr(img_left)
    car_images.append(img_left_flip)
    img_right_flip = np.fliplr(img_right)
    car_images.append(img_right_flip)
        
X_train = np.array(car_images)
y_train = np.array(steering_angles)

freeze_flag = True
weights_flag = 'imagenet'
preprocess_flag = True

from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing import image
import tensorflow as tf

input_size = 139
input_height = 160
input_width = 320

# Removing top layers
inception = InceptionV3(weights=weights_flag, include_top=False, input_shape=(input_height,input_width,3))

if freeze_flag == True:
    for layer in inception.layers:
        layer.trainable = False

# Resize the images to squares of 139x139
road_images = Input(shape=(160,320,3)
#resized_input = Lambda(lambda image: image, tf.image.resize(image,(input_size, input_size),bilinear,True))(road_images)

inp = inception(road_images)                   
x = GlobalAveragePooling2D()(inp)
x = Dense(512, activation='relu')(x)
predictions = Dense(1,activation='softmax')(x)

# Create and compile the model
model = Model(inputs=road_images, outputs=predictions)
model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

# Keras Callback
checkpoint = ModelCheckpoint(filepath='./home/workspace/', monitor='val_loss',save_best_only=True)
                    
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

# Split data to train and test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

# Shuffle the training & test data
X_train, y_train = shuffle(X_train, y_train)
X_val, y_val = shuffle(X_val, y_val)

# Use a generator to pre-process our images
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.incpetion_v3 import preprocess_input

if preprocess_flag == True:
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
else:
    datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()

# Train the model
batch_size = 128
epochs = 5
# Note: we aren't using callbacks here since we only are using 5 epochs to conserve GPU time
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=len(X_train)/batch_size, epochs=epochs, verbose=1, 
                    validation_data=val_datagen.flow(X_val, y_one_hot_val, batch_size=batch_size),
                    validation_steps=len(X_val)/batch_size)

model.save('test3.h5')