from scipy import ndimage
import csv
import cv2
import numpy as np

lines = []
# read in csv file
with open('/opt/images2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
# pull data from csv file to locate image files and steering measurements
for line in lines:
    for i in range(3):
        # pulling center, left, and right images
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = '/opt/images2/IMG/' + filename
        image = ndimage.imread(current_path)
        # adding to the array
        images.append(image)
        # adding the flipped version to the array as well
        images.append(cv2.flip(image,1))
        # locating the steering angle measurements
        measurement = float(line[3])
        # creating an offset for side images
        offset = 0.2
        # steering for center image
        if i == 0:
            measurements.append(measurement)
            measurements.append(-measurement)
        # steering for left image
        if i == 1:
            left_steer = measurement+offset
            measurements.append(left_steer)
            measurements.append(-left_steer)
        # steering for right image
        if i == 2:
            right_steer = measurement-offset
            measurements.append(right_steer)
            measurements.append(-right_steer)

# creating numpy arrays
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D

# building model based off of nvidia's model
model = Sequential()
# normalize and mean center pixels
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# crop portions of the image that are not relevant to driving
model.add(Cropping2D(cropping=((70,25), (0,0))))
# conv1
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
# maxpooling
#model.add(MaxPooling2D())
# conv2
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
# maxpooling
#model.add(MaxPooling2D())
# conv3
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
# conv4
model.add(Convolution2D(64,3,3,activation='relu'))
# conv5
model.add(Convolution2D(64,3,3,activation='relu'))
# flatten
model.add(Flatten())
# dense
model.add(Dense(100))
# dense
model.add(Dense(50))
# dense
model.add(Dense(1))

# using adam optimizer
model.compile(loss='mse', optimizer='adam')
# using a 80/20 train/test split with shuffling and 10 epochs
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

# save the model to use for autonomous driving
model.save('model.h5')
exit()