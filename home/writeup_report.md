# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[image2]: ./examples/left_camera_recovery.jpg "Recovery using left camera"
[image3]: ./examples/recovery_left.jpg "Recovery Image"
[image4]: ./examples/original_flip.jpeg "Original and Flip"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 for a recording of car in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the Nvidia model with 5 convolutions and 3 fully connected layers. This proved to be successful right away when implemented in my code. The initial lambda layer helps normalize the data and center mean the pixels. Next it is cropped to remove excess data in the images that are not needed for the neural net training. Each of the convolutions have relu layers to keep it nonlinear and continue to add more depth to the image. This is then flattened and completed with the 3 fully connected layers and ending with one output since I am only worried about the steering angle as my output. Code can be found in model.py lines 53-80.

#### 2. Attempts to reduce overfitting in the model

I originally had dropout layers but did not see a large benefit or any noticable improvements so continued without the dropout layers and the model successfully went around the track comfortably.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

The training data was based on how I drove around the track with center lane driving to keep the vehicle as centered as possible. I utilized the side camera images as well to help out with recovery with a steering offset applied to those images to help recovery efforts. I also included flipped images to help generalize so that my model is able to train on both versions of the track, regular and flipped.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a known working model that could be applied to applications similar to this project. Nvidia's seemed to be a clear winner for this since it was designed to take in images and help an autonomous vehicle navigate.

My first step was to create a simple model that takes in the images successfully and can train a model that only had one dense layer and built up the network from there.

I redrove the track after finding some flaws in some of my data from training that would cause the model to have hesitations in portions of the track. After recollecting data the model was much smoother around some turns that were problematic in the prior iteration.

The final step was to run the simulator to see how well the car was driving around the track. There were a few spots where the car would ride the edges due to the data points from my training dataset. To improve the driving behavior in these cases, I redrove the track and ran through the model to create a better neural net.

At the end of the process, the car is able to drive autonomously around the track without leaving the road and can go endlessly around the track over and over.

#### 2. Final Model Architecture

The final model architecture consists of normalization/mean centering, cropping, 5 convolutions, flatten, and 3 fully connected dense layers with a single output.

```
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
```

#### 3. Creation of the Training Set & Training Process

All my training data was center lane driving, or close to it.

![alt text][image1]

Then I utilized the side camera angles to assist with recovery with a steering correction offset to help bring the car back to the center.

![alt text][image2]
![alt text][image3]

I augmented the dataset to also have a flipped version of the track so that the model would be more generalized and have better examples of left and right curves in the road.

![alt text][image4]

In total there are 6,996 images from the center, left, and right camera images and a flipped version of those as well for a total of 13,992 images for training which were shuffled and with 20% set aside for validation dataset of 2,798 images.

I used an adam optimizer so that manually training the learning rate wasn't necessary and used 10 epochs for training the model.
