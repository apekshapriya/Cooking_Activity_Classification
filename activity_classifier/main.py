import os
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import json

import pickle
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import mlflow
import tempfile
from config import *

# mlflow.tensorflow.autolog()
mlflow.set_experiment(experiment_name="baselines_3")


seed_constant = args.seed
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

epochs = args.epochs
model_save = args.model_save_path
plot_path = args.plot_metrics_path
image_height, image_width = args.image_height, args.image_width
dataset_directory = args.dataset_directory
classes_list = args.classes_list
lr = args.lr
batch_size = args.batch_size

os.makedirs(plot_path, exist_ok=True)


def frames_extraction(video_path):
    # Empty List declared to store frames
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")

    frames_list = []

    images = os.listdir(video_path)
    for img in images:
        frame = cv2.imread(os.path.join(video_path, img))
    
        # Resize the Frame to fixed Dimensions
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_height, image_width))
    
        frame = frame-mean
        frames_list.append(frame)

    # returning the frames list 
    return frames_list



def create_dataset():

    # Declaring Empty Lists to store the features and labels values.
    temp_features = [] 
    features = []
    labels = []
    
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
        
        # Getting the list of  folders present in the specific class name directory
        folders_list = os.listdir(os.path.join(dataset_directory, class_name))

        # Iterating through all the folders present in the folders list
        for folder in folders_list:
 
            if  os.path.isfile(os.path.join(dataset_directory, class_name, folder)):
                continue

            # Construct the complete folder path
            folder_path = os.path.join(dataset_directory, class_name, folder)

            # Calling the frame_extraction method for every folder path
            frames = frames_extraction(folder_path)

            # Appending the frames to a temporary list.
            temp_features.extend(frames)
        
        # Adding frames to the features list
        features.extend(temp_features)

        # Adding number of labels to the labels list
        labels.extend([class_index] * len(temp_features))
        
        # Emptying the temp_features list so it can be reused to store all frames of the next class.
        temp_features.clear()

    # Converting the features and labels lists to numpy arrays
    features = np.asarray(features)    
    labels = np.array(labels) 
   
    return features, labels


# Let's create a function that will construct our model
def create_model():

    # We will use a Sequential model for model construction
    baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(image_width, image_height, 3)))  # 224, 224, since resnet accepts this dim
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
  
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(classes_list), activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the training process. The last 4 layers will be trainable
    for layer in baseModel.layers[:-4]:
        layer.trainable = False

    return model

def save_model():
    pass

def save_dict(d, filepath):
    """Save dict to a json file."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def load_dict(filepath):

    with open(filepath):
        d = json.load(filepath)

    return d


def train():
    # create dataset

    data, labels = create_dataset()

    class_weights = compute_class_weight(class_weight = "balanced",
                                            classes = np.unique(labels),
                                            y = labels)


    class_weights = dict(zip(np.unique(labels), class_weights))

    one_hot_encoded_labels = to_categorical(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, one_hot_encoded_labels,
        test_size=0.10, stratify=labels, random_state=42, shuffle=True)


    # Calling the create_model method
    model = create_model()

    print("[INFO] compiling model...")

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=1e-5),
        metrics=[tf.keras.metrics.CategoricalAccuracy()])

    # train the head of the network for a few epochs (all other layers
    # are frozen except the last 4) -- this will allow the new FC layers to start to become
    # initialized with actual "learned" values versus pure random

    mlflow.keras.autolog()
    print("[INFO] training head...")

    H = model.fit(
        x = trainX,
        y = trainY,
        steps_per_epoch=len(trainX) // batch_size,
        validation_split = 0.15,
        epochs=epochs,
        batch_size = batch_size,
        class_weight=class_weights)


    print("[INFO] serializing network...")
    print("Model Trained Successfully!")

    print("[INFO] evaluating network...")
    predictions = model.predict(x=testX.astype("float32"), batch_size=batch_size)
    report = classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=classes_list, output_dict=True)
    print(report)
    # print(report['macro avg'])

    # plot the training loss and accuracy
    N =  epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["categorical_accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_categorical_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("save_plot.png")

    return {
        'args': args,
        'model': model,

        'performance': {
            'precision':report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1-score': report['macro avg']['f1-score'],
            'accuracy': report['accuracy']
                    }
            }

if __name__ == """__main__""":
    
    with mlflow.start_run(run_name="start_exp_1"):

        # training and saving metrics
        artifacts = train()

        mlflow.log_metrics({'precision': artifacts['performance']['precision'],
                            'recall': artifacts['performance']['recall'],
                            'f1-score': artifacts['performance']['f1-score'],
                            'accuracy': artifacts['performance']['accuracy']})

        with tempfile.TemporaryDirectory() as dp:

                artifacts['model'].save(Path(dp, "activity.model"), save_format="h5")
                save_dict(artifacts['performance'], Path(dp, "performance.json"))

                mlflow.log_artifacts(dp)

        mlflow.log_params(vars(artifacts["args"]))

    