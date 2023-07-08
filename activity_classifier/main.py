import os
import cv2
import math
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend

import matplotlib.pyplot as plt
import numpy as np
import json
import pickle
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import compute_class_weight
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import plot_model
import mlflow
import tempfile
from pathlib import Path
from config import *




# Set seed
seed_constant = args.seed
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

# Set up directories
os.makedirs(args.plot_metrics_path, exist_ok=True)
def frames_extraction(video_path):
    """
    Extract frames from a video path.
    :param video_path: Path to the video.
    :return: List of frames extracted from the video.
    """
    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    frames_list = []
    images = os.listdir(video_path)
    for img in images:
        frame = cv2.imread(os.path.join(video_path, img))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (args.image_height, args.image_width))
        frame = frame - mean
        frames_list.append(frame)
    return frames_list


def create_dataset():
    """
    Create the dataset by extracting frames from videos in the specified directory.
    :return: Features (frames) and labels arrays.
    """
    temp_features = []
    features = []
    labels = []
    for class_index, class_name in enumerate(args.classes_list):
        print(f'Extracting Data of Class: {class_name}')
        folders_list = os.listdir(os.path.join(args.dataset_directory, class_name))
        for folder in folders_list:
            if os.path.isfile(os.path.join(args.dataset_directory, class_name, folder)):
                continue
            folder_path = os.path.join(args.dataset_directory, class_name, folder)
            frames = frames_extraction(folder_path)
            temp_features.extend(frames)
        features.extend(temp_features)
        labels.extend([class_index] * len(temp_features))
        temp_features.clear()
    features = np.asarray(features)
    labels = np.array(labels)
    return features, labels


def create_model():
    """
    Create the model architecture based on ResNet50.
    :return: Compiled Keras model.
    """
    baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(args.image_width, args.image_height, 3)))
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(len(args.classes_list), activation="softmax")(headModel)
    model = Model(inputs=baseModel.input, outputs=headModel)
    for layer in baseModel.layers[:-4]:
        layer.trainable = False
    return model


def save_dict(d, filepath):
    """
    Save a dictionary to a JSON file.
    :param d: Dictionary to save.
    :param filepath: Path to the JSON file.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)


def train():
    """
    Train the model using the dataset and evaluate its performance.
    :return: Dictionary containing trained model, performance metrics, and arguments.
    """
    # Create dataset
    data, labels = create_dataset()

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
    class_weights = dict(zip(np.unique(labels), class_weights))
    one_hot_encoded_labels = to_categorical(labels)
    trainX, testX, trainY, testY = train_test_split(data, one_hot_encoded_labels, test_size=0.10, stratify=labels, random_state=42, shuffle=True)

    # Create and compile the model
    model = create_model()
    print("[INFO] Compiling model...")
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=args.lr), metrics=[tf.keras.metrics.CategoricalAccuracy()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    mlflow.keras.autolog()
    print("[INFO] Training model...")
    H = model.fit(
        x=trainX,
        y=trainY,
        steps_per_epoch=len(trainX) // args.batch_size,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=[early_stopping] 
    )

    print("[INFO] Serializing network...")
    print("Model trained successfully!")

    print("[INFO] Evaluating network...")
    predictions = model.predict(x=testX.astype("float32"), batch_size=args.batch_size)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=args.classes_list, output_dict=True)
    print(report)

    # Plot the training loss and accuracy
    N = args.epochs
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
    plot_path = "save_metrics.png"
    
    plt.savefig(plot_path)

    return {
        'args': args,
        'model': model,
        'performance': {
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1-score': report['macro avg']['f1-score'],
            'accuracy': report['accuracy']
        }
    }


if __name__ == "__main__":
    mlflow.set_experiment(experiment_name="baselines_3")
    with mlflow.start_run(run_name="start_exp_1"):
        # Train and save metrics
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
