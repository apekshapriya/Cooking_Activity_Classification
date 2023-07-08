# Classification of cooking activity



## Installation:

### clone the repo
```python
$ git clone https://github.com/apekshapriya/Cooking_Activity_Classification.git

$ cd Cooking_Activity_Classification
```

### Create a new environment through venv and activate it

``` python
$ python3 -m venv venv_ac
$ source venv_ac/bin/activate
```

### Further, all the dependencies are listed in requirements.txt and can be installed by:

``` python
$ python3 -m pip install pip setuptools wheel

$ python3 -m pip install -e .
```

## Dataset:

Manually labelled and created dataset from EPIC-KITCHEN-100 by choosing random images for all the three classes "add_ingredients", "stir" and "nothing" and then labelled them. Kept on increasing the dataset until the model started picking up the different activities


Some detais of the final dataset:

Total samples:  291

Stir Samples: 127

Add Ingredients Sample: 87

Nothing Samples:  77


If required, dataset that was used for training can be accessed from [here](https://drive.google.com/drive/folders/1QWs9cNZ-_InbOgmaW0jwdbA54rjILfs-?usp=sharing)

To create the dataset:

    The structure of the dataset is shown below:

    data:
        add_ingredients
                frame1.jpg
                .........
                frame10.jpg
        stir
                frame1.jpg
                .........
                frame10.jpg
        nothing
                frame1.jpg
                .........
                frame10.jpg


#### Note:

If we want to better the model further, the dataset can be used wisely.


## Model

The model's checkpoint can be downloaded from [here](https://drive.google.com/drive/folders/1ad8UC83DibC9QWPH1lm1sD3ypU7ovV02?usp=sharing).


The model is based on architecture of resnet50 backbone originaly trained on ImageNet dataset for classification.


Updates that were made to the architecture:

    The top layer was removed
    The parameters of the last 4 layers were kept trainable (4 was chosen after iterative experiment)
    Average Pooling and Dropout Layer were added after the second last layer
    One Dense layer was added on top for classification


This model is chosen after thorough research and comparison of other pretrained models and repositories.

Few research that was done before taking this model.

    Thought of solving the activity classification as video classification problem so as to keep the context of temporal dpendencies. Therefore tried these different models:
        Experimented with "Slowfast" pretrained model trained on Kinetics dataset
        Experimented with "Slowfast" pretrained model trained on Epic-Kitchen dataset
        Video Classification model using "Early Fuse" where certain number of consecutive frames were selected randomly from videos and the input of all the frames were given to network.

    These  models failed for me terribly as the dataset was too small for these complex models.



#### Note
The config file is given to configure paths and models hyperparameters for prediction, training and testing.

## Predictions from the model on test video:

#### Update the following args in config.py

    model_checkpoint = "model/activity.model"
    input_video =  "One-PotChickenFajitaPasta.mp4"
    output_video = "output.avi"
    size = 1


#### Run the following command

```python
$ cd activity_classifier
$ python predict.py

```

## Training the model

#### Update the following args in config.py

    model_save_path = "model/activity.model"
    dataset_directory = "data/"
    classes_list = ["add_ingredients", "stir","nothing"]

    Update other hyperparameters if required

#### Run the following command:

```python
$ cd activity_classifier
$ python main.py
```


#### Note:

After training the model, it can be seen that the model is able to pick up the three classes but faces confusion with stir and nothing classes. This can be improved by carefully adding the samples for both the classes that differentiates them.


## Experiment Tracking and metrics:

The plot for Loss and Accuracy is shown below

![Loss_and_Accuracy_plot](https://github.com/apekshapriya/Cooking_Activity_Classification/blob/master/plots/plot_latest.png)


The confusion matrix is shown below.

                    precision    recall  f1-score   support

    add_ingredients       0.78      0.78      0.78         9
        stir              0.73      0.85      0.79        13
        nothing           1.00      0.75      0.86         8

        accuracy                              0.80        30
        macro avg         0.84      0.79      0.81        30
    weighted avg          0.82      0.80      0.80        30


## Results for the video

Results of activity classification is shared [here](https://drive.google.com/file/d/1Gh9HojqNytg3u6Aj-SlYtRJH7YWYdQ6n/view?usp=sharing)
