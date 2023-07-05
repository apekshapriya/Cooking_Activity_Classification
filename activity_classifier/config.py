from pathlib import Path
import mlflow

class  args:

    # training
    BASE_DIR = Path(__file__).parent.parent
    
    dataset_directory = Path(BASE_DIR,"data/train/")
    plot_metrics_path = Path(BASE_DIR,"plots/save_metrics.png") 

    classes_list = ["add_ingredients", "stir","nothing"]

    epochs = 5
    lr = 1e-5
    batch_size = 16
    image_height, image_width = 224, 224
    seed = 23
    size = 1


    # Set tracking URI
    MODEL_REGISTRY = Path(BASE_DIR, "experiments")
    Path(MODEL_REGISTRY).mkdir(exist_ok=True) # create experiments dir
    mlflow.set_tracking_uri(str(MODEL_REGISTRY.absolute()))
    
    
