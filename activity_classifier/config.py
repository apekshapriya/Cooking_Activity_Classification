class  args:

    # training
    model_save_path = "model/activity.model"
    dataset_directory = "data/train/"
    epochs = 12
    lr = 1e-5
    plot_metrics_path = "../save_metrics.png" 
    classes_list = ["add_ingredients", "stir","nothing"]
    batch_size = 16
    image_height, image_width = 224, 224
    seed = 23

    #prediction
    
    model_checkpoint = "../model/activity.model"
    input_video =  "One-PotChickenFajitaPasta.mp4"
    
    output_video = "output.avi"
    size = 1
