import os

import torch

import flash
from flash.core.data.utils import download_data
from flash.video import VideoClassificationData, VideoClassifier

# 1. Create the DataModule
# Find more datasets at https://pytorchvideo.readthedocs.io/en/latest/data.html
#download_data("https://pl-flash-data.s3.amazonaws.com/kinetics.zip", "./data")


datamodule = VideoClassificationData.from_folders(
    #train_folder=os.path.join(os.getcwd(), "data/kinetics/train"),
    train_folder=os.path.join(os.getcwd(), "/home/ewy/Desktop/videoTypes_videos/train"),
    #val_folder=os.path.join(os.getcwd(), "data/kinetics/val"),
    val_folder=os.path.join(os.getcwd(), "/home/ewy/Desktop/videoTypes_videos/validation"),
    clip_sampler="uniform",
    clip_duration=1,
    decode_audio=False,
)


# 2. Build the task
model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes, pretrained=False)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Make a prediction
#predictions = model.predict(os.path.join(os.getcwd(), "path"))
#print(predictions)

# 5. Save the states (parameters/weights) but NOT the model!
trainer.save_checkpoint("video_classification.pt")


# 6. load model (opción 1 que no funcionó pero aprendí stuff) usando las funciones de pytorch

# becsause before we saved only the states (parameters/weights) and not the model, then we forst need to define again the model structure
#model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes, pretrained=False)
# now we load the weights to the model. IMPORTANT to set strict = false
#model.load_state_dict(torch.load("video_classification.pt"), strict=False)
# Now we eval()
# Sets model in evaluation (inference) mode --> equivalent to model.train(False).:
#     • normalisation layers use running statistics
#     • de-activates Dropout layers
# model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you
#model.eval()

# opción 2 para upload the model (la que sí funcionó) suando las funciones de la librería de lighting flash
#model2 = VideoClassifier.load_from_checkpoint("video_classification.pt")

# 7. Make a prediction
#predictions = model2.predict(os.path.join(flash.PROJECT_ROOT, "/Users/wendy/Desktop/videoTypes_videos/predict"))
#print(predictions)


