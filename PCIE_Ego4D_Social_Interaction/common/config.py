import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument("--eval", action = "store_true", help = "Running mode")
argparser.add_argument("--startFromCkpt", type = bool, default = True, help = "Start from a checkpoint or random initialized model")
argparser.add_argument("--seed", type = int, default = 42, help = "Random seed")

# path
argparser.add_argument("--tensorboardPath", type = str, default = "./tensorboard", help = "Directory for tensorboard")
argparser.add_argument("--CkptLoadPath", type = str, default = "./checkpoints/best.ckpt", help = "Path to the checkpoint for audio branch")
qua
argparser.add_argument("--ckptSavePath", type = str, default = "./checkpoints", help = "Directory to store checkpoints")
argparser.add_argument("--audioPath", type = str, default = "./data/wave", help = "Directory for audio files")
argparser.add_argument("--ttmPath", type = str, default = "./data/ttm", help = "Directory for ttm label files")
argparser.add_argument("--trainSplitPath", type = str, default = "./data/split/train.txt", help = "Path to the text file storing IDs of training data")
argparser.add_argument("--validSplitPath", type = str, default = "./data/split/valid.txt", help = "Path to the text file storing IDs of validation data")
argparser.add_argument("--testDataPath", type = str, default = "./data/social_test/final_test_data", help = "Directory for test data")
argparser.add_argument("--testScorePath", type = str, default = "./data/social_test/qualityScoreTest.json", help = "Path to quality scores of training data")
argparser.add_argument("--audio_pred", type = str, default = "./audio_pred.json", help = "Path to audio prediction")
argparser.add_argument("--vision_pred", type = str, default = "./vision_pred.json", help = "Path to audio prediction")

# dataloader
argparser.add_argument("--batchSizeForward", type = int, default = 32, help = "Batch size per forward pass")
argparser.add_argument("--numWorkers", type = int, default = 4, help = "Number of workers used in dataloader")
argparser.add_argument("--minLength", type = float, default = 0.1, help = "Miniumum length for audio training data")
argparser.add_argument("--dataStride", type = int, default = 8, help = "Stride between each training sample")
qua

# training
argparser.add_argument("--lr", type = float, default = 1e-5, help = "Learning rate")
argparser.add_argument("--numEpoch", type = int, default = 1, help = "Maximum number of epochs")
argparser.add_argument("--gradientClip", type = float, default = 1, help = "Gradient clipping")
argparser.add_argument("--earlyStop", type = int, default = 100, help = "Maximum patience for early stopping")
argparser.add_argument("--updateEveryBatch", type = int, default = 1, help = "Number of backward passes per update")
argparser.add_argument("--saveEveryEpochs", type = int, default = 1, help = "Save a checkpoint every k epochs")

# model
argparser.add_argument("--BackboneType", type = str, default = "small", help = "variant of Whisper backbone used in the audio branch")
argparser.add_argument("--Dim", type = int, default = 768, help = "Feature dimension used in the audio branch")
argparser.add_argument("--NumHeads", type = int, default = 8, help = "Number of heads used in the self-attention modeules in the audio branch")
argparser.add_argument("--NumLayers", type = int, default = 1, help = "Number of additional self-attention modeules in the audio branch")
argparser.add_argument("--Dropout", type = float, default = 0.25, help = "Dropout rate used in the audio branch")
argparser.add_argument("--FreezeBackbone", type = bool, default = True, help = "Freeze the Whisper backbone in the audio branch")


