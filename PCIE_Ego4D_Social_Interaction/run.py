from common.config import argparser
from common.utils import *
from common.engine import train, inference
from dataset.dataAugmentation import *
import torchvision.transforms as Transforms

from model.Model import getModel
from dataset.Dataset import getLoader
    
def main(cfg):
    
    cfg.ckptLoadPath = cfg.CkptLoadPath
    cfg.audioTransform = AudioAugmentationHelper([(0.9, RandomCrop(3))])
    
    fixSeeds(cfg.seed)
    device = getDefaultDevice()
    model = getModel(cfg, device)
    
    if (not cfg.eval):
        trainLoader = getLoader(cfg, "train")
        validLoader = getLoader(cfg, "valid")
        train(cfg, model, trainLoader, validLoader, device)
    else:
        testLoader = getLoader(cfg, "test")
        freezeWeights(model)
        model.eval()
        inference(cfg, model, testLoader, device)

if (__name__ == "__main__"):
    cfg = argparser.parse_args()
    main(cfg)
