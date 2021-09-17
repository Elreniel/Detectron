import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch
import numpy as np
import os, json, cv2, random
from datetime import datetime
from myTrainer import MyTrainer
import detectron2.utils.comm as comm

torch.cuda.empty_cache()
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
datasetName = "hardHat"
pretrainedModel = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"

register_coco_instances("train_set", {}, datasetName + "/train/_annotations.coco.json", datasetName + "/train")
register_coco_instances("valid_set", {}, datasetName + "/valid/_annotations.coco.json", datasetName + "/valid")
register_coco_instances("test_set", {}, datasetName + "/test/_annotations.coco.json", datasetName + "/test")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(pretrainedModel))
cfg.DATASETS.TRAIN = ("train_set",)
cfg.DATASETS.TEST = ("valid_set",)
cfg.DATALOADER.NUM_WORKERS = 0 #Windows sadece 0 kabul ediyor

cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

# cfg.SOLVER.BASE_LR = 0.01
# cfg.SOLVER.STEPS = []

# cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.MAX_ITER = 10000  #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 100

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

cfg.OUTPUT_DIR = datasetName + "/2021_08_16_15_53_10"
resumeDir = cfg.OUTPUT_DIR + "/"

trainer = MyTrainer(cfg)
trainer.resume_or_load(resumeDir)# pass the resume_dir path here
trainer.train()