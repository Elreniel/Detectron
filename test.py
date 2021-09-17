import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import load_coco_json
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

datasetName = "bonnet"
pretrainedModel = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = "./bonnet/2021_08_17_19_27_57/model_0019999.pth"

predictor = DefaultPredictor(cfg)

register_coco_instances("test_set", {}, datasetName + "/test/_annotations.coco.json", datasetName + "/test")

dataset_dicts = DatasetCatalog.get("test_set")
dataset_metadata = MetadataCatalog.get("test_set")

# for d in random.sample(dataset_dicts, 10):
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=dataset_metadata,
#                    scale=1)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow("result", out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

finalList = []
imDirList = ['C:/Users/bcosk/Desktop/BoneData/video2/','C:/Users/bcosk/Desktop/BoneData/video3/','C:/Users/bcosk/Desktop/BoneData/video4/','C:/Users/bcosk/Desktop/BoneData/koray/']
saveDirectory = "C:/Users/bcosk/Desktop/BoneData_BC/results/"
saveFlag = 1
for imDir in imDirList:
    filelist=os.listdir(imDir)
    for temp in filelist:
        if temp.endswith(".png") or temp.endswith(".jpg"):
            finalList.append(imDir + temp)

for testSample in finalList:
    im = cv2.imread(testSample)
    outputs=predictor(im)
    v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Result", out.get_image()[:, :, ::-1])
    cv2.imwrite(saveDirectory + "Result_" + str(saveFlag) +".png", out.get_image()[:, :, ::-1])
    saveFlag+=1
    cv2.waitKey(0)


# cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# while True:
#     ret_val, im = cam.read()
#     img = cv2.flip(img, 1)
#     outputs = predictor(im)
#     v = Visualizer(im[:, :, ::-1],
#                    metadata=dataset_metadata,
#                    scale=0.5)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imshow("result", out.get_image()[:, :, ::-1])
#     if cv2.waitKey(1) == 27:
#         break  # esc to quit
# cv2.destroyAllWindows()