import numpy as np
import os, json, cv2, random
import torch
import pandas as pd

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer,ColorMode,BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog


class Detector:
      
      def __init__(self,img_dir):
          
          self.cfg = get_cfg()
          self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
          self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
          self.cfg.MODEL.WEIGHTS = os.path.join(img_dir, "model_final.pth")  # path to the model we just trained

          self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.75]
          self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

          self.cfg.MODEL.DEVICE = "cpu" 

          self.predictor = DefaultPredictor(self.cfg)

          def get_dicts(img_dir):
              json_file = os.path.join(img_dir, "json_annotation.json")
              with open(json_file) as f:
                   imgs_anns = json.load(f)

              dataset_dicts = []
              for idx, v in enumerate(imgs_anns):
                   record = {}
        
                   filename = os.path.join(img_dir, v["file_name"])
                   height = v["height"]
                   width = v["width"]
                   category_id = v["category_id"]
                   mask = v["TrueMask"]
                   bbox = v["bbox"]
        
                   record["file_name"] = filename
                   record["image_id"] = idx
                   record["height"] = height
                   record["width"] = width
      
                   objs = []
                   for i in range(len(category_id)):
                      obj = {
                           "bbox": bbox[i],
                           "bbox_mode": BoxMode.XYXY_ABS,
                           "segmentation": [mask[i]],
                           "category_id": category_id[i],
                            }
                      objs.append(obj)
                   record["annotations"] = objs
                   dataset_dicts.append(record)
              return dataset_dicts

          for d in ["train", "val"]:
             DatasetCatalog.register("capstone_" + d, lambda d=d: get_dicts(img_dir+ '/' + d))
             MetadataCatalog.get("capstone_" + d).set(thing_classes=["drivable surface","car","pedestrian","barrier","trafficcone","truck","motorcycle","construction worker"])
             MetadataCatalog.get("capstone_" + d).set(thing_colors=[[0, 0, 255],[255, 165, 0],[255, 0, 0],[0, 255, 0],[255, 255, 0],[255, 0, 255],[0, 255, 255],[255, 255, 255]])
          #self.capstone_train = {}
          self.capstone_metadata = MetadataCatalog.get("capstone_train")
          self.im_dir = img_dir

          print('Initialise model done!')


      def onImage(self, filename):
          
          class MyVisualizer(Visualizer):
            def _jitter(self, color):
             return color

          file_name = os.path.join(self.im_dir,filename)
          im = cv2.imread(file_name)
          outputs = self.predictor(im)  

          v = Visualizer(im[:, :, ::-1],
              metadata=self.capstone_metadata, 
              scale=3, 
              instance_mode = ColorMode.SEGMENTATION
               )
          out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
          cv2.imshow('Frame',out.get_image()[:, :, ::-1])
          cv2.waitKey(0)

      def onVideo(self, filename):

         class MyVisualizer(Visualizer):
            def _jitter(self, color):
             return color

         file_name = os.path.join(self.im_dir,filename)
         cap = cv2.VideoCapture(file_name)
         frames_per_second = cap.get(cv2.CAP_PROP_FPS)
         num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
         size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

         if (cap.isOpened()==False):
            print ("Error Opening the file...") 
            return

         (sucess, im) = cap.read()

         while sucess:
            
            #height, width, layers = im.shape
            #new_h = int(height / 2)
            #new_w = int(width / 2)
            #im= cv2.resize(im, (new_w, new_h))
            outputs = self.predictor(im)
            v = Visualizer(im[:, :, ::-1],
              metadata=self.capstone_metadata, #metadata=capstone_metadata
              scale=0.5, 
              instance_mode = ColorMode.SEGMENTATION
               )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow('Frame',out.get_image()[:, :, ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
               break
            (sucess, im) = cap.read()   

         cap.release()
         cv2.destroyAllWindows()


      def renderVideo(self, filename):


         file_name = os.path.join(self.im_dir,filename)
         cap = cv2.VideoCapture(file_name)
         frames_per_second = cap.get(cv2.CAP_PROP_FPS)
         # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
         size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

         fourcc = cv2.VideoWriter_fourcc(*'XVID')
         out_dir = os.path.join(self.im_dir,'your_video.avi')
         output = cv2.VideoWriter(out_dir, fourcc, fps=float(frames_per_second), frameSize=size)

         if (cap.isOpened()==False):
            print ("Error Opening the file...") 
            return
         
         (sucess, im) = cap.read()

         while sucess:
            

            outputs = self.predictor(im)
            v = Visualizer(im[:, :, ::-1],
              metadata=self.capstone_metadata, 
              scale=1, 
              instance_mode = ColorMode.SEGMENTATION
               )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow('Frame',out.get_image()[:, :, ::-1])
            output.write(out.get_image()[:, :, ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
               break
            (sucess, im) = cap.read() 
         
         cap.release()
         output.release()
         cv2.destroyAllWindows()  
          
