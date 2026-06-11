import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/')
    # model.load('weights/rtdetr-r18.pt, strict=False') # loading pretrain weights
    model.train(data='dataset4/data4.yaml',
                cache=False,
                imgsz=640,
                epochs=800,
                batch=16, 
                workers=4, 
                patience=0, 
                project='runs/train',
                name='',
                )

   
