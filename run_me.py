from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
imagePath = "images/test3.jpg"
threshold = 0.5


detector = Detector()
classFile = "dataset.txt"
detector.readClasses(classFile)
detector.downloadMode(modelURL)
detector.loadModel()


detector.predictImage(imagePath)
