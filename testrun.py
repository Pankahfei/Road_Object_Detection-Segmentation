from demo import *

im_dir = 'C:/Users/pkfei/OneDrive/Desktop/Git/RoadObjectDetection/assets'

detector  = Detector(img_dir=im_dir)

detector.onImage('images/test.png')

#detector.onVideo('nighttraffic.mp4')

#detector.renderVideo('nighttraffic.mp4')