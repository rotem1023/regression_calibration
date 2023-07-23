import pandas as pd
import cv2

path = 'C:/lior/studies/master/projects/calibration/regression calibration/Tracking_Robotic_Testing/Tracking/Dataset1/'

df = pd.read_csv(path + 'Right_Instrument_Pose.txt', header=None, delim_whitespace=True)
print(len(df.index))

vidcap = cv2.VideoCapture(path + 'Video.avi')
success,image = vidcap.read()
count = 0
while success:
    print(count)
    cv2.imwrite(path + f"frames/{(count+1):04}.png", image)     # save frame as PNG file   
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1