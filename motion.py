import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import time
import pandas as pd
from datetime import datetime

first_frame = None
status_list = [None, None]
times = []
df = pd.DataFrame(columns=["Start", "End"])
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 225, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    cnts, hierarchy = cv2.findContours(
        thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    status_list.append(status)
    status_list = status_list[-2:]

    # start motion
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    # end motion
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

        # append times to dataframe
        df = df.append(
            {"Start": times[-2], "End": times[-1]}, ignore_index=True)

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta", delta_frame)
    cv2.imshow("Threshold", thresh_frame)
    cv2.imshow("color", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

# release video and close windows
video.release()
cv2.destroyAllWindows()

# convert Start and End columns to datetime format
df["Start"] = pd.to_datetime(df["Start"], format="%Y-%m-%d %H:%M:%S.%f")
df["End"] = pd.to_datetime(df["End"], format="%Y-%m-%d %H:%M:%S.%f")

# calculate duration of each motion event and save to CSV file
df["Duration"] = df["End"] - df["Start"]
df.to_csv("motion_times.csv", index=False)
