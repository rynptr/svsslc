import sys
sys.path.insert(0, './yolov5')

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import non_max_suppression
from torchvision import models
from PIL import Image
import time
import math
from imutils.video import FPS
from collections import deque

#yolov5_weight_file = 'rider_helmet_number_small.pt' # ... may need full path
yolov5_weight_file = 'best.pt'
#helmet_classifier_weight = 'helment_no_helmet98.6.pth'
conf_set=0.1 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = attempt_load(yolov5_weight_file, map_location=device)
cudnn.benchmark = True 
names = models.module.names if hasattr(models, 'module') else models.names

source = 'video/video2.mp4' 

save_video = False # want to save video? (when video as source)
show_video=True # set true when using video file
save_img=True  # set true when using only image file to save the image
# when using image as input, lower the threshold value of image classification


pts = [deque(maxlen=30) for _ in range(1000)]

#saveing video as output
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, frame_size)

cap = cv2.VideoCapture(source)
frame_size=(800, 480) 
#weight = cap.get(3)
#height = cap.get(4)
count = 0
center_points_prev_frame = []
avg_fps = []

tracking_objects = {}
track_id = 0

fps = FPS().start()
idx_frame = 0

while(cap.isOpened()):
	ret, frame = cap.read()
	count += 1
	t0 = time.time()
	if ret == True:
		frame = cv2.resize(frame, frame_size)  # resizing image
		#orifinal_frame = frame.copy()
	img = torch.from_numpy(frame)
	img = img.permute(2, 0, 1).float().to(device)
	img /= 255.0  
	if img.ndimension() == 3:
		img = img.unsqueeze(0)

	pred = models(img, augment=False)[0]
	pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=(1,2,3,5,7))

	detection_result = []
	rects = []
	
	center_points_cur_frame = []

	for i, det in enumerate(pred):
		if len(det): 
			for d in det: # d = (x1, y1, x2, y2, conf, cls)
				x1 = int(d[0].item())
				y1 = int(d[1].item())
				x2 = int(d[2].item())
				y2 = int(d[3].item())
				conf = round(d[4].item(), 2)
				c = int(d[5].item())

				w = x2-x1
				h = y2-y1

				x = int((x1 + x2)/2)
				y = int((y1 + y2)/2)
				
				detected_name = names[c]
				detection_result.append([x1, y1, x2, y2, conf, c])
				center_points_cur_frame.append((x, y))
				#print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1}    y1:{y1}    x2:{x2}    y2:{y2}   class:{c}')
				frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h),  (36,255,12), 1) # box

			# Only at the beginning we compare previous and current frame
			if count <= 2:
				for pt in center_points_cur_frame:
					for pt2 in center_points_prev_frame:
						distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

						if distance < 20:
							tracking_objects[track_id] = pt
							track_id += 1
			else:

				tracking_objects_copy = tracking_objects.copy()
				center_points_cur_frame_copy = center_points_cur_frame.copy()

				for object_id, pt2 in tracking_objects_copy.items():
					object_exists = False
					for pt in center_points_cur_frame_copy:
						distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

						# Update IDs position
						if distance < 20:
							tracking_objects[object_id] = pt
							object_exists = True
							if pt in center_points_cur_frame:
								center_points_cur_frame.remove(pt)
							continue

					# Remove IDs lost
					if not object_exists:
						tracking_objects.pop(object_id)

				# Add new IDs found
				for pt in center_points_cur_frame:
					tracking_objects[track_id] = pt
					track_id += 1

				#if c!=1: # if it is not head bbox, then write use putText
				#(a, b), _ = cv2.getTextSize(f'{names[c]} {str(conf)}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  
				#frame = cv2.rectangle(frame, (x1, y1 - 15), (x1 + a, y1), (36,255,12), -1)
				#frame = cv2.putText(frame, f'{names[c]} {str(conf)}', (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

			t1 = time.time()
			avg_fps.append(t1 - t0)
			cv2.putText(frame, 'frame: %d fps: %.2f ' % (idx_frame, len(avg_fps) / sum(avg_fps)),
			(20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), thickness=2)

			for object_id, pt in tracking_objects.items():
				cv2.circle(frame, pt, 2, (0, 0, 255), -1)
				cv2.putText(frame, f'{str(object_id)} {names[c]} {str(conf)}', (pt[0], pt[1] - 7), 0, 0.4, (36,255,12), 1) 

				#(a, b), _ = cv2.getTextSize(f'{names[c]} {str(conf)}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  
				#cv2.rectangle(frame, (pt[0], pt[1] - 15), (pt[0]+ a, pt[1]), (36,255,12), -1)
				#cv2.putText(frame, f'{names[c]} {str(conf)}', (pt[0], pt[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

				#cv2.rectangle(frame, (x1, y1 - 15), (x1 + a, y1), (36,255,12), -1)
				#cv2.putText(frame,f'{names[c]} {str(conf)}',(x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], 1)

				pts[object_id].append(pt)

				for j in range(1, len(pts[object_id])):
					if pts[object_id][j-1] is None or pts[object_id][j] is None:
						continue
					thickness = int(np.sqrt(64/float(j+1))*1)
					cv2.line(frame, (pts[object_id][j-1]), (pts[object_id][j]), (36,255,12), thickness)

			print("Tracking objects")
			print(tracking_objects)

			
			print("CUR FRAME LEFT PTS")
			print(center_points_cur_frame)

			print("FPS :")
			fps = len(avg_fps) / sum(avg_fps)
			print(fps)

			cv2.imshow('Frame', frame)

			# Make a copy of the points
			center_points_prev_frame = center_points_cur_frame.copy()

			key = cv2.waitKey(1)
			if key == 27:
				break

	idx_frame += 1

cap.release()
cv2.destroyAllWindows()

