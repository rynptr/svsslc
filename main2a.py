import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
		check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

import numpy as np
from torchvision import models
from PIL import Image
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import deque

cudnn.benchmark = True

class VideoTracker(object):
		def __init__(self, args):
				print('Initialize : DeepSORT & YOLO-V5')
				# ***************** Initialize ******************************************************
				self.args = args

				self.img_size = args.img_size                   # image size in detector, default is 640
				self.frame_interval = args.frame_interval       # frequency

				#self.device = select_device(args.device)
				self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
				self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

				self.useIPCam = False
						
				# ***************************** initialize YOLO-V5 **********************************

				self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
				print(f'Initialize : {self.device}')

				self.models = attempt_load(args.weights, map_location=self.device)
				self.names = self.models.module.names if hasattr(self.models, 'module') else self.models.names
				self.stride = int(self.models.stride.max())  # model stride
				self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
				if self.half:
					self.models.half()  # to FP16

				#yolov5_weight_file = 'yolov5s.pt'
				self.detector = torch.load(args.weights, map_location=self.device)['model'].float()  # load to FP32
				self.detector.to(self.device).eval()
				if self.half:
						self.detector.half()  # to FP16

				self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names
				#self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

				# create video capture ****************
				if args.display:
					cv2.namedWindow("Video_Analytics", cv2.WINDOW_NORMAL)
					cv2.resizeWindow("Video_Analytics", args.display_width, args.display_height)

				if args.cam != -1:
						print("Using webcam " + str(args.cam))
						self.vdo = cv2.VideoCapture(args.cam)
						self.dataset = LoadStreams(self.args.input_path, img_size=self.img_size, stride=self.stride)
				else:
						self.vdo = cv2.VideoCapture()
						self.dataset = LoadImages(self.args.input_path, img_size=self.img_size, stride=self.stride) 
				
				print('Done..')
				if self.device == 'cpu':
						warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

				self.pts = [deque(maxlen=30) for _ in range(1000)]

		def __enter__(self):
				# ************************* Load video from camera *************************
				if self.args.cam != -1:
						print('Camera ...')
						ret, frame = self.vdo.read()
						assert ret, "Error: Camera error"
						self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
						self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

				# ************************* Load video from file *************************
				else:
						assert os.path.isfile(self.args.input_path), "Path error"
						self.vdo.open(self.args.input_path)
						self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
						self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
						assert self.vdo.isOpened()
						#print('Done. Load video file ', self.args.input_path)

				# ************************* create output *************************
				if self.args.save_path:
						os.makedirs(self.args.save_path, exist_ok=True)
						# path of saved video and results
						self.save_video_path = os.path.join(self.args.save_path, "results.mp4")

						# create video writer
						fourcc = cv2.VideoWriter_fourcc(*self.args.fourcc)
						self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
						self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height))
						print('Done. Create output file ', self.save_video_path)

				if self.args.save_txt:
						os.makedirs(self.args.save_txt, exist_ok=True)

				return self

		def __exit__(self, exc_type, exc_value, exc_traceback):
				self.vdo.release()
				self.writer.release()
				if exc_type:
						print(exc_type, exc_value, exc_traceback)        


		def run(self):            
				yolo_time, sort_time, avg_fps = [], [], []
				t_start = time.time()

				idx_frame = 0
				last_out = None

				ct = CentroidTracker()
				listDet = ['person','bicycle','car','motorcycle','bus','truck']
				trackableObjects = {}

				totalUpPerson = 0
				totalUpBicycle = 0
				totalUpCar = 0
				totalUpMotor = 0
				totalUpBus = 0
				totalUpTruck = 0

				totalDownPerson = 0
				totalDownBicycle = 0
				totalDownCar = 0
				totalDownMotor = 0
				totalDownBus = 0
				totalDownTruck = 0

				
				counter = []

				img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
				#_ = model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once
				for path, img, im0s, vid_cap in self.dataset:
					
					img = torch.from_numpy(img).to(self.device)
					img = img.half() if self.half else img.float()  # uint8 to fp16/32
					img /= 255.0  # 0 - 255 to 0.0 - 1.0
					if img.ndimension() == 3:
						img = img.unsqueeze(0)  


					pred = self.detector(img, augment=self.args.augment)[0]  # list: bz * [ (#obj, 6)]
					# Apply NMS and filter object other than person (cls:0)
					#pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,classes=self.args.classes, agnostic=self.args.agnostic_nms)
					pred = non_max_suppression(pred, 0.4, 0.5,classes=[0,1,2,3,5,7], agnostic=False)
					rects = []
					labelObj = []
					yObj = []
					arrCentroid = []

					# **************************** Process detections **********************************
					for i, det in enumerate(pred): 


						if(self.useIPCam):
							p, s, im0 = path[i], '%g: ' % i, im0s[i].copy() #if rtsp/camera
						else:
							p, s, im0 = path, '', im0s  

						height, width, channels = im0.shape

						cv2.line(im0, (0, int(height/1.5)), (int(width), int(height/1.5)), (0, 0, 255), thickness=1)
		

						#save_path = str(Path(out) / Path(p).name)
						s += '%gx%g ' % img.shape[2:]  # print string
						gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
						if det is not None and len(det):
							# Rescale boxes from img_size to im0 size
							det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

							for c in det[:, -1].unique():
								n = (det[:, -1] == c).sum()  # detections per class
								s += '%g %s, ' % (n, self.names[int(c)])  # add to string                        
							for *xyxy, conf, cls in det:
								label = '%s %.2f' % (self.names[int(cls)], conf)
								x = xyxy
								tl = None or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
								c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
								label1 = label.split(' ')
								if label1[0] in listDet:
									box = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
									rects.append(box)
									labelObj.append(label1[0])
									cv2.rectangle(im0, c1 , c2, (36,255,12), thickness=tl, lineType=cv2.LINE_AA)
									tf = max(tl - 1, 1)  
									t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
									c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
									cv2.rectangle(im0, c1, c2, (36,255,12), -1, cv2.LINE_AA)
									cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

					detCentroid = self.generateCentroid(rects)
					print(rects)
					objects = ct.update(rects)
					#print(objects)
					
					current_count = int(0) 

					for (objectID, centroid) in objects.items():
						#print(objectID)
						#print(centroid)
						text = "ID {}".format(objectID)
						cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
						cv2.circle(im0, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

						self.pts[objectID].append(centroid)

						for j in range(1, len(self.pts[objectID])):
							if self.pts[objectID][j-1] is None or self.pts[objectID][j] is None:
								continue
							thickness = int(np.sqrt(64/float(j+1))*2)
							cv2.line(im0, (self.pts[objectID][j-1]), (self.pts[objectID][j]), (0, 0, 255), thickness)

						#print(idxDict)
						to = trackableObjects.get(objectID, None)
						#print(to)
						if to is None:
							to = TrackableObject(objectID, centroid)
						else:           
							y = [c[1] for c in to.centroids]
							print(centroid)
							print(centroid[1])
							direction = centroid[1] - np.mean(y)
							print(direction)
							to.centroids.append(centroid)
							if not to.counted: #arah up
								#print(direction)
								if direction < 0 and centroid[1] < height / 1.5 and centroid[1] > height / 1.7: ##up truble when at distant car counted twice because bbox reappear
									idx = detCentroid.tolist().index(centroid.tolist())
									if(labelObj[idx] == 'person'):
										totalUpPerson += 1
										to.counted = True
									elif(labelObj[idx] == 'bicycle'):
										totalUpBicycle += 1
										to.counted = True
									elif(labelObj[idx] == 'car'):
										totalUpCar += 1
										to.counted = True
									elif(labelObj[idx] == 'motorbike'):
										totalUpMotor += 1
										to.counted = True
									elif(labelObj[idx] == 'bus'):
										totalUpBus += 1
										to.counted = True
									elif(labelObj[idx] == 'truck'):
										totalUpTruck += 1
										to.counted = True

								elif direction > 0 and centroid[1] > height / 1.5:  #arah down
									idx = detCentroid.tolist().index(centroid.tolist())
									if(labelObj[idx] == 'person'):
										totalDownPerson += 1
										to.counted = True
									elif(labelObj[idx] == 'bicycle'):
										totalDownBicycle += 1
										to.counted = True
									elif(labelObj[idx] == 'car'):
										totalDownCar += 1
										to.counted = True
									elif(labelObj[idx] == 'motorbike'):
										totalDownMotor += 1
										to.counted = True
									elif(labelObj[idx] == 'bus'):
										totalDownBus += 1
										to.counted = True
									elif(labelObj[idx] == 'truck'):
										totalDownTruck += 1
										to.counted = True

						#if centroid[1] <= int(3*height/6+height/20) and centroid[1] >= int(3*height/6-height/20):
							#idx = detCentroid.tolist().index(centroid.tolist())
							#if labelObj[idx] == 'car' or labelObj[idx] == 'truck':
								#counter.append(int(objectID))
								#current_count += 1  

						trackableObjects[objectID] = to

						#cv2.putText(im0, "Current Vehicle Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
						#cv2.putText(im0, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

						cv2.putText(im0, 'Down Person : ' + str(totalDownPerson), (int(width * 0.7) , int(height * 0.05)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
						cv2.putText(im0, 'Down bicycle : ' + str(totalDownBicycle), (int(width * 0.7) , int(height * 0.1)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
						cv2.putText(im0, 'Down car : ' + str(totalDownCar), (int(width * 0.7) , int(height * 0.15)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
						cv2.putText(im0, 'Down motorbike : ' + str(totalDownMotor), (int(width * 0.7) , int(height * 0.2)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,120), 2)
						cv2.putText(im0, 'Down bus : ' + str(totalDownBus), (int(width * 0.7) , int(height * 0.25)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
						cv2.putText(im0, 'Down truck : ' + str(totalDownTruck), (int(width * 0.7) , int(height * 0.3)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

						cv2.putText(im0, 'Up Person : ' + str(totalUpPerson), (int(width * 0.02) , int(height * 0.05)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
						cv2.putText(im0, 'Up bicycle : ' + str(totalUpBicycle), (int(width * 0.02) , int(height * 0.1)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
						cv2.putText(im0, 'Up car : ' + str(totalUpCar), (int(width * 0.02) , int(height * 0.15)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (136,255,12), 2)
						cv2.putText(im0, 'Up motorbike : ' + str(totalUpMotor), (int(width * 0.02) , int(height * 0.2)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
						cv2.putText(im0, 'Up bus : ' + str(totalUpBus), (int(width * 0.02) , int(height * 0.25)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)
						cv2.putText(im0, 'Up truck : ' + str(totalUpTruck), (int(width * 0.02) , int(height * 0.3)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.8, (136,255,12), 2)


						# display on window ******************************
						if self.args.display:
								cv2.imshow("Video_Analytics", im0)
								if cv2.waitKey(1) == ord('q'):  # q to quit
										cv2.destroyAllWindows()
										break
						# save to video file *****************************
						if self.args.save_path:
								self.writer.write(im0)

					idx_frame += 1

				#print('Avg YOLO time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),sum(sort_time)/len(sort_time)))
				t_end = time.time()
				print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

		

		def generateCentroid(self,rects):
			inputCentroids = np.zeros((len(rects), 2), dtype="int")
			for (i, (startX, startY, endX, endY)) in enumerate(rects):
				cX = int((startX + endX) / 2.0)
				cY = int((startY + endY) / 2.0)
				inputCentroids[i] = (cX, cY)
			return inputCentroids                                


if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		# input and output
		parser.add_argument('--input_path', type=str, default='0', help='source')  # file/folder, 0 for webcam
		parser.add_argument('--save_path', type=str, default='output/', help='output folder')  # output folder
		parser.add_argument("--frame_interval", type=int, default=2)
		parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
		parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
		parser.add_argument('--save_txt', default='output/predict/', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

		# camera only
		parser.add_argument("--display", action="store_true")
		parser.add_argument("--display_width", type=int, default=800)
		parser.add_argument("--display_height", type=int, default=480)
		parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

		# YOLO-V5 parameters
		parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
		parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
		parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
		parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
		parser.add_argument('--classes', nargs='+', type=int, default=[0,1,2,3,5,7], help='filter by class')
		parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
		parser.add_argument('--augment', action='store_true', help='augmented inference')

		# deepsort parameters
		parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")

		args = parser.parse_args()
		args.img_size = check_img_size(args.img_size)
		#print(args)

		with VideoTracker(args) as vdo_trk:
				vdo_trk.run()

