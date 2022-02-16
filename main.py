import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
		check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_synchronized
from yolov5.utils.datasets import letterbox

from utils_ds.parser import get_config
from deep_sort import build_tracker

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
import json
import requests

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
						

				# ***************************** initialize DeepSORT **********************************
				cfg = get_config()
				cfg.merge_from_file(args.config_deepsort)

				use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
				self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

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


				# ***************************** create video capture ****************
				if self.args.cam != -1:
						print("Using webcam " + str(self.args.cam))
						self.vdo = cv2.VideoCapture(self.args.cam)
				else:
						self.vdo = cv2.VideoCapture(self.args.input_path)

				print('Done..')
				if self.device == 'cpu':
						warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

				self.pts = [deque(maxlen=40) for _ in range(9999)]
				self.counter = []
				self.selection_dict = {'img': None, 'points selected': []}
				self.model_image_size = (750, 750)
				self.ct = CentroidTracker()
				self.trackableObjects = {}
				

				self.totalUpCar = 0
				self.totalDownCar = 0
				self.totalUpMotor = 0
				self.totalDownMotor = 0
				self.totalUpTruck = 0
				self.totalDownTruck = 0
				self.totalUpBus = 0
				self.totalDownBus = 0

		def __enter__(self):
				# ************************* Load video from camera *************************
				if self.args.cam != -1:
						print('Camera ...')
						print("Using webcam " + str(self.args.cam))
						self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
						self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

				# ************************* Load video from file *************************
				else:
						assert os.path.isfile(self.args.input_path), "Path error"
						self.vdo.open(self.args.input_path)
						self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
						self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
						assert self.vdo.isOpened()
						print('Done. Load video file ', self.args.input_path)

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
			
				ct = CentroidTracker()
				yolo_time, sort_time, avg_fps = [], [], []
				t_start = time.time()
				idx_frame = 0
				last_out = None
				ret, image = self.vdo.read()
				img0 = self.letterbox_image(image, tuple(reversed(self.model_image_size)))
				if self.select_quadrilateral_from(img0) == -1:
					print("You must select 4 points")
					self.vdo.release()
					exit(0)

				self.quad_as_contour = self.selection_dict['points selected'].reshape((-1, 1, 2))
				
				while self.vdo.grab():
					# ***************************** Inference *********************************************************************
					t0 = time.time() 
					#ret, img0 = self.vdo.read()
					_, image = self.vdo.retrieve()
					# ***************************** enviroment *********************************************************************

					img0 = self.letterbox_image(image, tuple(reversed(self.model_image_size)))
					

					if idx_frame % self.args.frame_interval == 0:
							outputs, confs, yt, st = self.image_track(img0)        # (#ID, 5) x1,y1,x2,y2,id
							last_out = outputs
							#print(outputs)
							yolo_time.append(yt)
							sort_time.append(st)
							print('Frame %d Done. YOLO-time:(%.3fs) SORT-time:(%.3fs)' % (idx_frame, yt, st))
					else:
							outputs = last_out  # directly use prediction in last frames

					t1 = time.time()
					avg_fps.append(t1 - t0)

					# add FPS information on output video
					text_scale = max(1, img0.shape[1] // 1600)
					cv2.putText(img0, 'frame: %d fps: %.2f ' % (idx_frame, len(avg_fps) / sum(avg_fps)), (20, 20 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), thickness=1)

					# ***************************** post-processing ***************************************************************
					# *****************************  visualize bbox  ********************************
					if len(outputs) > 0:
							bbox_xyxy = outputs[:, :4]
							identities = outputs[:, -2]
							clss = outputs[:, -1]
							img0 =  self.draw_boxes(img0, bbox_xyxy, confs, identities, clss)  # BGR
							#print(img0.shape[1])
							

					# ***************************** display on window ******************************
					if self.args.display:
							#cv2.namedWindow("Video_Analytics", cv2.WINDOW_NORMAL)
							#cv2.resizeWindow("Video_Analytics", self.args.display_width, self.args.display_height)
							cv2.imshow("Video_Analytics", img0)
							if cv2.waitKey(1) == ord('q'):  # q to quit
									cv2.destroyAllWindows()
									break
					# ***************************** save to video file *****************************
					if self.args.save_path:
							self.writer.write(img0)

					idx_frame += 1

				#print('Avg YOLO time (%.3fs), Sort time (%.3fs) per frame' % (sum(yolo_time) / len(yolo_time),sum(sort_time)/len(sort_time)))
				t_end = time.time()
				print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

		
		def image_track(self, im0):
				"""
				:param im0: original image, BGR format
				:return:
				"""
				# ***************************** preprocess ************************************************************
				# Padded resize
				img = letterbox(im0, new_shape=self.img_size)[0]

				# Convert
				img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
				img = np.ascontiguousarray(img)

				# numpy to tensor
				img = torch.from_numpy(img).to(self.device)
				img = img.half() if self.half else img.float()  # uint8 to fp16/32
				img /= 255.0  # 0 - 255 to 0.0 - 1.0
				if img.ndimension() == 3:
						img = img.unsqueeze(0)
				s = '%gx%g ' % img.shape[2:]    # print string
				# ***************************** Detection time *********************************************************
				# Inference
				t1 = time_synchronized()
				#pred = self.models(img, augment=self.args.augment)[0]
				pred = self.detector(img, augment=self.args.augment)[0]  # list: bz * [ (#obj, 6)]
				# Apply NMS and filter object other than person (cls:0)
				pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,
																	 classes=self.args.classes, agnostic=self.args.agnostic_nms) 
				

				t2 = time_synchronized()

				# ***************************** get all obj ************************************************************
				det = pred[0]  # for video, bz is 1
				
				if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls
						# Rescale boxes from img_size to original im0 size
						det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
						# Print results. statistics of number of each obj
						for c in det[:, -1].unique():
							n = (det[:, -1] == c).sum()  # detections per class
							s += '%g %ss, ' % (n, self.names[int(c)])  # add to string
						for *xyxy, conf, cls in det:
							label = '%s %.2f' % (self.names[int(cls)], conf)
							x = xyxy
							tl = None or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
							c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
							box = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
							#print(xyxy)

						bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
						
						confs = det[:, 4].cpu()
						clas = det[:, 5].cpu()
						#print(confs)   

						# ****************************** deepsort ****************************
						outputs = self.deepsort.update(bbox_xywh, confs, clas, im0)
				else:
						outputs = torch.zeros((0, 5))   
						confs = 0

				t3 = time.time()

				return outputs, confs, t2-t1, t3-t2                                

		def draw_boxes(self, img, bbox, confs=None, identities=None, clss=None, offset=(0,0)):
				rects = []
				stat_palang = 'palang pintu terbuka'
				colour_barrier = ()
				current_count = int(0)
				height, width, channels = img.shape

				pts1 = self.quad_as_contour[0][0]
				pts2 = self.quad_as_contour[1][0]
				pts3 = self.quad_as_contour[2][0]
				pts4 = self.quad_as_contour[3][0]

				for i, (box, conf) in enumerate(zip(bbox, confs)):
					x1,y1,x2,y2 = [int(i) for i in box]
					x1 += offset[0]
					x2 += offset[0]
					y1 += offset[1]
					y2 += offset[1]

					# box text and bar
					id = int(identities[i]) if identities is not None else 0
					idclss = int(clss[i]) if clss is not None else 0    
					color = self.compute_color_for_labels(id)
					#label = '{}{:d}'.format("", id)
					label = f'{id} {self.names[idclss]} {conf:.2f}'
						

					x = int((x1 + x2)/2)
					y = int((y1 + y2)/2)    
					
					center = (x,y)

					box = (x1, y1 , x2, y2)
					rects.append(box)
					objects = self.ct.update(rects)

					self.pts[id].append(center)

					for j in range(1, len(self.pts[id])):
						if self.pts[id][j-1] is None or self.pts[id][j] is None:
							continue
						thickness = int(np.sqrt(64/float(j+1))*1.5)
						cv2.line(img, (self.pts[id][j-1]), (self.pts[id][j]), color, thickness)


					#================================= bounding box =========================================
					(a, b), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  
					cv2.rectangle(img,(x1, y1),(x2,y2),color,1)
		
					#cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
					cv2.rectangle(img, (x1, y1 - 13), (x1 + a, y1), color, -1)
					cv2.putText(img,label,(x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
						  
					cv2.circle(img,center, 2, (0, 0, 255), -1)

					#print(stat)

					cv2.polylines(img, [self.quad_as_contour], True, (0,0,255), thickness=1)



					#================================= cek barrier =========================================
					cv2.line(img, (0, int(height / 2.5)), (int(width), int(height / 2.5)), (0, 0, 255), thickness=1)

					nameclass = self.names[idclss]
					if(nameclass == 'barrier' and y < height / 2.5 ):
						#cv2.circle(img,(x1,y1), 5, (0, 0, 255), -1)
						stat_palang = 'palang pintu terbuka'
						#print(stat)
						colour_barrier = (36,255,12)
					elif(nameclass == 'barrier' and y > height / 2.5 ):
						stat_palang = 'palang pintu tertutup'
						#print(stat)
						colour_barrier = (0,0,255)

					#stat_palang = stat
					#print(stat_palang)
					center_y = (y)

					overlay = img.copy()
					cv2.fillPoly(img, [self.quad_as_contour], color = colour_barrier)
					alpha = 0.97  # Transparency factor.
					img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

					confidence = float("{:.2f}".format(conf))

					#================================= area yellow box =========================================
					if(stat_palang =='palang pintu terbuka'):
						if cv2.pointPolygonTest(self.selection_dict['points selected'], center, measureDist=False) >=0 :
							#print(id)
							self.counting(img, id, objects, center, nameclass, confidence)
							self.counter.append(int(id))
							current_count += 1
								
					elif(stat_palang =='palang pintu tertutup'):
						if cv2.pointPolygonTest(self.selection_dict['points selected'], center, measureDist=False) >=0 :
							#print(id)
							self.counting_pelanggaran(img, id, objects, x1, x2, y1, y2, nameclass,confidence)
							self.counter.append(int(id))
							current_count += 1
													
					
					
				text_scale = max(1, img.shape[1] // 1600) 
				cv2.putText(img, 'Down Mobil  : ' + str(self.totalDownCar), (600, 20 + text_scale),cv2.FONT_HERSHEY_DUPLEX , 0.4, (36,255,12), 1)
				cv2.putText(img, 'Up Mobil : ' + str(self.totalUpCar), (600, 40 + text_scale),cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1) 
				cv2.putText(img, 'Down Motor  : ' + str(self.totalDownMotor), (600, 60 + text_scale),cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
				cv2.putText(img, 'Up Motor : ' + str(self.totalUpMotor), (600, 80 + text_scale),cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)  
				cv2.putText(img, 'Down Truck  : ' + str(self.totalDownTruck), (600, 100 + text_scale),cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
				cv2.putText(img, 'Up Truck : ' + str(self.totalUpTruck), (600, 120 + text_scale),cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)  
				cv2.putText(img, 'Down Bus  : ' + str(self.totalDownBus), (600, 140 + text_scale),cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
				cv2.putText(img, 'Up Bus : ' + str(self.totalUpBus), (600, 160 + text_scale),cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)    

				#cv2.line(img, (0, int(height/1.8)), (int(width), int(height/1.8)), (0, 0, 255), thickness=1)
				#cv2.line(img0, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)
				#cv2.line(img0, (0, int(3*height/6+height/5)), (width, int(3*height/6+height/5)), (0, 255, 0), thickness=2)

				#cv2.polylines(img0, [pts1], True, (45, 255, 0), thickness=2)
				#cv2.polylines(img0, [pts2], True, (45, 255, 0), thickness=2)
				#cv2.polylines(img, [self.quad_as_contour], True, (36,255,12), thickness=1)
				#cv2.line(img0, (quad_as_contour[0]), (quad_as_contour[1]), (255, 255, 255), thickness=2)
				text_scale = max(1, img.shape[1] // 1600) 
				total_count = len(set(self.counter))    
				cv2.putText(img, "Total Obstacle Count: " + str(current_count), (20, 40 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
				cv2.putText(img, "Total Vehicle Count: " + str(total_count), (20, 60 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
				cv2.putText(img, "Status barrier: " + stat_palang, (20, 80 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)


				

				#cv2.line(img, (pts1[0], pts1[1]), (pts2[0], pts2[1]), (36,255,12), thickness=2)
				#cv2.line(img, (pts3[0], pts3[1]), (pts4[0], pts4[1]), (36,255,12), thickness=2)

				return img
		

		def counting(self, img, id, objects, center, nameclass, conf):
					
					#cv2.fillPoly(img, [self.quad_as_contour], 255)
					#t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
					#if center_y <= int(3*height/6+height/5) and center_y >= int(3*height/6-height/20):
					#idx = detCentroid.tolist().index(centroid.tolist())
					#if labelObj[idx] == 'car' or labelObj[idx] == 'truck':
					#self.countingkendaraan(img,objects,self.quad_as_contour)
					
					
					for (objectID, centroid) in objects.items():
						#print(centroid[1])
						center_ = (int(centroid[0]),int(centroid[1]))
						to = self.trackableObjects.get(objectID, None)
						#print(to)
						if to is None:
							to = TrackableObject(objectID, centroid)
						else:           
							y = [c[1] for c in to.centroids]
							direction = centroid[1] - np.mean(y)
							to.centroids.append(centroid)
							if not to.counted: #arah up
								#if direction < 0 and centroid[1] < height / 1.8 and centroid[1] > height / 2:
								if direction < 0 and cv2.pointPolygonTest(self.selection_dict['points selected'], center_, measureDist=False) >=0 : 
								#and centroid[1] > 357: ##up truble when at distant car counted twice because bbox reappear
									#idx = detCentroid.tolist().index(centroid.tolist())
									if(nameclass == 'car'):
										self.totalUpCar += 1
										to.counted = True
									elif(nameclass == 'motorbike'):
										self.totalUpMotor += 1
										to.counted = True
									elif(nameclass == 'truck'):
										self.totalUpTruck += 1
										to.counted = True
									elif(nameclass == 'Bus'):
										self.totalUpBus += 1
										to.counted = True


									send_json = {
										"lokasi":'perlintasan sebidang andir',
										"jenis_objek":nameclass,
										"level_conf":conf,
										"tanggal":time.strftime("%Y-%m-%d"),
										"waktu":time.strftime("%H:%M:%S"),
										"arah_objek":'1', 
										"id_objek":id,  
									}
									#print(id)
									print(send_json)

									try:
										headers =  {"Content-Type":"application/json"}
										requests.post(url='http://localhost:4000/data_traffic', json=send_json, headers=headers)
									except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
										requests.exceptions.URLRequired) as e:
										print(e)


								#elif direction > 0 and centroid[1] > height / 1.8:  #arah down
								elif direction > 0 and cv2.pointPolygonTest(self.selection_dict['points selected'], center_, measureDist=False) >=0 : 
									#idx = detCentroid.tolist().index(centroid.tolist())
									if(nameclass == 'car'):
										self.totalDownCar += 1
										to.counted = True
									elif(nameclass == 'motorbike'):
										self.totalDownMotor += 1
										to.counted = True
									elif(nameclass == 'truck'):
										self.totalDownTruck += 1
										to.counted = True
									elif(nameclass == 'Bus'):
										self.totalDowBus += 1
										to.counted = True

									send_json = {
										"lokasi":'perlintasan sebidang andir',
										"jenis_objek":nameclass,
										"level_conf":conf,
										"tanggal":time.strftime("%Y-%m-%d"),
										"waktu":time.strftime("%H:%M:%S"),
										"arah_objek":'2', 
										"id_objek":id,   
									}
									print(send_json)
									try:
										headers =  {"Content-Type":"application/json"}
										requests.post(url='http://localhost:4000/data_traffic', json=send_json, headers=headers)
									except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
										requests.exceptions.URLRequired) as e:
										print(e)


						self.trackableObjects[objectID] = to


		def counting_pelanggaran(self, img, id, objects, x1, x2, y1, y2, nameclass,conf):
				dir_output ='output'

				for (objectID, centroid) in objects.items():
						#print(centroid[1])
						center_ = (int(centroid[0]),int(centroid[1]))
						to = self.trackableObjects.get(objectID, None)
						#print(to)
						if to is None:
							to = TrackableObject(objectID, centroid)
						else:           
							y = [c[1] for c in to.centroids]
							direction = centroid[1] - np.mean(y)
							to.centroids.append(centroid)
							if not to.counted: #arah up
								#if direction < 0 and centroid[1] < height / 1.8 and centroid[1] > height / 2:
								if direction < 0 and cv2.pointPolygonTest(self.selection_dict['points selected'], center_, measureDist=False) >=0 : 
								#and centroid[1] > 357: ##up truble when at distant car counted twice because bbox reappear
									#idx = detCentroid.tolist().index(centroid.tolist())
									crop_img = img[y1:y2+5, x1:x2+5]
									p = os.path.sep.join([dir_output, time.strftime("%Y%m%d")+"{}.png".format(str(id).zfill(5))])
									print(p)
									cv2.imwrite(p, crop_img)
									to.counted = True

									json_pelanggaran = {
										"lokasi":'perlintasan sebidang andir',
										"jenis_pelanggaran":'menerobos palang perlintasan sebidang',
										"objek_pelanggaran":nameclass,
										"level_conf":conf,
										"tanggal":time.strftime("%Y-%m-%d"),
										"waktu":time.strftime("%H:%M:%S"),
										"arah_objek":'1',
										"id_objek":id,
										"image_objek":p
									}

									print(json_pelanggaran)
									try:
										headers =  {"Content-Type":"application/json"}
										requests.post(url='http://localhost:4000/data_pelanggaran', json=json_pelanggaran, headers=headers)
									except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
										requests.exceptions.URLRequired) as e:
										print(e)

								elif direction > 0 and cv2.pointPolygonTest(self.selection_dict['points selected'], center_, measureDist=False) >=0 : 

									crop_img = img[y1:y2+5, x1:x2+5]
									p = os.path.sep.join([dir_output, time.strftime("%Y%m%d")+"{}.png".format(str(id).zfill(5))])
									print(p)
									cv2.imwrite(p, crop_img)
									to.counted = True

									json_pelanggaran = {
										"lokasi":'perlintasan sebidang andir',
										"jenis_pelanggaran":'menerobos palang perlintasan sebidang',
										"objek_pelanggaran":nameclass,
										"level_conf":conf,
										"tanggal":time.strftime("%Y-%m-%d"),
										"waktu":time.strftime("%H:%M:%S"),
										"arah_objek":'2',
										"id_objek":id,
										"image_objek":p
									}

									print(json_pelanggaran)
									try:
										headers =  {"Content-Type":"application/json"}
										requests.post(url='http://localhost:4000/data_pelanggaran', json=json_pelanggaran, headers=headers)
									except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
										requests.exceptions.URLRequired) as e:
										print(e)


						self.trackableObjects[objectID] = to

		def compute_color_for_labels(self, label):
				"""
				Simple function that adds fixed color depending on the class
				"""
				palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
				color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
				return tuple(color)

		def letterbox_image(self, image, size):
				'''resize image with unchanged aspect ratio using padding'''
				ih, iw = image.shape[:2]
				h, w = size
				scale = min(w/iw, h/ih)
				nw = int(iw*scale)
				nh = int(ih*scale)

				new_image = cv2.resize(image, (nw,nh), fx=0.75, fy=0.75)
				#image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
				#new_image = np.zeros((w, h, 3), np.uint8)
				#new_image[:,:] = (128, 128, 128)
				#new_image[(h-nh)//2:(h-nh)//2 +nh, (w-nw)//2:(w-nw)//2 + nw] = image
				return new_image

		def select_point(self, event, x, y, flags,param):
			if event == cv2.EVENT_LBUTTONDOWN:
				cv2.circle(self.selection_dict['img'],(x,y), 5, (0, 255, 0), -1)
				self.selection_dict['points selected'].append([x, y])       


		def select_quadrilateral_from(self, image):
			self.selection_dict['img'] = image
			cv2.namedWindow('selection frame')
			cv2.setMouseCallback('selection frame', self.select_point)

			while(1):
				cv2.imshow('selection frame', image)
				if cv2.waitKey(20) & 0xFF == 27:
					break
				if len(self.selection_dict['points selected']) >= 4:
					break

			cv2.destroyAllWindows()
			if len(self.selection_dict['points selected']) != 4:
				return -1

			self.selection_dict['points selected'].sort(key=lambda point: point[1])

			"""
				After sorting with y coordinate as key, the first two points represent the top two 
				points of the quadrilateral, and the next two represent the bottom two.
			"""

			if self.selection_dict['points selected'][0][0] > self.selection_dict['points selected'][1][0]:
				self.selection_dict['points selected'][0], self.selection_dict['points selected'][1] = \
				self.selection_dict['points selected'][1], self.selection_dict['points selected'][0]

			if self.selection_dict['points selected'][3][0] > self.selection_dict['points selected'][2][0]:
				self.selection_dict['points selected'][3], self.selection_dict['points selected'][2] = \
				self.selection_dict['points selected'][2], self.selection_dict['points selected'][3]

			self.selection_dict['points selected'] = np.array(self.selection_dict['points selected'], dtype=np.int32)
			return 1


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
		parser.add_argument("--display_height", type=int, default=600)
		parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

		# YOLO-V5 parameters
		parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
		parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
		parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
		parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
		parser.add_argument('--classes', nargs='+', type=int, default=[0,1,2,3,4,5,6], help='filter by class')
		parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
		parser.add_argument('--augment', action='store_true', help='augmented inference')

		# deepsort parameters
		parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")

		args = parser.parse_args()
		args.img_size = check_img_size(args.img_size)
		#print(args)

		with VideoTracker(args) as vdo_trk:
				vdo_trk.run()

