import sys
sys.path.insert(0, './yolov5')

#from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.general import (
        LOGGER, check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.datasets import letterbox, LoadStreams

#from utils_ds.parser import get_config
#from deep_sort import build_tracker

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

import numpy as np
from PIL import Image
import argparse
import os
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from collections import deque
import json
import requests
import datetime
import csv


cudnn.benchmark = True

class VideoTracker(object):
        def __init__(self, args):
                print('Initialize : YOLO-V5')
                # ***************** Initialize ******************************************************
                self.args = args

                self.img_size = args.img_size                   # image size in detector, default is 640
                self.frame_interval = args.frame_interval       # frequency

                #self.device = select_device(args.device)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
                print(f'Initialize : {self.device}')
                
                self.augment = args.augment = False
                # ***************************** initialize DeepSORT **********************************
                #cfg = get_config()
                #cfg.merge_from_file(args.config_deepsort)

                #use_cuda = self.device.type != 'cpu' and torch.cuda.is_available()
                #self.deepsort = build_tracker(cfg, use_cuda=use_cuda)

                # ***************************** initialize YOLO-V5 **********************************

                #self.models = attempt_load(args.weights, map_location=self.device)
                #self.names = self.models.module.names if hasattr(self.models, 'module') else self.models.names
                #self.stride = int(self.models.stride.max())  # model stride
                #self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
                #if self.half:
                    #self.models.half()  # to FP16

                self.model_palang = 'model/model_palang_416_16_100.pt'
                #self.model_palang = 'model/yolov5s/prune/416/model_palang_416_32_100_0.00005_0.45_ft_best.pt'
                self.detector_palang = torch.load(self.model_palang, map_location=self.device)['model'].float()  # load to FP32
                self.detector_palang.to(self.device).eval()
                if self.half:
                        self.detector_palang.half()  # to FP16

               # self.names = self.detector_palang.module.names if hasattr(self.detector, 'module') else self.detector.names

                self.model_object = 'model/perlintasansebidang_data_v5_416_16_100.pt' 
                #self.model_object = 'model/yolov5s/prune/model_perlintasansebidang_data_v2.1_pruning_0.00005_640_0.45ft_best.pt' 
                self.detector_object = torch.load(self.model_object, map_location=self.device)['model'].float()  # load to FP32
                self.detector_object.to(self.device).eval()
                if self.half:
                        self.detector_object.half()  # to FP16


                # ***************************** create video capture ****************
                if self.args.cam != -1:
                        print("Using webcam " + str(self.args.cam))
                        self.vdo = cv2.VideoCapture(self.args.cam)
                else:
                        self.vdo = cv2.VideoCapture(self.args.input_path)
                        #self.vdo = LoadStreams(self.args.input_path)

                print('Done..')
                if self.device == 'cpu':
                        warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

                self.pts = [deque(maxlen=40) for _ in range(9999)]
                self.counter = []
                self.selection_dict = {'img': None, 'points selected': []}
                self.selection_dict2 = {'img': None, 'points selected': []}
                self.model_image_size = (self.args.display_width,self.args.display_height)
                #self.model_image_size = (640,480)
                self.ct = CentroidTracker()
                self.trackableObjects = {}

                self.totalpelanggaran = 0
                self.start_palang_tertutup = ''

                
                self.URL_API_lalulintas = 'http://localhost:3000/lalulintas'
                self.URL_API_Pelanggaran = 'http://localhost:3000/inputPelanggaran'

                self.frame_rate = 0.0

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
                #if self.args.save_path:
                        #os.makedirs(self.args.save_path, exist_ok=True)
                        # path of saved video and results
                        #self.save_video_path = os.path.join(self.args.save_path, "results2.mp4")

                        # create video writer
                        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        #self.writer = cv2.VideoWriter(self.save_video_path, fourcc,
                        #self.vdo.get(cv2.CAP_PROP_FPS), (self.im_width, self.im_height), True)
                        #print('Done. Create output file ', self.save_video_path)


                if self.args.save_txt:
                        os.makedirs(self.args.save_txt, exist_ok=True)

                return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
                self.vdo.release()
                #self.writer.release()
                if exc_type:
                        print(exc_type, exc_value, exc_traceback)        


        def run(self):             
            
                deteksi_objek1_time, klasifikasi1_time, deteksi_objek2_time, klasifikasi2_time, fps, avg_perframe, avg_perframe2, avg_perframe3, avg_perframe4, fps_arr = [], [], [], [], [], [], [], [], [], []
                idx_frame = 0
                t_start = time.time()
                last_out = None
                ret, image = self.vdo.read()

                img0 = self.letterbox_image(image, tuple(reversed(self.model_image_size)))
                
                
                if self.select_quadrilateral_from(img0) == -1:
                    print("You must select 4 points")
                    self.vdo.release()
                    exit(0)


                        
                self.quad_as_contour = self.selection_dict['points selected'].reshape((-1, 1, 2))
                img0 = self.crop_frame(img0)
                A, B, C = self.split_list(self.quad_as_contour)

                print(self.quad_as_contour)
                #self.quad_as_contour2 = self.selection_dict2['points selected'].reshape((-1, 1, 2))
                # used to record the time when we processed last frame
                prev_frame_time = 0
                 
                # used to record the time at which we processed current frame
                new_frame_time = 0
                
                while self.vdo.grab():
                    # ***************************** Inference *********************************************************************
                    t0 = time.time() 
                    #ret, img0 = self.vdo.read()
                    _, image = self.vdo.retrieve()
                    # ***************************** enviroment *********************************************************************

                    img0 = self.letterbox_image(image, tuple(reversed(self.model_image_size)))
                    

                    if idx_frame % self.args.frame_interval == 0:
                            #outputs, confs, names, yt, st = self.image_track(img0, self.detector_palang)        # (#ID, 5) x1,y1,x2,y2,id

                            results, confs, names, deteksi_objek1, klasifikasi1 = self.image_track(img0, self.detector_palang) 


                            last_out = results
                            #print(outputs)
                            #yolo_time.append(yt)
                            #sort_time.append(st)
                            #length = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
                            frame_count = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
                            deteksi_objek1_time.append(deteksi_objek1)
                            klasifikasi1_time.append(klasifikasi1)
                            print('Frame %d / %f' % (idx_frame, frame_count))


                            
                            #print('Deteksi objek model 1 - time:(%.3fs) Klasifikasi model 1 - time:(%.3fs) ' % (deteksi_objek1, klasifikasi1))

                            
                            
                    else:
                            outputs = last_out  # directly use prediction in last frames
                   
                    # ***************************** post-processing ***************************************************************
                    # *****************************  visualize bbox  ********************************
                    if len(results) > 0:
                            #bbox_xyxy = outputs[:, :4]
                            #identities = outputs[:, -2]
                            #clss = outputs[:, -1]
                            img0, deteksi_objek2, klasifikasi2 = self.draw_boxes(img0, results, confs, names)  # BGR
                            #print(img0.shape[1])


                            deteksi_objek2_time.append(deteksi_objek2)
                            klasifikasi2_time.append(klasifikasi2)

                            idx_frame += 1

                            #add FPS information on output video
                            t1 = time.time() 
                            
                            #fps.append(t1 - t0)
                            #fps_perframe = (len(fps) / sum(fps))
                            #print('elapsed time : (%.3fs)' % (t1 - t0))
                            #print('FPS 1 : %.2f ' % fps_perframe)
                            #avg_perframe.append(fps_perframe)


                            frame_rate = ( self.frame_rate + (1./(time.time()-t0)) ) 
                            fps_arr.append((time.time(), frame_rate))
                            print("FPS 1: %.2f " % frame_rate)
                            avg_perframe.append(frame_rate)


                             # convert inference time in milliseconds to frames per second as well?
                            fpsm = 1 / (time.time() - t0)
                            LOGGER.info(f'FPS 2: {fpsm:.1f}')
                            avg_perframe2.append(fpsm)

            
                            #dt = time.strftime("%I:%M:%S")
                            #print(dt)
                            #height = self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            #width = self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH)
                            #print(height)
                            #print(width)

                            
                            print('model detection time 1 (%.3fs)' % (sum(deteksi_objek1_time) / len(deteksi_objek1_time)))
                            print('model detection time 2 (%.3fs)' % (sum(deteksi_objek2_time) / len(deteksi_objek2_time)))


                            avg_fps1 = (sum(avg_perframe) / len(avg_perframe))
                            print('Avg FPS 1 : %.2f ' % (avg_fps1))

                            avg_fps2 = (sum(avg_perframe2) / len(avg_perframe2))
                            print('Avg FPS 2 : %.2f ' % (avg_fps2))



                            
                                 
                            text_scale = max(1, img0.shape[1] // 1600)
                            cv2.putText(img0, 'frame: %d fps: %.2f ' % (idx_frame, fpsm), (20, 20 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), thickness=1)
                            #cv2.putText(img0, 'FPS: %.2f ' % (int(1 / (t1 - t0))), (20, 80 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), thickness=1)

                    # ***************************** display on window ******************************
                    if self.args.display:
                            #cv2.namedWindow("Video_Analytics", cv2.WINDOW_NORMAL)
                            #cv2.resizeWindow("Video_Analytics", self.args.display_width, self.args.display_height)
                            cv2.imshow("Video_Analytics", img0)
                            if cv2.waitKey(1) == ord('q'):  # q to quit
                                    cv2.destroyAllWindows()
                                    break
                    # ***************************** save to video file *****************************
                    #if self.args.save_path:
                            #self.writer.write(img0)

                
                header = ['fps']

                with open('hasil_fps.csv', 'w', encoding='UTF8', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
         
                    for timestamp, ff in fps_arr:
                        writer.writerow([f'{ff:.2f}'])
                
                t_end = time.time()
                           
                #avg_fps = (sum(fps_perframe) / len(fps_perframe))
                print('Sum model 1 detection time (%.3fs), SUm model 2 detection time (%.3fs)' % (sum(deteksi_objek1_time),sum(deteksi_objek2_time)))
                print('Avg model 1 detection time (%.3fs), Avg model 2 detection time (%.3fs) per frame' % (sum(deteksi_objek1_time) / len(deteksi_objek1_time),sum(deteksi_objek2_time) / len(deteksi_objek2_time)))
                print(f'Total pelanggaran : {self.totalpelanggaran}') 
                print('Avg FPS 1 : %.2f ' % (avg_fps1))
                print('Avg FPS 2 : %.2f ' % (avg_fps2))
                print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))
                

        
        def image_track(self, im0, model):
                """
                :param im0: original image, BGR format
                :return:
                """
                # ***************************** preprocess ************************************************************
                print(f'load model : {self.model_palang}')
                
                names = model.module.names if hasattr(model, 'module') else model.names

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
                t1 = time_sync()
                #pred = self.models(img, augment=self.args.augment)[0]
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False
                pred = model(img, augment=self.augment, visualize=visualize)[0]  # list: bz * [ (#obj, 6)]
                # Apply NMS and filter object other than person (cls:0)
                pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,classes=self.args.classes, agnostic=self.args.agnostic_nms) 
                

                t2 = time_sync()



                # ***************************** get all obj ************************************************************
                det = pred[0]  # for video, bz is 1

                detection_result = []
                
                #for i, det in enumerate(pred):
                        #if len(det): 
                            #for d in det: # d = (x1, y1, x2, y2, conf, cls)
                                #x1 = int(d[0].item())
                                #y1 = int(d[1].item())
                                #x2 = int(d[2].item())
                                #y2 = int(d[3].item())
                                #conf = round(d[4].item(), 2)
                                #c = int(d[5].item())


                                #detected_name = names[c]

                                #bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
                                #frame = cv2.rectangle(im0, (x1, y1), (x2, y2), (255,0,0), 1) # box
                
                if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls
                            # Rescale boxes from img_size to original im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            # Print results. statistics of number of each obj
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        for *xyxy, conf, cls in det:
                            label = '%s %.2f' % (names[int(cls)], conf)
                            x = xyxy
                            tl = None or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
                            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                            box = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
                                #print(xyxy)

                        bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
                                
                        confs = det[:, 4].cpu()
                        clas = det[:, 5].cpu()

                        detected_name = names[int(cls)]
                        #print(confs)   

                        #print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1}    y1:{y1}    x2:{x2}    y2:{y2}')
                        label = f'{detected_name} {conf:.2f}'

                        detection_result.append([int(x[0]), int(x[1]), int(x[2]), int(x[3]), conf, detected_name, label])

                        # ****************************** deepsort ****************************

                        #outputs = self.deepsort.update(bbox_xywh, confs, clas, im0)
                else:
                        #outputs = torch.zeros((0, 5))   
                        confs = 0

                t3 = time.time()
            
                #return outputs, confs, names, t2-t1, t3-t2  
                return (detection_result, confs, names, t2-t1, t3-t2)                              

        def draw_boxes(self, img, results, confs=None, names=None):
                rects, waktu_start_palang_tertutup = [], []
                stat_palang = 'palang pintu terbuka'
                colour_barrier = ()
                current_count = int(0)
                height, width, channels = img.shape
                deteksi_objek2 = 0
                klasifikasi2 = 0

                for result in results:
                        x1,y1,x2,y2,cnf,detected_name,label = result

                        x = int((x1 + x2)/2)
                        y = int((y1 + y2)/2)    
                                
                        center = (x,y)

                        box = (x1, y1 , x2, y2)
                        #rects.append(box)
                        #objects = self.ct.update(rects)

                        #for (objectID, centroid) in objects.items():

                                #text = "ID {}".format(objectID)
                                #cv2.putText(img,text,(centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
                                #cv2.circle(img, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)

                                #self.pts[id].append(center)

                                #for j in range(1, len(self.pts[id])):
                                #if self.pts[id][j-1] is None or self.pts[id][j] is None:
                                #continue
                                #thickness = int(np.sqrt(64/float(j+1))*1.5)
                                #cv2.line(img, (self.pts[id][j-1]), (self.pts[id][j]), color, thickness)


                        #================================= bounding box =========================================
                        (a, b), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  
                        cv2.rectangle(img,(x1, y1),(x2,y2),(0, 0, 255),1)
                
                        #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
                        cv2.rectangle(img, (x1, y1 - 13), (x1 + a, y1), (0, 0, 255), -1)
                        cv2.putText(img,label,(x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
                        
                        cv2.circle(img,center, 2, (0, 0, 255), -1)
                        cv2.polylines(img, [self.quad_as_contour], True, (0,0,255), thickness=1)


                        #================================= cek barrier =========================================
                        cv2.line(img, (0, int(height / 2.3)), (int(width), int(height / 2.3)), (0, 0, 255), thickness=1)

                        nameclass = detected_name
                        #print(nameclass)
                    
                        if(nameclass == 'palang' and y > height / 2.3 ):
                                stat_palang = 'palang pintu tertutup'
                                start_palang_tertutup = time.strftime("%I:%M:%S")
         
                                img, results2, deteksi_objek2, klasifikasi2 = self.load_model(img, self.detector_object) 
                                #print('Deteksi objek model 2 - time:(%.3fs) Klasifikasi model 2 - time:(%.3fs)' % (deteksi_objek2, klasifikasi2))
                                if len(results2) > 0:
                                        objects = []
                                        items = []
                                        for result2 in results2:
                                                x1,y1,x2,y2,cnf,idclass,detected_name,label = result2
                                                x = int((x1 + x2)/2)
                                                y = int((y1 + y2)/2)    
                                        
                                                center = (x,y)

                                                box = (x1, y1 , x2, y2)
                                                rects.append(box)     
                                                
                                                color = self.compute_color_for_labels(idclass)
                                                (a, b), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                                                frame = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1) # box
                                                frame = cv2.rectangle(img, (x1, y1 - 13), (x1 + a, y1), (255,0,00), -1)
                                                frame = cv2.putText(img,label,(x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1) 
                                                frame = cv2.circle(img,center, 2, (0, 0, 255), -1)

                                        objects = self.ct.update(rects)
                                        for (objectID, centroid) in objects.items():


                                                center_ = (int(centroid[0]),int(centroid[1]))

                                                if cv2.pointPolygonTest(self.selection_dict['points selected'], center_, measureDist=False) >=0 :
                                                        text = "ID {}".format(objectID)
                                                        cv2.putText(img,text,(centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
                                                        cv2.circle(img, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)

                                                        self.counting_pelanggaran(img, objects, x1, x2, y1, y2, detected_name,cnf)
                                                        current_count += 1

                                        colour_barrier = (0,0,255)
                        
                        elif(nameclass == 'palang' and y < height / 2.3 ):
                                stat_palang = 'palang pintu terbuka'
                                colour_barrier = (36,255,12)


                        overlay = img.copy()
                        cv2.fillPoly(img, [self.quad_as_contour], color = colour_barrier)
                        alpha = 0.80  # Transparency factor.
                        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                        if (stat_palang == 'palang pintu terbuka' and self.totalpelanggaran > 0):
                                #print('total pelanggaran %d' % (self.totalpelanggaran))
                                print(self.start_palang_tertutup)

                                print(stat_palang)
                                print(time.strftime("%I:%M:%S"))

                                #print('durasi waktu palang ditutup (%.3fs)' % (time.time() - self.start_palang_tertutup))
                                                
                                totalpelanggaran = {
                                        "lokasi":'JPL 156 Km 152+375 Stasiun Andir',
                                        "jenis_pelanggaran":'menerobos palang perlintasan',
                                        "tanggal":time.strftime("%Y-%m-%d"),
                                        "palang tertutup":'-',
                                        "palang terbuka":time.strftime("%I:%M:%S"),
                                        #"durasi waktu":(time.time() - self.start_palang_tertutup),
                                        "pelanggaran":self.totalpelanggaran
                                }
                                #print(totalpelanggaran)
                                #self.totalpelanggaran = 0

                
                text_scale = max(1, img.shape[1] // 1600) 
                #total_count = len(set(self.counter))    
                cv2.putText(img, "Jumlah Hambatan: " + str(current_count), (20, 40 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                #cv2.putText(img, "Total Vehicle Count: " + str(total_count), (20, 60 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                cv2.putText(img, "Status Palang: " + stat_palang, (20, 60 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                cv2.putText(img, "Jumlah Pelanggaran: " + str(self.totalpelanggaran), (20, 80 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                #cv2.putText(img, "Waktu Palang Pintu Tertutup: " + str(self.start_palang_tertutup), (20, 120 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)

                return (img, deteksi_objek2, klasifikasi2)


        def load_model(self, img, model):

                #print('cek model 2 !')
                print(f'load model : {self.model_object}')
   
                t1 = time_sync()

                #outputs2, confs, names, yt, st = self.image_track(image2, self.detector_object) 
                #print(outputs2)


                # Padded resize
                img2 = letterbox(img, new_shape=self.img_size)[0]

                # Convert
                img2 = img2[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img2 = np.ascontiguousarray(img2)

                # numpy to tensor
                img2 = torch.from_numpy(img2).to(self.device)
                img2 = img2.half() if self.half else img2.float()  # uint8 to fp16/32
                img2 /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img2.ndimension() == 3:
                        img2 = img2.unsqueeze(0)

                s = '%gx%g ' % img2.shape[2:]    # print string
                #pred = self.models(img, augment=self.args.augment)[0]
                pred2 = model(img2, augment=self.augment)[0]  # list: bz * [ (#obj, 6)]
                names = model.module.names if hasattr(model, 'module') else model.names

                # Apply NMS and filter object other than person (cls:0)
                pred2 = non_max_suppression(pred2, self.args.conf_thres, self.args.iou_thres,classes=self.args.classes, agnostic=self.args.agnostic_nms) 


                t2 = time_sync()

                detection_result = []
                for i, det in enumerate(pred2):
                        if len(det): 
                            for d in det: # d = (x1, y1, x2, y2, conf, cls)
                                x1 = int(d[0].item())
                                y1 = int(d[1].item())
                                x2 = int(d[2].item())
                                y2 = int(d[3].item())
                                conf = round(d[4].item(), 2)
                                c = int(d[5].item())
                                
                                detected_name = names[c]

                                bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
                                confs = det[:, 4].cpu()
                                clas = det[:, 5].cpu()

                                #print(f'Detected: {detected_name} conf: {conf}  bbox: x1:{x1}    y1:{y1}    x2:{x2}    y2:{y2}')
                                label = f'{detected_name} {conf:.2f}'

                                detection_result.append([x1, y1, x2, y2, conf, c, detected_name, label])
                                #detection_result.append([x1, y1, x2, y2, conf, c])

                                


                                #outputs = self.deepsort.update(bbox_xywh, confs, clas, img)
                        else:
                                #outputs = torch.zeros((0, 5))   
                                confs = 0
                        #outputs2 = self.deepsort.update(bbox_xywh, confs, clas, image2)
                        #print(outputs2)

                #else:
                        #outputs2 = torch.zeros((0, 5))

                        #print(f'Detected: {detected_name} conf: {confs}  bbox: {bbox_xywh}')
                #outputs2, confs, names, yt, st = self.image_track(image2, self.detector_object)
                #print(outputs2)
                #cv2.imshow("frame", image2)
                #
                #cv2.imshow('Deteksi Pelanggaran',frame)
                #cv2.imshow('Frame', frame)
                t3 = time.time()


                return (img, detection_result, t2-t1, t3-t2 )

                #return outputs, confs, names, t2-t1, t3-t2 


        def counting_pelanggaran(self, img, objects, x1, x2, y1, y2, nameclass,conf):
                dir_output ='output'
                arah_objek = '0'
                print(conf)
                w = x2-x1
                h = y2-y1
                
                for (objectID, centroid) in objects.items():
                        
                        center_ = (int(centroid[0]),int(centroid[1]))

                        to = self.trackableObjects.get(objectID, None)
                        #\print(to)
                        if to is None:
                            to = TrackableObject(objectID, centroid)
                        else:           
                            y = [c[1] for c in to.centroids]
                            direction = centroid[1] - np.mean(y)
                            #print(direction)
                            to.centroids.append(centroid)
                            if not to.counted: #arah up
                                #if direction < 0 and centroid[1] < height / 1.8 and centroid[1] > height / 2:
                                if cv2.pointPolygonTest(self.selection_dict['points selected'], center_, measureDist=False) >=0 : 
                                        #nm = str(nameclass)+str(objectID)
                                        #crop_img = img[y1:y1+h+5, x1:x1+w+5]
                                        #p = os.path.sep.join([dir_output, time.strftime("%Y%m%d%I%M%S_")+"{}.png".format(nm.zfill(5))])
                                        #print(p)
                                        #cv2.imwrite(p, crop_img)

                                        self.totalpelanggaran += 1
                                        to.counted = True
                                        
                                        
                                        json_pelanggaran = {
                                        "lokasi":'JPL 156 Km 152+375 Stasiun Andir',
                                        "jenis_pelanggaran":'menerobos palang perlintasan',
                                        "objek_pelanggaran":nameclass,
                                        "level_conf":conf,
                                        "tanggal":time.strftime("%Y-%m-%d"),
                                        "waktu":time.strftime("%I:%M:%S"),
                                        "id_objek":objectID,
                                        "image_objek":'-'
                                        }

                                        #print(json_pelanggaran)
                                        #try:
                                            #headers =  {"Content-Type":"application/json"}
                                            #requests.post(url=self.URL_API_Pelanggaran, json=json_pelanggaran, headers=headers)
                                        #except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
                                            #requests.exceptions.URLRequired) as e:
                                            #print(e)

                        self.trackableObjects[objectID] = to
                                
                        objCount = []        
                        objCount.append(self.totalpelanggaran)
                        #print(objCount)

                text_scale = max(1, img.shape[1] // 1600) 
                cv2.putText(img, "Jumlah Pelanggaran: " + str(objCount), (20, 100 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)


                return self.totalpelanggaran

        def compute_color_for_labels(self, label):
                """
                Simple function that adds fixed color depending on the class
                """
                palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
                color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
                return tuple(color)

                
        def crop_frame(self, img0):
                polygon = [[[22, 185], [175, 146], [618, 226], [492, 357]]]
                 # First find the minX minY maxX and maxY of the polygon
                minX = img0.shape[1]
                maxX = -1
                minY = img0.shape[0]
                maxY = -1
                for point in polygon[0]:

                        x = point[0]
                        y = point[1]

                        if x < minX:
                                minX = x
                        if x > maxX:
                                maxX = x
                        if y < minY:
                                minY = y
                        if y > maxY:
                                maxY = y

                # Go over the points in the image if thay are out side of the emclosing rectangle put zero
                # if not check if thay are inside the polygon or not
                cropedImage = np.zeros_like(img0)
                for y in range(0, img0.shape[0]):
                        for x in range(0, img0.shape[1]):

                                if x < minX or x > maxX or y < minY or y > maxY:
                                        continue

                                if cv2.pointPolygonTest(np.asarray(polygon),(x,y),False) >= 0:
                                        cropedImage[y, x, 0] = img0[y, x, 0]
                                        cropedImage[y, x, 1] = img0[y, x, 1]
                                        cropedImage[y, x, 2] = img0[y, x, 2]

                # Now we can crop again just the envloping rectangle
                new_frame = cropedImage[minY:maxY,minX:maxX]  
                return new_frame

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

        def split_list(self, a_list):
            half = len(a_list)//4
            return a_list[0:4], a_list[4:9], a_list[9:13]

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

            self.selection_dict['points selected'] = np.array(self.selection_dict['points selected'], dtype=np.int32)
            

            return 1


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        # input and output
        parser.add_argument('--input_path', type=str, default='0', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--save_path', type=str, default='output/', help='output folder')  # output folder
        parser.add_argument("--frame_interval", type=int, default=1)
        #parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save_txt', default='output/predict/', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

        # camera only
        parser.add_argument("--display", action="store_true")
        parser.add_argument("--display_width", type=int, default=416)
        parser.add_argument("--display_height", type=int, default=416)
        parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

        # YOLO-V5 parameters
        parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
        parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--classes', nargs='+', type=int, default=[0,2,3,4,5,6], help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')

        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)
        #print(args)

        with VideoTracker(args) as vdo_trk:
                vdo_trk.run()

