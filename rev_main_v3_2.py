import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.general import (
        LOGGER, check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.datasets import letterbox

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

import base64
import paho.mqtt.client as mqtt
import threading
import csv

class VideoTracker(object):
        def __init__(self, args):
                print('Initialize : YOLO-V5')
                # ***************** Initialize ******************************************************
                self.args = args
                self.augment = args.augment = False
                self.img_size = args.img_size                   # image size in detector, default is 640
                self.frame_interval = args.frame_interval       # frequency

                #self.device = select_device(args.device)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
                print(f'Initialize : {self.device}')

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
                        #self.vdo = cv2.VideoCapture(self.args.input_path)
                        #os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp' # Use tcp instead of udp if stream is unstable
                        self.vdo = cv2.VideoCapture(self.args.input_path)
                        #print(self.vdo.isOpened())
                        #if not self.vdo.isOpened():
                            #print('Cannot open RTSP stream')
                            #exit(-1)
                

                #print('Done..')
                if self.device == 'cpu':
                        warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

                self.pts = [deque(maxlen=40) for _ in range(9999)]
                self.counter = []
                self.selection_dict = {'img': None, 'points selected': []}
                self.selection_dict2 = {'img': None, 'points selected': []}
                #self.model_image_size = (self.args.display_width,self.args.display_height)
                self.model_image_size = (416,416)
                self.ct = CentroidTracker()
                self.trackableObjects = {}          
                self.start_palang_tertutup = ''

                self.URL_API_lalulintas = 'http://localhost:4000/lalulintas'
                self.URL_API_Pelanggaran = 'http://localhost:4000/pelanggaran'
                self.frame_rate = 0.0

                #self.points_reshape = np.array([[22, 185], [175, 146], [618, 226], [492, 357]])
                #self.points_reshape = np.array([[33, 194], [162, 147], [602, 217], [531, 351]])
                #self.points_reshape = np.array([[44, 197], [185, 155], [599, 205], [526, 352]])
    
                self.points_reshape = np.array([[42, 135], [142, 111], [372, 144], [321, 228]])
                #self.points_reshape = np.array([[23, 146], [115, 118], [350, 157], [298, 229]])

                self.broker = '127.0.0.1'
                self.port = 1883
                self.MQTT_RECEIVE = "video/pelanggaran"
                self.topic=self.MQTT_RECEIVE
                self.frame = np.zeros((240, 320, 3), np.uint8)
                #self.frame = []
                
        def subscribe(self):
                
                #self.client.loop_forever() # Start networking daemon
                self.client.loop_start()

        def unsubscribe(self):
                
                #self.client.loop_forever() # Start networking daemon
                self.client.loop_stop()
        
        # The callback for when the client receives a CONNACK response from the server.
        def on_connect(self, client, userdata, flags, rc):
                
                if rc == 0:
                        print("Connected to MQTT Broker!")
                        print("Connected with result code "+str(rc))
                        # Subscribing in on_connect() means that if we lose the connection and
                        # reconnect then subscriptions will be renewed.
                        self.client.subscribe(self.MQTT_RECEIVE)
                else:
                        print("Failed to connect, return code %d\n", rc)
                
        

        # The callback for when a PUBLISH message is received from the server.
        def on_message(self, client, userdata, msg):
                date = time.strftime('%Y-%m-%d %H:%M:%S')
                global stat_string
                # Decoding the message
                stat = base64.b64decode(msg.payload)
                self.stat_string = stat.decode("ascii")
                if self.stat_string =='palang pintu tertutup':   
                    ret, self.frame = self.vdo.read()
                    #print(f"Received `{self.stat_string}` from `{msg.topic}` topic")

                elif self.stat_string =='palang pintu terbuka':
                    print('palang pintu terbuka')
    
                # converting into numpy array from buffer
                #npimg = np.frombuffer(img, dtype=np.uint8)
                # Decode to Original Frame
                #self.frame = cv2.imdecode(npimg, 1)
                #subs = self.frame
                #self.status_gate = "palang pintu Tertutup"

                #self.stat = msg.payload

                #print("Message received-> " + msg.topic)  # Print a received msg
                print(f"Received `{self.stat_string}` from `{msg.topic}` topic")

                    


        def __enter__(self):
                # ************************* Load video from camera *************************
                #if self.args.cam != -1:
                        #print('Camera ...')
                        #print("Using webcam " + str(self.args.cam))
                        #self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
                        #self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # ************************* Load video from file *************************
                #else:
                #assert os.path.isfile('video/videotes2.mp4'), "Path error"
                #self.vdo.open('video/videotes2.mp4')
                #self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
                #self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #assert self.vdo.isOpened()
                #print('Done. Load video file ', 'video/videotes2.mp4')

                # =============================== mqtt =====================================
                
                self.client = mqtt.Client()  # Create instance of client 
                self.client.on_connect = self.on_connect  # Define callback function for successful connection
                self.client.message_callback_add(self.topic,self.on_message)
                self.client.connect(self.broker,self.port)  # connecting to the broking server
                t=threading.Thread(target=self.subscribe) # make a thread to loop for subscribing
                t.start() # run this thread
                
                return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
                # Stop the Thread
                #self.client.loop_stop()
                #self.client.disconnect()
                if exc_type:
                        print(exc_type, exc_value, exc_traceback)        


        def run(self):             
                
                deteksi_objek_time, fps, avg_perframe, avg_perframe2, fps_arr = [], [], [], [], []
                idx_frame = 0
                self.stat_string =''
                t_start = time.time()
                self.totalpelanggaran = 0
                
                
                while True:

                    if self.stat_string == 'palang pintu tertutup' or self.stat_string !='palang pintu terbuka':
                            # ***************************** Inference *********************************************************************
        
                            t0 = time.time() 
                            img0 = self.letterbox_image(self.frame, tuple(reversed(self.model_image_size)))
                            
                            results, deteksi_objek = self.load_model(img0, self.detector_object) 
                            if len(results) > 0:
                                    deteksi_objek_time.append(deteksi_objek)                
                                    img0 = self.draw_boxes(img0, results)  # BGR
                                    idx_frame += 1


                                    #add FPS information on output video
                                    t1 = time.time() 
                                    frame_rate = ( self.frame_rate + (1./(time.time()-t0)) ) 
                                    #fps_arr.append((time.time(), frame_rate))
                                    print("FPS 1: %.2f " % frame_rate)
                                    avg_perframe.append(frame_rate)

                                    #fpsm = 1 / (time.time() - t0)
                                    #LOGGER.info(f'FPS 2: {fpsm:.1f}')
                                    #avg_perframe2.append(fpsm)    


                                    print('model detection time 2 (%.3fs)' % (sum(deteksi_objek_time) / len(deteksi_objek_time)))

                                    #avg_fps1 = (sum(avg_perframe) / len(avg_perframe))
                                    print('Avg FPS : %.2f ' % (sum(avg_perframe) / len(avg_perframe)))

                                    #avg_fps2 = (sum(avg_perframe2) / len(avg_perframe2))
                                    #print('Avg FPS 2 : %.2f ' % (avg_fps2))

                                    text_scale = max(1, img0.shape[1] // 1600)
                                    cv2.putText(img0, 'fps: %.2f ' % (frame_rate), (20, 20 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), thickness=1)
                                    #cv2.putText(img0, 'FPS: %.2f ' % (int(1 / (t1 - t0))), (20, 80 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), thickness=1)
                                    
                           
                                    # ***************************** display on window ******************************
                                    cv2.imshow("Edge Server", img0)
                                    if cv2.waitKey(1) == ord('q'):  # q to quit
                                            cv2.destroyAllWindows()
                                            break     

                    else:
                            #print('palang pintu terbuka')
                            cv2.destroyAllWindows()
                                
                t_end = time.time()
                #avg_fps = (sum(fps_perframe) / len(fps_perframe))
                #print('Sum model detection time 2 (%.3fs)' % (sum(deteksi_objek_time)))
                #print('Avg model detection time 2 (%.3fs)' % (sum(deteksi_objek_time) / len(deteksi_objek_time)))
                #print(f'Total pelanggaran : {self.totalpelanggaran}')
                #print('Avg FPS 1 : %.2f ' % (avg_fps1))
                #print('Avg FPS 2 : %.2f ' % (avg_fps2))
                #print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))

        
        def load_model(self, im0, model):
                t1 = time_sync()

                #outputs2, confs, names, yt, st = self.image_track(image2, self.detector_object) 
                #print(outputs2)

                # Padded resize
                #img = letterbox(im0, new_shape=self.args.display_width)[0]
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
                #pred = self.models(img, augment=self.args.augment)[0]
                pred = model(img, augment=self.augment)[0]  # list: bz * [ (#obj, 6)]
                names = model.module.names if hasattr(model, 'module') else model.names

                # Apply NMS and filter object other than person (cls:0)
                pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,classes=self.args.classes, agnostic=self.args.agnostic_nms) 

                t2 = time_sync()
                detection_result = []
                for i, det in enumerate(pred):
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
                        else: 
                                confs = 0


                t3 = time.time()
                return (detection_result, t3-t1)

        def draw_boxes(self, img, results):
                rects = []
                stat_palang = 'palang pintu tertutup'
                colour_barrier = ()
                current_count = int(0)
                height, width, channels = img.shape
                #points = np.array([[17, 124], [128, 96], [396, 139], [349, 232]])
                #points_reshape = points.reshape((-1,1,2))
                
        
                for result in results:
                        x1,y1,x2,y2,cnf,idclass,detected_name,label = result
                        x = int((x1 + x2)/2)
                        y = int((y1 + y2)/2)    
                        center = (x,y)
                        
                        box = (x1, y1 , x2, y2)
                        rects.append(box)
                        #print(label)

                        #================================= bounding box =========================================
                        #color = (self.compute_color_for_labels(idclass))
                        (a, b), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  
                        cv2.rectangle(img,(x1, y1),(x2,y2),(255,0,00),1)
                        #cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
                        cv2.rectangle(img, (x1, y1 - 13), (x1 + a, y1), (255,0,00), -1)
                        cv2.putText(img,label,(x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1) 
                        cv2.circle(img,center, 2, (0, 0, 255), -1)
                        cv2.polylines(img, [self.points_reshape], True, (0,0,255), thickness=1)


                        #================================= ROI =========================================
                        colour_barrier = (0,0,255)
                        overlay = img.copy() 
                        cv2.fillPoly(img, pts=[self.points_reshape], color=colour_barrier)
                        alpha = 0.95  # Transparency factor.
                        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
                       
                
                objects = self.ct.update(rects)
                for (objectID, centroid) in objects.items():
                        
                        center_ = (int(centroid[0]),int(centroid[1]))

                        #================================= counting pelanggaran ==================================
                        if cv2.pointPolygonTest(self.points_reshape, center_, measureDist=False) >=0 :
                                text = "ID {}".format(objectID)
                                cv2.putText(img,text,(centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)
                                cv2.circle(img, (centroid[0], centroid[1]), 2, (0, 0, 255), -1)
                                self.counting_pelanggaran(img, objects, detected_name,cnf)
                                #self.counter.append(int(objectID))
                                current_count += 1
       
                stat_palang = 'Palang Pintu Tertutup'
                text_scale = max(1, img.shape[1] // 1600) 

                #total_count = len(set(self.counter))    
                cv2.putText(img, "Jumlah Hambatan: " + str(current_count), (20, 40 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                #cv2.putText(img, "Total Pelanggaran: " + str(total_count), (20, 60 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                cv2.putText(img, "Status : " + stat_palang, (20, 80 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                cv2.putText(img, "Jumlah Pelanggaran: " + str(self.totalpelanggaran), (20, 60 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                #cv2.putText(img, "Waktu Palang Pintu Tertutup: " + str(self.start_palang_tertutup), (20, 120 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)

                return (img)


        def counting_pelanggaran(self, img, objects, nameclass,conf):
                #print('counting_pelanggaran')
                dir_output ='output'
                arah_objek = '0'
                jsondata = []
                #w = x2-x1
                #h = y2-y1
                
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
                                if cv2.pointPolygonTest(self.points_reshape, center_, measureDist=False) >=0 : 

                                        #nm = str(nameclass)+str(objectID)
                                        #crop_img = img[y1:y1+h+5, x1:x1+w+5]
                                        #p = os.path.sep.join([dir_output, time.strftime("%Y%m%d%I%M%S_")+"{}.png".format(nm.zfill(5))])
                                        #print(p)
                                        #cv2.imwrite(p, crop_img)

                                        self.totalpelanggaran += 1
                                        to.counted = True
                                        
                                        json_pelanggaran = {
                                                "tanggal":time.strftime("%Y-%m-%d"),
                                                "waktu":time.strftime("%I:%M:%S"),
                                                "lokasi":'JPL 156 Km 152+375 Stasiun Andir',
                                                "jenis_pelanggaran":'menerobos palang perlintasan',
                                                "objek_pelanggaran":nameclass,
                                                "level_conf":conf,
                                                "id_objek":objectID
                                                #"image_objek":p
                                        }

                                        print(json_pelanggaran)
                                        #try:
                                            #headers =  {"Content-Type":"application/json"}
                                            #requests.post(url=self.URL_API_Pelanggaran, json=json_pelanggaran, headers=headers)
                                        #except (requests.exceptions.RequestException, requests.exceptions.ConnectionError,
                                            #requests.exceptions.URLRequired) as e:
                                            #print(e)
                                        json_string = json.dumps(json_pelanggaran, indent = 4) 
                                        jsondata = json.loads(json_string)
                                        
                                        headers = ['tanggal', 'waktu', 'lokasi','jenis_pelanggaran', 'objek_pelanggaran', 'level_conf', 'id_objek']
                                        with open('data_pelanggaran.csv', 'a', encoding='UTF8', newline='') as csvfile:
                                            dictwriter_object = csv.DictWriter(csvfile, fieldnames=headers)
                                            #writer.writeheader()
                                            dictwriter_object.writerow(jsondata)
                                            csvfile.close()


                        self.trackableObjects[objectID] = to
                                
                        objCount = []        
                        objCount.append(self.totalpelanggaran)
                        #print(objCount)

                text_scale = max(1, img.shape[1] // 1600) 
                #cv2.putText(img, "Jumlah Pelanggaran: " + str(objCount), (20, 100 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)


                return self.totalpelanggaran

        

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

                return new_image

        def select_point(self, event, x, y, flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(self.selection_dict['img'],(x,y), 5, (0, 255, 0), -1)
                self.selection_dict['points selected'].append([x, y])       


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
        parser.add_argument('--classes', nargs='+', type=int, default=[0,2,3], help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')



        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)
    
        with VideoTracker(args) as vdo_trk:
                vdo_trk.run()