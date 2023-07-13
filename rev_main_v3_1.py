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
import paho.mqtt.client as mqtt
import base64

cudnn.benchmark = True

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

                #self.model_palang = 'model/yolov5s/prune/416/model_palang_416_32_100_0.00005_0.45_ft_best.pt'
                self.model_palang = 'model/model_palang_416_16_100.pt'
                #self.model_palang = 'model/yolov5s/prune/yolov5s_model_palang_416_32_100_0.00005_0.4_best.pt'
                self.detector_palang = torch.load(self.model_palang, map_location=self.device)['model'].float()  # load to FP32
                self.detector_palang.to(self.device).eval()
                if self.half:
                        self.detector_palang.half()  # to FP16


                # ***************************** create video capture ****************
                if self.args.cam != -1:
                        print("Using webcam " + str(self.args.cam))
                        self.vdo = cv2.VideoCapture(self.args.cam)
                else:
                        #self.vdo = cv2.VideoCapture(self.args.input_path)
                        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp' # Use tcp instead of udp if stream is unstable
                        self.vdo = cv2.VideoCapture(self.args.input_path, cv2.CAP_FFMPEG)
                        #print(self.vdo.isOpened())
                        if not self.vdo.isOpened():
                            print('Cannot open RTSP stream')
                            exit(-1)

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
                
                self.URL_API_lalulintas = 'http://localhost:4000/lalulintas'
                self.URL_API_Pelanggaran = 'http://localhost:4000/pelanggaran'

                self.frame_rate = 0.0
                self.posisi_palang = 2.3

                # =============================== MQTT =====================================
                #mqttBroker ="mqtt.eclipseprojects.io"
                self.broker = '127.0.0.1'
                self.port = 1883
                # Topic on which frame will be published
                self.MQTT_SEND = "video/pelanggaran"
                # Object to capture the frames
                #cap = cv.VideoCapture(video_source)
                # Phao-MQTT Clinet
                self.client = mqtt.Client()
                # Establishing Connection with the Broker
                self.client.connect(self.broker, self.port)

        def __enter__(self):
                # ************************* Load video from camera *************************
                if self.args.cam != -1:
                        print('Camera ...')
                        print("Using webcam " + str(self.args.cam))
                        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
                        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # ************************* Load video from file *************************
                #else:
                        #assert os.path.isfile(self.args.input_path), "Path error"
                        #self.vdo.open(self.args.input_path)
                        #self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
                        #self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        #assert self.vdo.isOpened()
                        #print('Done. Load video file ', self.args.input_path)

                return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
                self.vdo.release()
                if exc_type:
                        print(exc_type, exc_value, exc_traceback)        


        def run(self):             
                deteksi_objek1_time, klasifikasi1_time, deteksi_objek2_time, klasifikasi2_time, fps, avg_perframe, avg_perframe0, avg_perframe2, avg_perframe3, avg_perframe4, fps_arr = [], [], [], [], [], [], [], [], [], [], []
                idx_frame = 0
                t_start = time.time()
                last_out = None
                ret, image = self.vdo.read()
                img0 = self.letterbox_image(image, tuple(reversed(self.model_image_size)))

                # ***************************** Inference *********************************************************************
                while self.vdo.grab():
                    t0 = time.time() 
                    _, image = self.vdo.retrieve()
                    img0 = self.letterbox_image(image, tuple(reversed(self.model_image_size)))

                    if idx_frame % self.args.frame_interval == 0:
                            results, confs, names, deteksi_objek1, klasifikasi1 = self.load_model(img0, self.detector_palang) 
                            last_out = results
                            frame_count = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
                            deteksi_objek1_time.append(deteksi_objek1)
                            klasifikasi1_time.append(klasifikasi1)

                            #print('Frame %d / %f' % (idx_frame, frame_count))
                            #print('Deteksi objek model - time:(%.3fs)' % (deteksi_objek1))
                            
                    else:
                            outputs = last_out 
                   
                    # ***************************** post-processing **********************************************************
                    
                    #if len(results) > 0:

                    

                    #add FPS information on output video
                    t1 = time.time() 
                    fps.append(t1 - t0)
                            
                    #frame_rate = (self.frame_rate + (1./(time.time()-t0)) ) 
                    #fps_arr.append((time.time(), frame_rate))
                    #print("FPS : %.2f " % frame_rate)
                    #avg_perframe.append(frame_rate)

                    #fpsm = 1 / (time.time() - t0)
                    #LOGGER.info(f'FPS : {fpsm:.1f}')
                    #avg_perframe2.append(fpsm)
                                                
                    #avg_fps1 = ()
                    #print('Avg FPS : %.2f ' % (sum(avg_perframe) / len(avg_perframe)))
                    print('FPS : %.2f ' % (len(fps) / sum(fps)))
                    #avg_fps2 = (sum(avg_perframe2) / len(avg_perframe2))
                    #print('Avg FPS : %.2f ' % (avg_fps2))
     
                    text_scale = max(1, img0.shape[1] // 1600)
                    cv2.putText(img0, 'FPS: %.2f ' % (len(fps) / sum(fps)), (20, 20 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0,0,255), thickness=1)


                    # *****************************  visualize bbox  ***************************************************
                    if len(results) > 0:
                        img0, deteksi_objek2, klasifikasi2 = self.draw_boxes(img0, results, confs, names)  # BGR

                    #deteksi_objek2_time.append(deteksi_objek2)
                    #klasifikasi2_time.append(klasifikasi2)
                    #print('model detection time 1 (%.3fs)' % (sum(deteksi_objek1_time) / len(deteksi_objek1_time)))
                            
                    # ***************************** display on window ******************************
                    if self.args.display:
                            #cv2.namedWindow("Video_Analytics", cv2.WINDOW_NORMAL)
                            #cv2.resizeWindow("Video_Analytics", self.args.display_width, self.args.display_height)
                            cv2.imshow("Edge Device", img0)
                            if cv2.waitKey(1) == ord('q'): 
                                    cv2.destroyAllWindows()
                                    break

                    idx_frame += 1
                
                t_end = time.time()
                #print('Sum model 1 detection time (%.3fs))' % (sum(deteksi_objek1_time)))
                #print('Avg model 1 detection time (%.3fs) per frame' % (sum(deteksi_objek1_time) / len(deteksi_objek1_time)))
                #print('Avg FPS 1 : %.2f ' % (avg_fps1))
                #print('Avg FPS 2 : %.2f ' % (avg_fps2))
                print('Total time (%.3fs), Total Frame: %d' % (t_end - t_start, idx_frame))
        
        def load_model(self, im0, model):
                # ***************************** preprocess ************************************************************
                print('----------------------------------------------------')
                print(f'load model : {self.model_palang}')
                names = model.module.names if hasattr(model, 'module') else model.names

                # Padded resize
                img = letterbox(im0, new_shape=self.model_image_size)[0]

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
                t1 = time_sync()
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False
                pred = model(img, augment=self.augment, visualize=visualize)[0]  # list: bz * [ (#obj, 6)]
                # Apply NMS and filter object other than person (cls:0)
                pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres,classes=self.args.classes, agnostic=self.args.agnostic_nms) 
                
                t2 = time_sync()



                # ***************************** get all obj ************************************************************
                det = pred[0]  # for video, bz is 1

                detection_result = []
                
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
                        label = f'{detected_name} {conf:.2f}'
                        detection_result.append([int(x[0]), int(x[1]), int(x[2]), int(x[3]), conf, detected_name, label])
                else: 
                        confs = 0

                t3 = time.time()
                LOGGER.info(f'{s}Done. ({t2 - t1:.3f}s)')

                # convert inference time in milliseconds to frames per second as well?
                #fpsm = 1 / (t2 - t1)
                #LOGGER.info(f'FPS: {fpsm:.1f}')
                #LOGGER.info(f'{s}Done. ({t2 - t1:.3f}s)')

                return (detection_result, confs, names, t2-t1, t3-t2)                              

        def draw_boxes(self, img, results, confs=None, names=None):
                stat_palang = 'palang pintu terbuka'
                string_bytes = stat_palang.encode("ascii")
                base64_bytes = base64.b64encode(string_bytes)
                base64_string = base64_bytes.decode("ascii")

                height, width, channels = img.shape
                deteksi_objek2 = 0
                klasifikasi2 = 0

                for result in results:
                        x1,y1,x2,y2,cnf,detected_name,label = result

                        x = int((x1 + x2)/2)
                        y = int((y1 + y2)/2)    
                        center = (x,y)

                        #================================= cek barrier =========================================
                        

                        nameclass = detected_name
                        if(nameclass == 'palang' and y > height / self.posisi_palang ):
                                # Encoding the Frame
                                #_, buffer = cv2.imencode('.jpg', img)
                                # Converting into encoded bytes
                                stat_palang = 'palang pintu tertutup'
                                string_bytes = stat_palang.encode("ascii")
                                base64_bytes = base64.b64encode(string_bytes)
                                base64_string = base64_bytes.decode("ascii")
                            
                                #encode = base64.b64encode(stat_palang)
                                self.start_palang_tertutup = time.strftime("%I:%M:%S")
                                colour_barrier = (0,0,255)
                                # Publishig the Frame on the Topic home/server
              
                        
                        elif(nameclass == 'palang' and y < height / self.posisi_palang ):
                                stat_palang = 'palang pintu terbuka'
                                string_bytes = stat_palang.encode("ascii")
                                base64_bytes = base64.b64encode(string_bytes)
                                base64_string = base64_bytes.decode("ascii")
                                colour_barrier = (36,255,12)


                        result_mqtt = self.client.publish(self.MQTT_SEND, base64_string)
                        status = result_mqtt[0]
                        if status == 0:
                                print(f"Send '{stat_palang}' frame to topic '{self.MQTT_SEND}'")
                        else:
                                print(f"Failed to send message to topic {self.MQTT_SEND}")

                        #================================= garis palang pintu =========================================
                        cv2.line(img, (0, int(height / self.posisi_palang)), (int(width), int(height / self.posisi_palang)), (0, 0, 255), thickness=1)

                        #================================= bounding box =========================================
                        (a, b), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)  
                        cv2.rectangle(img,(x1, y1),(x2,y2),(0, 0, 255),1)        
                        cv2.rectangle(img, (x1, y1 - 13), (x1 + a, y1), (0, 0, 255), -1)
                        cv2.putText(img,label,(x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], 1)  
                        cv2.circle(img,center, 2, (0, 0, 255), -1)

                text_scale = max(1, img.shape[1] // 1600) 
                #total_count = len(set(self.counter))    
                #cv2.putText(img, "Jumlah Hambatan: " + str(current_count), (20, 40 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                #cv2.putText(img, "Total Vehicle Count: " + str(total_count), (20, 60 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (36,255,12), 1)
                cv2.putText(img, "Status : " + stat_palang, (20, 40 + text_scale), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0,0,255), 1)
            
                return (img, deteksi_objek2, klasifikasi2)


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


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        # input and output
        parser.add_argument('--input_path', type=str, default='0', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--save_path', type=str, default='output/', help='output folder')  # output folder
        parser.add_argument("--frame_interval", type=int, default=1)
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--save_txt', default='output/predict/', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

        # camera only
        parser.add_argument("--display", action="store_true")
        parser.add_argument("--display_width", type=int, default=416)
        parser.add_argument("--display_height", type=int, default=416)
        parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")

        # YOLO-V5 parameters
        parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')

        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)

        with VideoTracker(args) as vdo_trk:
                vdo_trk.run()

