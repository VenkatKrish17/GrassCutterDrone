# load libraries
import cv2
import sys
import tensorflow as tf
import numpy as np
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import os
from utils import label_map_util
from utils import visualization_utils as vis_util
import pickle as pickle
import struct
import socket

import zstd
#import lz4.frame

# Path to frozen detection graph
MODEL_PATH = sys.argv[1]
PATH_TO_CKPT = MODEL_PATH

# List of the strings that is used to add correct label for each box.
PB_TXT_PATH = sys.argv[2]
PATH_TO_LABELS = os.path.join('D:/scripts/frozen_graph', PB_TXT_PATH)

NUM_CLASSES = 3

# loading image map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# socket connection establishment
clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
clientsocket.connect(('localhost',8089))

# stream using opencv and store video file
cap=cv2.VideoCapture(int(sys.argv[3])) # webcam

# load frozen graph into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),50]

with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
                while (True):
                        ret,image_np_org =cap.read()
                        image_np = cv2.resize(image_np_org, (416,416))
                        # Each box represents a part of the image where a particular object was detected.
                        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        # Each score represent how level of confidence for each of the objects.
                        # Score is shown on the result image, together with the class label.
                        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)

                        # Actual detection.
                        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})

                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8,  min_score_thresh=.7)
                        cv2.imshow('live_detection',image_np)

                        print("before compression..." + str(sys.getsizeof(image_np)))
                        print("compressing..")
                        result, imgencode = cv2.imencode('.jpg', image_np, encode_param)
                        data_img = np.array(imgencode)
                        stringData = data_img.tostring()
                        print("after encoding and converting to numpy array..." + str(sys.getsizeof(stringData)))
                        data_compressed = zstd.compress(stringData,20)
                        print("after compression..." + str(sys.getsizeof(data_compressed)))
                        data = pickle.dumps(data_compressed)
                        print("after pickling..." + str(sys.getsizeof(data)))
                        print("sending..")
                        clientsocket.send((str(len(data)).ljust(16)).encode('utf-8','ignore'))
                        clientsocket.emit("videoStream", data)
                        if cv2.waitKey(25) & 0xFF==ord('q'):
                           break
                cv2.destroyAllWindows()
                cap.release()
                clientsocket.close()
