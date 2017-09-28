"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import detect_face
import cv2

def main(args):

    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO: create video capture class (cv2 package)
    cap = cv2.VideoCapture(0)

    # TODO: set the paramters about MT-CNN (minsize, threshold, scale factor)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    # TODO: create MT-CNN network
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

            np.random.seed(seed=args.seed)

            # Load the model
            facenet.load_model(args.model)

            # TODO: load classifier model
            with open(args.classifier_filename, 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            # TODO: get input and output tensors about embedding network
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # TODO: run face detection for each frame if camera is opened
            # detection -> alignment -> prewhitening -> embedding -> classify -> showing
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret == True:
                 # <<detection>>
                 # bounding_boxes / landmarks : no face, one face, or many faces ...
                    bounding_boxes, landmarks = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        # if you want to classify about all face which is detected in frame
                        for face_no in range(nrof_faces):
                            # get bounding box and landmark about each face from bounding boxes and landmarks
                            # <<alignment : face_alignment>>
                            aligned = face_alignment(frame, args.image_size, landmarks[:,face_no])

                            # <<prewhiten : facenet.prewhiten>>
                            prewhitened = facenet.prewhiten(aligned)

                            # <<embedding : session run>>
                            feed_dict = {images_placeholder: [prewhitened], phase_train_placeholder: False}
                            emb = sess.run(embeddings, feed_dict=feed_dict)
                            # <<classify : model.predict>>

                            # indexes = np.where(emb == class_names)[0]

                            #print(class_names)

                            prob = model.predict_proba(emb)

                            maxprob = max(max(prob))
                            maxIdx = np.where(prob == maxprob)[1][0]
                            txt = str(np.round(maxprob*1000)/1000) + ' : ' + class_names[int(maxIdx)]
                            cv2.rectangle(frame,(int(bounding_boxes[face_no,0]),int(bounding_boxes[face_no,1])),(int(bounding_boxes[face_no,2]),int(bounding_boxes[face_no,3])),color=(0,255,0))
                            cv2.putText(frame,txt,(int(bounding_boxes[face_no,0]),int(bounding_boxes[face_no,3])),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5 , color=(0,255,0))

                        # <<showing>>
                        cv2.imshow('frame',frame)
                        # cv2.imwrite('TestImage.jpg',frame)
                # break if key interrupted
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break
        cv2.destroyAllWindows()
        cap.release()
        print("detection close")





def face_alignment(img, face_size, f_point):

    # TODO: face alignment
    # TODO: face alignment (4week-1day ex)
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)
    right_eye_center = (f_point[0], f_point[5])
    left_eye_center = (f_point[1], f_point[6])

    # TODO : Compute eyes center, angle and image scale
    eyesCenter = ((right_eye_center[0] + left_eye_center[0]) / 2, (right_eye_center[1] + left_eye_center[1]) / 2)

    eye_diff = (left_eye_center[0] - right_eye_center[0], left_eye_center[1] - right_eye_center[1])
    angle = np.degrees(np.arctan2(eye_diff[1], eye_diff[0]))

    scale = (desired_right_eye[0] - desired_left_eye[0]) * face_size / np.sqrt(
        np.power(eye_diff[0], 2) + np.power(eye_diff[1], 2))
    # print(eye_diff[0])
    # print(eye_diff[1])

    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    tX = face_size * 0.5
    tY = face_size * desired_left_eye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (face_size, face_size)

    output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    return output

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('output_dir', type=str, help='Directory with class probability results.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.7)
    parser.add_argument('--cam_device', type=int,
        help='ID of the opened video capturing device (a camera index).', default=0)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
