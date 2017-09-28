"""Performs face alignment and calculates L2 distance between the embeddings of images."""

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

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import detect_face
import cv2

def main(args):

    images = load_and_align_data(args.image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(args.model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)

            path_exp = os.path.expanduser(args.image_files)
            image_paths = facenet.get_image_paths(path_exp)

            nrof_images = len(image_paths)

            print('Images:')
            for i in range(nrof_images):
                print('%1d: %s' % (i, image_paths[i]))
            print('')
            
            # Print distance matrix
            print('Distance matrix')
            print('    ', end='')
            for i in range(nrof_images):
                print('    %1d     ' % i, end='')
            print('')
            for i in range(nrof_images):
                print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    print('  %1.4f  ' % dist, end='')
                print('')
            
      
# TODO: face detection code (4week-1day ex)      
def load_and_align_data(image_files, image_size, margin, gpu_memory_fraction):

    # TODO: set the parameters (minsize, threshold, scale factor)

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')

    # TODO: create MT-CNN (P-net, R-net, O-net)
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    path_exp = os.path.expanduser(image_files)
    image_paths = facenet.get_image_paths(path_exp)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):

        # TODO: face detection and alignment
        img = cv2.imread(image_paths[i])
        bounding_boxes, landmarks = detect_face.detect_face(
                        img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            landmark = landmarks
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                bounding_box_size = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0]+det[:, 2])/2-img_center[1],
                                     (det[:, 1]+det[:, 3])/2-img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0)  # some extra weight on the centering
                det = det[index, :]
                landmark = landmark[:, index]
            det = np.squeeze(det)
            landmark = np.squeeze(landmark)

            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = face_alignment(cropped, image_size, landmark)

            prewhitened = facenet.prewhiten(aligned)
            img_list[i] = prewhitened
    images = np.stack(img_list)

    return images


def face_alignment(img, face_size, f_point):
    desired_left_eye = (0.35, 0.35)
    desired_right_eye = (0.65, 0.35)
    right_eye_center = (f_point[0], f_point[5])
    left_eye_center = (f_point[1], f_point[6])

    # TODO : Compute eyes center, angle and image scale
    eyesCenter_x, eyesCenter_y = np.add(right_eye_center, left_eye_center)
    eyesCenter = (eyesCenter_x/2, eyesCenter_y/2)
    distance = np.sqrt(np.add(np.square(f_point[0]-f_point[1]),np.square(f_point[5]-f_point[6])))
    desired_distance = np.sqrt(np.add(np.square(0.65-0.35),np.square(0.35-0.35)))
    rad = np.arctan2((f_point[6]-f_point[5]), (f_point[1]-f_point[0]))
    angle = np.degrees(rad)
    scale =  distance / (desired_distance * face_size)
 
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
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, help='Images to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
