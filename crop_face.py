
import os
import cv2
import utils
import numpy as np

from mediaio.video_io import VideoFileWriter, VideoFileReader
from scipy.misc import imresize, imsave

FRAME_SIZE = 256

def detect_face(frame, classifier_type='lbp'):

	if classifier_type == 'lbp':
		classifier_path = '/cs/labs/peleg/asaph/face_classifiers/lbp_classifier.xml'
	elif classifier_type == 'haar':
		classifier_path = '/cs/labs/peleg/asaph/face_classifiers/haar_classifier.xml'
	else:
		print('unknown classifier: exiting')
		return

	detector = cv2.CascadeClassifier(classifier_path)

	faces = detector.detectMultiScale(frame, scaleFactor=1.2)

	if len(faces) == 0:
		print('no face detected')
		return

	if len(faces) > 1:
		print len(faces), 'faces detected'
		return faces[1]

	return faces[0]

def crop_face(frame):
	face = detect_face(frame, 'lbp')

	if face is None:
		pass

	x, y, w, h = face
	if w < h:
		w = h
	else:
		h = w

	return imresize(frame[y:y+h, x:x+w], (FRAME_SIZE, FRAME_SIZE))

def crop_video(video_path):

	detector = utils.FaceDetector()
	frames = utils.get_frames(video_path)
	cropped = np.zeros([frames.shape[0], FRAME_SIZE, FRAME_SIZE], dtype=np.uint8)

	for i in range(frames.shape[0]):
		cropped_frame = detector.crop_face(frames[i], bounding_box_shape=(FRAME_SIZE, FRAME_SIZE))
		cropped[i, :, :] = cropped_frame
		# cropped[i, :, :] = crop_face(frames[i])

	return cropped

def save_cropped(video_path, cropped_frames):

	with VideoFileWriter(video_path, 25) as writer:
		for i in range(cropped_frames.shape[0]):
			writer.write_frame(cropped_frames[i])


if __name__ == '__main__':
	cropped_f = crop_video('/cs/grad/asaph/testing/obama2.avi')
	save_cropped('/cs/grad/asaph/testing/obama_crop2.avi', cropped_f)


