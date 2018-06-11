
from facedetection.face_detection import FaceDetector
from mediaio.video_io import VideoFileReader, VideoFileWriter
import argparse
import os
import numpy as np


def crop_lips(dataset_folder, crop_shape):
    f = FaceDetector()
    for speaker in os.listdir(dataset_folder):
        vids_dir = os.path.join(dataset_folder, speaker, 'video')
        lips_dir = os.path.join(dataset_folder, speaker, 'lips')

        if not os.path.exists(lips_dir):
            os.mkdir(lips_dir)

        for vid in os.listdir(vids_dir):
            vid_path = os.path.join(vids_dir, vid)
            print('processing: %s' % vid_path)
            try:
                with VideoFileReader(vid_path) as in_vid:
                    fps = in_vid.get_frame_rate()
                    frames = in_vid.read_all_frames(convert_to_gray_scale=True)
                    cropped_lips = []
                    for i in range(frames.shape[0]):
                        cropped_lips.append(f.crop_mouth(frames[i], bounding_box_shape=crop_shape))

                lips_path = os.path.join(lips_dir, vid)
                with VideoFileWriter(lips_path, fps) as out_lip:
                    for frame in cropped_lips:
                        out_lip.write_frame(frame)
            except Exception as e:
                print('failed to process:\n%s' % str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, required=True)
    parser.add_argument('-s', '--crop_shape', type=int, nargs='+', default=[128, 128])

    args = parser.parse_args()

    crop_lips(args.dataset_dir, args.crop_shape)




