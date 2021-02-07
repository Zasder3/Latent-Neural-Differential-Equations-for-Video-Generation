import cv2
import h5py
import os

import numpy
import pandas
import imageio


import matplotlib.pyplot as plt

src_dir = 'D:/Video Datasets/UCF101_min'
src_split_dir = 'D:/Video Datasets/ucf101/ucfTrainTestList'
dst_dir = 'D:/Video Datasets/ucf101_64px_tgan'
#img_rows, img_cols = 192, 256
img_rows, img_cols = 64, 85


def process_video(video_path):
    video_reader = imageio.get_reader(video_path)
    
    video = []
    while True:
        try:
            img = video_reader.get_next_data()
        except IndexError:
        # except (imageio.core.CannotReadFrameError, IndexError):
            break
        else:
            dst_img = cv2.resize(
                img, (img_cols, img_rows),
                interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
            video.append(dst_img)
    T = len(video)
    video = numpy.concatenate(video).reshape(T, 3, img_rows, img_cols)
    return video


def make_frame(filepath):
    frame = pandas.read_csv(
        filepath, sep=' ', header=None, names=['filename', 'label'])
    del frame['label']
    frame['filename'] = frame['filename'].apply(lambda x: os.path.basename(x))
    frame['category'] = frame['filename'].apply(lambda x: x.split('_')[1])
    return frame


def main():
    for name in ['train', 'test']:
        print('Processing {} dataset'.format(name))
        path = os.path.join(src_split_dir, '{}list01.txt'.format(name))
        frame = make_frame(path)

        n_frames = 0
        # n_frames = 1786820
        for ind, row in frame.iterrows():
            if ind % 100 == 0:
                print('Processing {} / {}'.format(ind, len(frame)))
            path = os.path.join(src_dir, row['filename'])
            reader = imageio.get_reader(path)
            T = reader.count_frames() + 1
            n_frames += T

        print('# of frames: {}'.format(n_frames))
        h5file = h5py.File(os.path.join(dst_dir, '{}.h5'.format(name)), 'w')
        dset = h5file.create_dataset(
            'image', (n_frames, 3, img_rows, img_cols), dtype=numpy.uint8)
        conf = []
        start = 0
        for ind, row in frame.iterrows():
            print('Processing {} / {}'.format(ind, len(frame)))
            path = os.path.join(src_dir, row['filename'])
            video = process_video(path)
            T = len(video)
            dset[start:(start + T)] = video
            conf.append({
                'start': start, 'end': (start + T),
                'category': row['category']})
            start += T
        conf = pandas.DataFrame(conf)
        conf.to_json(os.path.join(dst_dir, '{}.json'.format(name)), orient='records')


if __name__ == '__main__':
    main()