import numpy as np
import os
from PIL import Image
import h5py
import cv2

img_path = '/home/imaxct/code/imgs'

h5_file = './imgs/dataset.h5'


def get_all_data():
        x_data = np.zeros((0, 20 * 60 * 1), dtype=np.uint8)
        y_data = np.zeros((0, 4 * 9), dtype=np.uint8)
        filenames = os.listdir(img_path)
        i = 0
        read = 0
        for num, f in enumerate(filenames):
                i += 1
                if i % 500 == 0:
                    print(i)
                    print(f)
                    print(list(f[:4]))
                full_path = os.path.join(img_path, f)
                img = Image.open(full_path)
                img = np.array(img, dtype=np.uint8)
                if img is not None:
                        read += 1
                        img = img.reshape((-1,))
                        x_data = np.vstack( (x_data, img) )
                        l = list(f[:4])
                        y_tmp = np.zeros( (4 * 9) )
                        y_tmp[ 0 * 9 + int(l[0]) - 1] = 1.0
                        y_tmp[ 1 * 9 + int(l[1]) - 1] = 1.0
                        y_tmp[ 2 * 9 + int(l[2]) - 1] = 1.0
                        y_tmp[ 3 * 9 + int(l[3]) - 1] = 1.0
                        y_data = np.row_stack( (y_data, y_tmp) )
                if i % 3000 == 0:
                        if os.path.exists(h5_file):
                                with h5py.File(h5_file, 'a') as hf:
                                        x_data = x_data.reshape( (-1, 20, 60, 1) )
                                        y_data = y_data
                                        hf['x'].resize( (hf['x'].shape[0] + x_data.shape[0]), axis=0 )
                                        hf['x'][-x_data.shape[0]:] = x_data

                                        hf['y'].resize( (hf['y'].shape[0] + y_data.shape[0]), axis=0 )
                                        hf['y'][-y_data.shape[0]:] = y_data
                        else:
                                with h5py.File(h5_file, 'w') as hf:
                                        x_data = x_data.reshape( (-1, 20, 60, 1) )
                                        y_data = y_data
                                        hf.create_dataset('x', data=x_data, maxshape=(None, 20, 60, 1))
                                        hf.create_dataset('y', data=y_data, maxshape=(None, 4 * 9))
                        x_data = np.zeros((0, 20 * 60 * 1), dtype=np.uint8)
                        y_data = np.zeros((0, 4 * 9), dtype=np.uint8)

        # print(read)
        x_data = x_data.reshape( (-1, 20, 60, 1) )
        y_data = y_data
        f = h5py.File(h5_file, 'a')
        f['x'].resize( (f['x'].shape[0] + x_data.shape[0]), axis=0 )
        f['x'][-x_data.shape[0]:] = x_data

        f['y'].resize( (f['y'].shape[0] + y_data.shape[0]), axis=0 )
        f['y'][-y_data.shape[0]:] = y_data
        f.close()

if __name__ == '__main__':
        # get_all_data()
    f = h5py.File(h5_file, 'r')
    print(f['x'].shape)
    print(f['y'].shape)
    f.close()
