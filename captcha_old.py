from PIL import Image
import numpy as np

# 9999_1469952682.gif

if __name__ == '__main__':
        img = Image.open('/home/imaxct/code/bad_img/9999_1469952682.gif')
        arr = np.array(img)
        print(arr.shape)