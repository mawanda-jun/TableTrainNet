# import sys
# from common_func import *
# from data_structure import *
# import os
# import cv2
# from os.path import join as osj
# import shutil
# fileList = getFileList('Annotations')
# for txtname in fileList:
#     txtpath = osj('Annotations', txtname)
#     bmpname = txtname.replace('.xml', '.bmp')
#     bmppath = osj('Image', bmpname)
#     respath = osj('LabeledImage', bmpname)
#     data = filedata()
#     data.readtxt(txtpath)
#     img = cv2.imread(bmppath)
#     saveProcessImage(img, data.pageLines, respath)
#     #break
