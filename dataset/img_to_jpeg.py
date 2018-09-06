"""
Generates a new jpeg 8-bit greyscale single channel image for every one in PATH_TO_IMAGES folder
"""
from PIL import Image
import cv2
import numpy as np
import os
from dataset_costants import \
	PATH_TO_IMAGES, \
	IMAGES_EXTENSION, \
	DPI_EXTRACTION
import pyprind
import logging
from logger import TimeHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TimeHandler().handler)


def uglify_image(pil_image):
	img = np.asarray(pil_image)
	img = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
	return Image.fromarray(img)


def img_to_jpeg(img_path):
	"""
	Transform every image in path into a png one for making it compatible with TF pre-trained NN
	:param img_path: path to image folder
	:return: None. It writes images on disk
	"""
	img_found_counter = 0
	img_converted_counter = 0
	for(gen_path, bmp_paths, img_names) in os.walk(img_path):
		bar = pyprind.ProgPercent(len(img_names))
		# print(gen_path, bmp_paths, img_names)
		for file_name in img_names:
			if not file_name.endswith(IMAGES_EXTENSION):
				file_no_extension = os.path.splitext(file_name)[0]
				# file_no_extension = file_name.replace('.bmp', '')
				img_found_counter += 1
				# if (file_no_extension + IMAGES_EXTENSION) not in img_names:
				if True:
					logger.info('Now processing: {}'.format(file_name))

					img = Image.open(os.path.join(gen_path, file_name)).convert('L')
					img = uglify_image(img).convert('L')
					img.save(os.path.join(gen_path, file_no_extension + IMAGES_EXTENSION), IMAGES_EXTENSION.replace('.', ''), dpi=(DPI_EXTRACTION, DPI_EXTRACTION))
					img_converted_counter += 1
					logger.info('{} succesfully written on disk!'.format(file_name))
			bar.update()
	if img_found_counter == 0:
		logger.warning('No img to convert found!')
	else:
		if img_converted_counter == 0:
			logger.info('No img to convert left!')


if __name__ == '__main__':
	img_to_jpeg(PATH_TO_IMAGES)
