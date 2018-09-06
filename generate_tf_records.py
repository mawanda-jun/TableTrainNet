"""
It generates the train and tf records that are set by dataset_costants.py file.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from dataset_costants import TABLE_DICT

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple
from dataset_costants import \
	TF_TRAIN_RECORD_TO_PATH, \
	TF_TRAIN_RECORD_NAME, \
	TF_TEST_RECORD_TO_PATH, \
	TF_TEST_RECORD_NAME, \
	PATH_TO_IMAGES, \
	TRAIN_CSV_TO_PATH, \
	TEST_CSV_TO_PATH, \
	TRAIN_CSV_NAME, \
	TEST_CSV_NAME

import logging
from logger import TimeHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TimeHandler().handler)


def class_text_to_int(row_label):
	"""
	Replace text with a int. Zero is for showing no boxes at all
	:param row_label:
	:return: returns corresponding int
	"""
	return 1 if row_label == TABLE_DICT['name'] else 0


def split(df, group):
	"""
	Splits name, sys function
	:param df:
	:param group:
	:return:
	"""
	data = namedtuple('data', ['filename', 'object'])
	gb = df.groupby(group)
	return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
	"""
	Creates the effective tf example given a group of data and a path.
	:param group: all the information from csv of a single image
	:param path: to images
	:return: a tf example (a single image tensorflow image)
	"""
	# looking for filename image of csv in path folder
	logger.debug('Now creating a single-page tf_example...')
	with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
		encoded_jpeg = fid.read()
		logger.debug('Image found in {}'.format(path))
	# encode it as a pillow image
	encoded_jpeg_io = io.BytesIO(encoded_jpeg)
	image = Image.open(encoded_jpeg_io)
	width, height = image.size

	filename = group.filename.encode('utf8')
	image_format = b'jpeg'
	xmins = []
	xmaxs = []
	ymins = []
	ymaxs = []
	classes_text = []  # class text that are written in costants.TABLE_DICT. In our case we have a single table lable
	classes = []  # class number

	# it now append the coordinates of the boxes inside csv file per image
	logger.debug('Appending csv {fn} infos to page tf_example...'.format(fn=filename))
	for index, row in group.object.iterrows():
		xmins.append(row['xmin']) / width
		xmaxs.append(row['xmax']) / width
		ymins.append(row['ymin']) / height
		ymaxs.append(row['ymax']) / height
		classes_text.append(row['class'].encode('utf8'))
		classes.append(class_text_to_int(row['class']))

	# create single tf_example object
	tf_example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': dataset_util.int64_feature(height),
		'image/width': dataset_util.int64_feature(width),
		'image/filename': dataset_util.bytes_feature(filename),
		'image/source_id': dataset_util.bytes_feature(filename),
		'image/encoded': dataset_util.bytes_feature(encoded_jpeg),
		'image/format': dataset_util.bytes_feature(image_format),
		'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
		'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}))
	logger.debug('Successfully created single-page tf_example')
	return tf_example


def main(_):
	# write train record:
	csv_input = os.path.join(TRAIN_CSV_TO_PATH, TRAIN_CSV_NAME)
	output_path = os.path.join(TF_TRAIN_RECORD_TO_PATH, TF_TRAIN_RECORD_NAME)
	writer = tf.python_io.TFRecordWriter(output_path)
	path = os.path.join(os.getcwd(), PATH_TO_IMAGES)
	# examples is the csv file
	examples = pd.read_csv(csv_input)
	grouped = split(examples, 'filename')
	logger.info('Now creating {}:'.format(TRAIN_CSV_NAME))
	for group in grouped:
		tf_example = create_tf_example(group, path)
		# append tf_example to the writer
		writer.write(tf_example.SerializeToString())
	writer.close()
	# output_path = os.path.join(os.getcwd(), TF_TRAIN_RECORD_TO_PATH)
	logger.info('Successfully created the TF train records: {}'.format(output_path))

	# write test record:
	csv_input = os.path.join(TEST_CSV_TO_PATH, TEST_CSV_NAME)
	output_path = os.path.join(TF_TEST_RECORD_TO_PATH, TF_TEST_RECORD_NAME)
	writer = tf.python_io.TFRecordWriter(output_path)
	path = os.path.join(os.getcwd(), PATH_TO_IMAGES)
	examples = pd.read_csv(csv_input)
	grouped = split(examples, 'filename')
	logger.info('Now creating {}:'.format(TEST_CSV_NAME))
	for group in grouped:
		tf_example = create_tf_example(group, path)
		writer.write(tf_example.SerializeToString())
	writer.close()
	# output_path = os.path.join(os.getcwd(), TF_TEST_RECORD_TO_PATH)
	logger.info('Successfully created the TF test record: {}'.format(output_path))


if __name__ == '__main__':
	tf.app.run()
