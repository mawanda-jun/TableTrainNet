"""
In this file the user can test some models and some other configurations for finding tables into some documents.
It takes the test images from PATHS_TO_TEST_IMAGE, the models from PATHS_TO_CKPTS, the evaluation scores from TEST_SCORES
and the number of classes from NUM_CLASSES.
# How the pipeline works
## Only bmp 3-channel allowed
Since the pre-trained NN comes from a RGB dataset, even if my training starts from B/W images, we cannot feed the model
with 8-bit depth images. So the first thing to do is to convert it into a bmp format to make it suitable for inference.
In future, when I will train a personal NN from scratch, then it will be possible to use only B/W images to infer.

## Loading the graph: should-not-be-touched part
Nothing important to do in this part: this part is a common practice to load the model in memory

## Detection: the nice part
This part is composed of various pieces:
### Transform the image into a numpy array
This function transform the image into a one-axis long array. This is done for Tensorflow reads the image in this way
### Retrieving results
The infer results are made of: boxes, scores, classes, num_detections, that are:
`boxes`: array[n_class][n_box][coordinates]. For example:
`boxes[0][0][0]=ymin, boxes[0][0][1]=xmin, boxes[0][0][2]=ymax, boxes[0][0][3]=xmax`
are the four coordinates of the first box of the first class
`scores`: array[n_box][score]. For example:
`scores[0][0]`
is the score of the first box
`classes` is the number of the classes that has been found. This is not so useful since we have only one class.
`num_detections` is the number of the detections done. In my experience this number is always equal to the max_detections set in the config file.

### We know our data: some assumptions and how make profit from them

The data we are going to analyze are insurance booklet. As the reader can see most of the pages have tables, however they never have text on their side.
We can assume this is a general behaviour. So we can take only the y_min and y_max coordinate and crop the page widely in its width.


In addition we can see a table as a set of sub-tables, so we can assume that the NN would see it in the same way. For this reason I've created the function
`keep_best_boxes`. This function takes the best boxes over SCORE score. As the boxes are ordered by higher score, we append the first one to the list
we want to return. Then we see if the second one is vertically overlapping the first one. If not, we append the box to the returning list. If so,
we take the minor y_min and the x_max coordinate between the two and we create a "merged" box. Then we check if this new box is overlapping some of the
previous one recursively.


This two assumptions let us considering more boxes and with very low scores to make our prediction better.

### Let's draw!

With the new boxes and their scores - even if not precise because of merged one - we can the boxes found.
This script provides some for loops over all the PATS_TO_TEST_IMAGE for every TEST_SCORES and every PATHS_TO_CKPTS that are present in
inference_costants.py. They are written in TEST_PATH and every folder will have the name of the bmp image you put as test.

"""
import numpy as np
import os
import tensorflow as tf
import errno
import copy
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from inference_costants import \
	PATHS_TO_TEST_IMAGE, \
	PATHS_TO_CKPTS, \
	TEST_SCORES, \
	NUM_CLASSES, \
	PATH_TO_LABELS, \
	BMP_IMAGE_TEST_TO_PATH

TEST_IMAGE_PATHS = PATHS_TO_TEST_IMAGE
PATH_TO_CKPTs = PATHS_TO_CKPTS
SCORES = TEST_SCORES


def from_png_to_bmp(png_path, output_path=BMP_IMAGE_TEST_TO_PATH):
	"""
	Convert a png_path image into a bmp 3-channel one and return the path to the converted image
	:param png_path: path of the image
	:param output_path: path in which we save the image
	:return: the file path
	"""
	# convert a .png image file to a .bmp image file using PIL
	file_name = os.path.splitext(png_path)[0] \
		.split("/")[-1]
	file_in = png_path
	img = Image.open(file_in)

	file_out = os.path.join(output_path, str(file_name), str(file_name) + '.bmp')
	len(img.split())  # test
	if len(img.split()) == 4:
		# prevent IOError: cannot write mode RGBA as BMP
		r, g, b, a = img.split()
		img = Image.merge("RGB", (r, g, b))
		img.save(file_out)
	else:
		img.save(file_out)
	return file_out


def check_if_vertically_overlapped(coord_a, coord_b):
	"""
	Return if coord_b is intersected vertically with coord_a in:
	top_b, bottom_b, b includes a, a includes b
	:param coord_a:
	:param coord_b:
	:return: true if intersected, false instead
	"""
	return \
		coord_a['y_min'] <= coord_b['y_min'] <= coord_a['y_max'] or \
		coord_a['y_min'] <= coord_b['y_max'] <= coord_a['y_max'] or \
		(coord_a['y_min'] > coord_b['y_min'] and coord_a['y_max'] < coord_b['y_max']) or \
		(coord_a['y_min'] < coord_b['y_min'] and coord_a['y_max'] > coord_b['y_max'])


def merge_vertically_overlapping_boxes(boxes):
	merged_boxes = [boxes[0]]
	flag = False
	for box in boxes[1:]:
		for m_box in merged_boxes:
			coord_m_box = {
				'y_min': m_box[0],
				'x_min': m_box[1],
				'y_max': m_box[2],
				'x_max': m_box[3]
			}
			coord_box = {
				'y_min': box[0],
				'x_min': box[1],
				'y_max': box[2],
				'x_max': box[3]
			}
			if check_if_vertically_overlapped(coord_m_box, coord_box):
				flag = True
				if m_box[0] > box[0]:
					m_box[0] = box[0]
				if m_box[2] < box[2]:
					m_box[2] = box[2]
		if not flag:
			merged_boxes.append(box)
	if flag:
		return merge_vertically_overlapping_boxes(merged_boxes)
	else:
		return merged_boxes


def check_if_intersected(coord_a, coord_b):
	"""
	Check if the rectangular b is not intersected with a
	:param coord_a: dict with {y_min, x_min, y_max, x_max}
	:param coord_b: same as coord_a
	:return: true if intersected, false instead
	"""
	return \
		coord_a['x_max'] > coord_b['x_min'] and \
		coord_a['x_min'] < coord_b['x_max'] and \
		coord_a['y_max'] > coord_b['y_min'] and \
		coord_a['y_min'] < coord_b['x_max']


def keep_no_intersected_boxes(boxes, scores, max_num_boxes=5, min_score=0.8):
	"""
		return a list of the max_num_boxes not overlapping boxes found in inference
		boxes are: box[0]=ymin, box[1]=xmin, box[2]=ymax, box[3]=xmax

		:param boxes: list of boxes found in inference
		:param scores: likelihood of the boxes
		:param max_num_boxes: max num of boxes to be saved
		:param min_score: min box score to check
		:return: list of the best not overlapping boxes
		"""

	kept_scores = []
	kept_boxes = []  # always keep the firs box, which is the best one.
	num_boxes = 0
	i = 0
	if scores[0] > min_score:
		kept_boxes.append(boxes[0])
		kept_scores.append(scores[0])
		num_boxes += 1
		i += 1
		for b in boxes[1:]:
			if num_boxes < max_num_boxes and scores[i] > min_score:
				intersected = False
				coord_b = {
					'y_min': b[0],
					'x_min': b[1],
					'y_max': b[2],
					'x_max': b[3]
				}
				for kb in kept_boxes:
					# checks if box score is high enough
					coord_kb = {
						'y_min': kb[0],
						'x_min': kb[1],
						'y_max': kb[2],
						'x_max': kb[3]
					}
					intersected = check_if_intersected(
						coord_a=coord_b,
						coord_b=coord_kb
					)
				if not intersected:
					kept_boxes.append(b)
					num_boxes += 1
					kept_scores.append(scores[i])

				i += 1
			else:
				break

		# kept_boxes = merge_vertically_overlapping_boxes(kept_boxes)
	else:
		kept_boxes = []

	return kept_boxes, kept_scores


def keep_best_boxes(boxes, scores, max_num_boxes=5, min_score=0.8):
	"""
	return a list of the max_num_boxes not overlapping boxes found in inference
	boxes are: box[0]=ymin, box[1]=xmin, box[2]=ymax, box[3]=xmax

	:param boxes: list of boxes found in inference
	:param scores: likelihood of the boxes
	:param max_num_boxes: max num of boxes to be saved
	:param min_score: min box score to check
	:return: list of the best boxes
	"""
	kept_scores = []
	kept_boxes = []
	num_boxes = 0
	if scores[0] > min_score:
		kept_boxes.append(boxes[0])  # always keep the firs box, which is the best one.
		kept_scores.append(scores[0])
		num_boxes += 1
		for b in boxes[1:]:
			if num_boxes < max_num_boxes and scores[num_boxes] > min_score:
				kept_boxes.append(b)
				num_boxes += 1
				kept_scores.append(scores[num_boxes])
			else:
				break
		# keep the overlapping boxes
		kept_boxes = merge_vertically_overlapping_boxes(kept_boxes)
	else:
		kept_boxes = []

	return kept_boxes, kept_scores


def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	# print(im_width)
	# zeros = np.zeros((im_height, im_width, 2))
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)
	# np_array = np.array(image.getdata()).reshape((im_height, im_width, 1)).astype(np.uint8)
	# np_array = np.concatenate((np_array, zeros), axis=2)
	# print(np_array.size)
	# print('ciao')
	# return np_array


if __name__ == '__main__':
	TEST_BMP_PATHS = []
	# ONLY BMP 3-CHANNEL ALLOWED
	# make a list of the paths to the bmp files. See description at the top.
	for test_path in TEST_IMAGE_PATHS:
		file_name = os.path.splitext(test_path)[0] \
			.split("/")[-1]

		try:
			if not os.path.isdir(file_name):
				os.makedirs(os.path.join(BMP_IMAGE_TEST_TO_PATH, file_name))
		except OSError as exc:  # Guard against race condition
			if exc.errno != errno.EEXIST:
				# logger.warning('Parent folder was not created correctly. Probably already present')
				raise

		file_path = from_png_to_bmp(
			png_path=test_path,
			output_path=BMP_IMAGE_TEST_TO_PATH
		)

		TEST_BMP_PATHS.append(file_path)

	for PATH_TO_CKPT in PATH_TO_CKPTs:
		graph_name = os.path.splitext(PATH_TO_CKPT)[0].split("/")[-3]		# take the name of the graph for renaming reasons
		graph_name = graph_name.replace("model__rcnn_inception_", "")		# the name was too long :)
		# since the frozen script makes the frozen graph always at the same place
		# PATH_TO_CKPT = PATH_TO_CKPT + '/frozen/frozen_inference_graph.pb'
		for test_path in TEST_BMP_PATHS:
			file_name = os.path.splitext(test_path)[0] \
				.split("\\")[-1]

			# LOADING THE GRAPH: SHOULD-NOT-BE-TOUCHED PART
			detection_graph = tf.Graph()
			with detection_graph.as_default():
				# other TF commands
				od_graph_def = tf.GraphDef()
				with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
					serialized_graph = fid.read()
					od_graph_def.ParseFromString(serialized_graph)
					tf.import_graph_def(od_graph_def, name='')

			# load label map
			label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
			categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
																							use_display_name=True)
			category_index = label_map_util.create_category_index(categories)

			# DETECTION: THE NICE PART

			with detection_graph.as_default():
				with tf.Session(graph=detection_graph) as sess:
					image = Image.open(test_path)
					image.convert()
					# the array based representation of the image will be used later in order to prepare the
					# result image with boxes and labels on it.
					image_np = load_image_into_numpy_array(image)
					# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
					image_np_expanded = np.expand_dims(image_np, axis=0)
					image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
					# Each box represents a part of the image where a particular object was detected.
					boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
					# Each score represent how level of confidence for each of the objects.
					# Score is shown on the result image, together with the class label.
					scores = detection_graph.get_tensor_by_name('detection_scores:0')
					classes = detection_graph.get_tensor_by_name('detection_classes:0')
					num_detections = detection_graph.get_tensor_by_name('num_detections:0')
					# Actual detection.
					(boxes, scores, classes, num_detections) = sess.run(
						[boxes, scores, classes, num_detections],
						feed_dict={image_tensor: image_np_expanded})

					# print('boxes: ', boxes[0])		# boxes[0][0][0]=ymin, boxes[0][0][1]=xmin, boxes[0][0][2]=ymax, boxes[0][0][3]=xmax
					print('scores', scores[0])		# scores[0][0]=score of first box
					# print('classes', classes[0])
					# print('num_detections', num_detections)
					# print(type(scores[0]))
					# scores = list(scores)
					# print(type(scores))

					boxes = boxes[0]
					scores = scores[0]

					# WE KNOW OUR DATA: SOME ASSUMPTIONS AND HOW TO MAKE PROFIT FROM THEM

					for SCORE in SCORES:
						# we merge the best vertical overlapping boxes
						best_boxes, best_scores = keep_best_boxes(
							boxes=boxes,
							scores=scores,
							max_num_boxes=10,
							min_score=SCORE
						)
						coords = []
						# print(best_boxes)
						if not best_boxes == []:
							for box in best_boxes:
								# print('prima')
								# print(box[0])
								coord = {
									'y_min': box[0],
									'x_min': box[1],
									'y_max': box[2],
									'x_max': box[3]
								}
								# print('What''s happening here?')
								coords.append(coord)
							# print(coords)
							# print(boxes[0][0][0], boxes[0][0][1], boxes[0][0][2], boxes[0][0][3])
							# image1 = copy.deepcopy(image)
							# image2 = copy.deepcopy(image)
							new_image = copy.deepcopy(image)
							merged = 0
							for coord in coords:
								vis_util.draw_bounding_box_on_image(
									new_image,
									coord['y_min'],
									0,
									coord['y_max'],
									1,
									color='red',
									# vis_util.STANDARD_COLORS[i*2],
									thickness=4
								)
								merged += 1
							path = os.path.join('data', str(file_name), "{fn}_merged_{sc}_{gn}.bmp".format(fn=file_name, sc=SCORE, gn=graph_name))
							new_image.save(path)
							print('{} box merged from found!'.format(merged))
						else:
							print('No boxes found')

					i = 0
					coords = []
					while scores[i] > 0.2:
						# print('prima')
						# print(box[0])
						coord = {
							'y_min': boxes[i][0],
							'x_min': boxes[i][1],
							'y_max': boxes[i][2],
							'x_max': boxes[i][3]
						}
						# print('What''s happening here?')
						coords.append(coord)
						i += 1
					# print(coords)
					# print(boxes[0][0][0], boxes[0][0][1], boxes[0][0][2], boxes[0][0][3])
					# image1 = copy.deepcopy(image)
					# image2 = copy.deepcopy(image)
					i = 0
					new_image = copy.deepcopy(image)
					for coord in coords:
						vis_util.draw_bounding_box_on_image(
							new_image,
							coord['y_min'],
							coord['x_min'],
							coord['y_max'],
							coord['x_max'],
							# color='red',
							vis_util.STANDARD_COLORS[10 + i],
							thickness=4
						)
						i += 1
						print(str(i) + ' new box')
					new_image.save(os.path.join('data', str(file_name), str(file_name) + '_filtered_0.2_' + str(graph_name) + '.bmp'))

					i = 0
					coords = []
					while scores[i] > 0.4:
						# print('prima')
						# print(box[0])
						coord = {
							'y_min': boxes[i][0],
							'x_min': boxes[i][1],
							'y_max': boxes[i][2],
							'x_max': boxes[i][3]
						}
						# print('What''s happening here?')
						coords.append(coord)
						i += 1

					i = 0
					new_image = copy.deepcopy(image)
					for coord in coords:
						vis_util.draw_bounding_box_on_image(
							new_image,
							coord['y_min'],
							coord['x_min'],
							coord['y_max'],
							coord['x_max'],
							# color='red',
							vis_util.STANDARD_COLORS[10 + i],
							thickness=4
						)
						i += 1
						print(str(i) + ' new box')
					new_image.save(os.path.join('data', str(file_name), str(file_name) + '_filtered_0.4_' + str(graph_name) + '.bmp'))

					i = 0
					coords = []
					while scores[i] > 0.6:
						# print('prima')
						# print(box[0])
						coord = {
							'y_min': boxes[i][0],
							'x_min': boxes[i][1],
							'y_max': boxes[i][2],
							'x_max': boxes[i][3]
						}
						# print('What''s happening here?')
						coords.append(coord)
						i += 1
					i = 0
					new_image = copy.deepcopy(image)
					for coord in coords:
						vis_util.draw_bounding_box_on_image(
							new_image,
							coord['y_min'],
							coord['x_min'],
							coord['y_max'],
							coord['x_max'],
							# color='red',
							vis_util.STANDARD_COLORS[10 + i],
							thickness=4
						)
						i += 1
						print(str(i) + ' new box')
					new_image.save(os.path.join('data', str(file_name), str(file_name) + '_filtered_0.6_' + str(graph_name) + '.bmp'))
