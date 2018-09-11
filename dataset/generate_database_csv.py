"""
This script generates a train/test database on the basis of the given percentage
it takes the images and the annotations written on the same folder, it shuffles them,
then copy into the upper train/test folders and create a relative csv file to manipulate
with TensorFlow.
More details are coming with the code.
"""

import pandas as pd
import os
import xml.etree.ElementTree as ET
from PIL import Image
import pyprind
from random import shuffle
from personal_errors import InputError, OutputError
from dataset_costants import \
    TRAINING_PERCENTAGE, \
    TABLE_DICT, \
    ANNOTATIONS_EXTENSION, \
    PATH_TO_ANNOTATIONS, \
    TRAIN_CSV_TO_PATH, \
    TRAIN_CSV_NAME, \
    TEST_CSV_TO_PATH, \
    TEST_CSV_NAME, \
    PATH_TO_IMAGES, \
    IMAGES_EXTENSION, \
    MIN_HEIGHT_BOX, \
    MIN_WIDTH_BOX

import logging
from logger import TimeHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(TimeHandler().handler)


def get_file_list_per_extension(path, ext):
    """
    Returns the folder and the list of the files with the 'ext' extension in the 'path' folder
    :param path:
    :param ext:
    :return: path list
    """
    ext_list = []
    for (gen_path, file_paths, file_names) in os.walk(path):
        for file in file_names:
            if file.endswith(ext):
                ext_list.append(file)
        return gen_path, ext_list


def sanitize_coord(coordinates, width, height):
    """
    points are: [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]
    it sanitize the coordinates that are extracted from a xml file. Valid for this dataset,
    to be updated in case the dataset changes
    Returning as dict: xmin, ymin, xmax, ymax
    :param coordinates:[[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]
    :return: dict with xmin, ymin, xmax, ymax coordinates
    """
    coordinates = coordinates.split()
    points = []
    for point in coordinates:
        point = point.split(',')
        points.append(point)
    new_points = {
        'xmin': points[0][0],
        'ymin': points[0][1],
        'xmax': points[3][0],
        'ymax': points[3][1]
    }
    # logger.info(new_points)
    # logger.info('width: {w}, height: {h}'.format(w=width, h=height))
    # check if coords are inverted
    if int(new_points['ymin']) > int(new_points['ymax']):
        logger.info('I found you y!')
        temp = int(new_points['ymin'])
        new_points['ymin'] = int(new_points['ymax'])
        new_points['ymax'] = temp
    if int(new_points['xmin']) > int(new_points['xmax']):
        logger.info('I found you x!')
        temp = new_points['xmin']
        new_points['xmin'] = int(new_points['xmax'])
        new_points['xmax'] = temp
    if int(new_points['ymin']) < 0:
        logger.info('Found some ymin at zero:')
        new_points['ymin'] = 0
    if int(new_points['xmin']) < 0:
        logger.info('Found some xmin at zero')
        new_points['xmin'] = 0
    if int(new_points['ymax']) > height:
        logger.info('Found some ymax beyond height: \nwidth: {w}, height: {h}\nnew_point["ymax"]: {npyx}' \
                    .format(w=width, h=height, npyx=new_points['ymax']))
        new_points['ymax'] = height
    if int(new_points['xmax']) > width:
        logger.info('Found some xmax beyond height: \nwidth: {w}, height: {h}\nnew_point["xmax"]: {npxx}' \
                    .format(w=width, h=height, npxx=new_points['xmax']))
        new_points['xmax'] = width

    if (int(new_points['xmax']) - int(new_points['xmin'])) < MIN_WIDTH_BOX or \
            (int(new_points['ymax']) - int(new_points['ymin'])) < MIN_HEIGHT_BOX:
        logger.info('Box {} was too small. Going to delete it'.format(new_points))
        new_points = None
    return new_points


def xml_to_csv(img_folder, img_list, xml_folder, xml_list):
    """
    it takes the file list and create a dedicated csv from the provided images with xml
    :param img_folder: path to jpeg folder
    :param img_list: list of files in jpeg folder
    :param xml_folder: path to xml folder
    :param xml_list: list of files in xml folder
    :return: csv dataframe with the right informations for tensorflow
    """
    logger.info('Generating csv from img list...')
    xml = []
    for img_file in img_list:
        bar.update()
        is_table = False
        img_name = img_file.replace(IMAGES_EXTENSION, '')
        xml_file = (img_name + ANNOTATIONS_EXTENSION)
        if xml_file not in xml_list:
            logger.warning('XML DESCRIPTION FILE NOT FOUND. PLEASE CHECK DATASET')
        tree = ET.parse(os.path.join(xml_folder, xml_file))
        root = tree.getroot()
        # img_name = img_file
        width, height = Image.open(os.path.join(img_folder, img_file)).size
        value = None
        for child in root.findall('.//tableRegion'):
            # if table is present, value will report the correct label and the coordinates of the boxes.
            # else, value will report img name, width and height but no other informations.
            if not is_table:
                is_table = True
            coords = child.find('.//Coords')
            coordinates = coords.get('points')
            points = sanitize_coord(coordinates, width, height)  # returning as dict: xmin, ymin, xmax, ymax
            if points is None:
                value = (img_file, 'no_table', 0, 0, 0, 0)
            else:
                # setting box as percentage of the image. This can be done in generate_tf_records also.
                xmin = int(points['xmin'])
                ymin = int(points['ymin'])
                xmax = int(points['xmax'])
                ymax = int(points['ymax'])
                value = (img_file, TABLE_DICT['name'], xmin, ymin, xmax, ymax)
            xml.append(value)
        if not is_table:
            value = (img_file, 'no_table', 0, 0, 0, 0)
            xml.append(value)
        logger.debug('Added new value: {}'.format(value))
    logger.info('CSV successfully generated!')
    # column_name columns must be remembered while generating tf records
    column_name = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml, columns=column_name)
    return xml_df


if __name__ == '__main__':
    # let's mix the jpeg files for creating a train/test uniform distribution
    # shuffle(img_list)  # de-shuffling list to make another distribution
    logger.info('Generating jpeg and xml file list...')
    # generate the xml and jpeg file list
    xml_folder, xml_list = get_file_list_per_extension(PATH_TO_ANNOTATIONS, ANNOTATIONS_EXTENSION)
    img_folder, img_list = get_file_list_per_extension(PATH_TO_IMAGES, IMAGES_EXTENSION)

    # create a bar for visual understanding of the progress
    bar = pyprind.ProgPercent(len(img_list) * 2)

    # prepare the size of train/test distribution
    n = len(img_list)
    limiter = n * TRAINING_PERCENTAGE

    i = 0
    training_list = []
    test_list = []

    # copy each jpeg/xml file into one upper folder, for train and test separately and
    # create two separate list for testing and training
    for jpeg in img_list:
        if i < limiter:
            training_list.append(jpeg)
            i = i + 1
        else:
            test_list.append(jpeg)
        bar.update()

    # creating csv file
    logger.info('Creating csv file for train and test...')
    xml_df_training = xml_to_csv(img_folder, training_list, xml_folder, xml_list)
    xml_df_test = xml_to_csv(img_folder, test_list, xml_folder, xml_list)
    # xml_df = xml_to_csv(img_folder, img_list, xml_folder, xml_list)
    logger.info('Writing train csv file at: {}'.format(os.path.join(TRAIN_CSV_TO_PATH, TRAIN_CSV_NAME)))
    xml_df_training.to_csv(os.path.join(TRAIN_CSV_TO_PATH, TRAIN_CSV_NAME), index=None)
    logger.info('Writing test csv file at: {}'.format(os.path.join(TEST_CSV_TO_PATH, TEST_CSV_NAME)))
    xml_df_test.to_csv(os.path.join(TEST_CSV_TO_PATH, TEST_CSV_NAME), index=None)
    # xml_df.to_csv('../data/all_labels_map_jpeg')
    logger.info('Successfully converted xml to csv.')
# logger.info('SOME EXCEPTIONS TO MANAGE WHILE WRITING FILES')
