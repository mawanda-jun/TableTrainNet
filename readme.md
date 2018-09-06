# TableTrainNet
## A simple project for training and testing table recognition in documents.
This project was developed to make a neural network which recognizes tables inside documents.
I needed an "intelligent" ocr for work, which could automatically recognize tables to treat them separately.

# General overview
The project uses the pre-trained neural network 
[offered](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
by Tensorflow. In addition, a 
[config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)
file was used, according to the pre-trained model choosen, to train with 
[object detections tensorflow API](https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-api)

The datasets was taken from:
* [ICDAR 2017 POD Competition](http://www.icst.pku.edu.cn/cpdp/ICDAR2017_PODCompetition/dataset.html).
* Not already implemented:
    * [UNLV dataset](https://github.com/tesseract-ocr/tesseract/wiki/UNLV-Testing-of-Tesseract#downloading-the-images)
    with its own
    [ground truth](http://www.iapr-tc11.org/mediawiki/index.php?title=Table_Ground_Truth_for_the_UW3_and_UNLV_datasets);
    * [Marmot Dataset](http://www.icst.pku.edu.cn/cpdp/data/marmot_data.htm)

# Required libraries
Before we go on make sure you have everything installed to be able to use the project:
* Python 3
* Tensorflow
* It's [object-detection API](https://github.com/tensorflow/models/tree/master/research/object_detection#tensorflow-object-detection-api)
* Pillow
* opencv-python
* pandas
* pyprind (useful for process bars)

# Project pipeline
The project is made up of different parts that acts as a pipeline.

### Take confidence with costants
I have prepared two "costants" files: `dataset_costants.py` and `inference_constants.py`.
The first contains all those costants that are useful to use to create dataset, the second to make
inference with the frozen graph. If you just want to run the project you should modify only those two files.
 
### Transform the images from RGB to single-channel 8-bit grayscale jpeg images
Since colors are not useful for table detection, we can convert all the images in `.jpeg` 8-bit single channel images.
Use `dataset/img_to_jpeg.py` [CHECKING THIS (fig1)](https://www.researchgate.net/publication/320243569_Table_Detection_Using_Deep_Learning)

### Prepare the dataset for Tensorflow
The dataset was take from 
[ICDAR 2017 POD Competition](http://www.icst.pku.edu.cn/cpdp/ICDAR2017_PODCompetition/dataset.html)
. It comes with a `xml` notation file with formulas, images and tables per image.
Tensorflow instead can build its own TFRecord from csv informations, so we need to convert
the `xml` files into a `csv` one.
Use `dataset/generate_database_csv.py` to do this conversion. More details are in the code.

### Generate TF records file
`csv` files and images are ready: now we need to create our TF record file to feed Tensorflow.
Use `generate_tf_records.py` to create two `.record` files that we will need later.

### Train the network
Inside `trained_models` there are some folders. In each one there are two files, a `.config` and a `.txt` one.
The first contains a tensorflow configuration, that has to be personalized:
* `fine_tune_checkpoint`: path to the frozen graph from pre-trained tensorflow models networks;
* `tf_record_input_reader`: path to the `train.record` and `test.record` file we created before;
* `label_map_path`: path to the labels of your dataset.

The latter contains the command to launch from `tensorflow/models/research/object-detection`
and follows this pattern:
```angular2html
python model_main.py \
--pipeline_config_path=path/to/your_config_file.config \
--model_dir=here/we/save/our/model" \ 
--num_train_steps=num_of_iterations \
--alsologtostderr
```
Other options are inside `tensorflow/models/research/object-detection/model_main.py`

### Prepare frozen graph
When the net has finished the training, you can export a frozen graph to make inference.
Tensorflow offers the utility: from `tensorflow/models/research/object-detection` run:
```angular2html
python export_inference_graph.py \ 
--input_type=image_tensor \
--pipeline_config_path=path/to/automatically/created/pipeline.config \ 
--trained_checkpoint_prefix=path/to/last/model.ckpt-xxx \
--output_directory=path/to/output/dir
```

### Test your graph!
Now that you have your graph you can try it out:
* create a folder in which you save your `bmp` test images
* modify the `inference_costants` accordingly
* run `inference_with_net.py`

Accordingly to `inference_costants`, this will show the best `MAX_NUM_BOXES` with score higher than `MIN_SCORE`
for all the `TEST_SCORES` and `PATH_TO_CKPTS` graphs.
In addition it will print a "merged" version of the boxes, in which
all the best vertically overlapping boxes are merged together to gain accuracy.
The procedure is better described in `inference_with_net.py`.

