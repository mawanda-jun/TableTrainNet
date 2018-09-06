The dataset is made up of Annotations in xml format regarding the Images, which are in bmp.
Please run
python img_to_jpeg.py
to convert those images in jpeg and grayscale them. This is needed for Tensorflow compatibility.

Then please run "generate_database_csv.py" to create a "train_jpeg.csv" and a "test_jpeg.csv" into the ../data/ folder to create two dedicated csv files.
The proportions are 0.7 train and 0.3 test, they can be changed editing the very first lines of the code.

