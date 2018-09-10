# FAQ for TableTrainNet project.
I decided to write some explanation about the choices I have made for this project.

## Inference
### Do images need to be beautiful?
Not so much indeed.
The pre-trained neural network has been trained over normal, 24-bit depth 3-channels images.
It is not simple for it to understand that tables are _objects_, since for it those are lines that have not so much
correlation with others.
For this reason it is useful to apply
[this](https://www.researchgate.net/publication/320243569_Table_Detection_Using_Deep_Learning)
transformation, that let my NN perform much better.

### Is this black box?
What the neural network returns is a list of boxes and scores. In my opinion, it is not wise to 
use them without understanding what the NN is _seeing_.

I observed some pattern inside the insure policies I had to analyzed:
* tables are always single-column;
* they mostly contain number and text;
* every page hardly contains more then 4 tables each.

Since I could not prepare a specific dataset for this kind of pages due to lack of time, 
the inference is not so precise:
* it mistakes big tables and considering it as a set of ones; 
* scores rarely goes over 80%;

For this reason I decided to interpret boxes:
1. First step is to **consider the first 10 boxes** (`MAX_NUM_BOXES`) with **score over 0.4** (`MIN_SCORE`): this let me 
consider very imprecise boxes, which have some _kind of idea_ of where the boxes are;
2. Then **merge all vertical overlapping boxes**: if a box is overlapping another one there is good probability
that both of them are looking at the same table. So merging them - considering the min `y_min` and the max `y_max` of
the two - let me increase accuracy;
3. Crop widely: since tables are in single-column with no inline text I could consider only 2 parameters out of 4
(`y_min` and `y_max` and not `x_min` and `x_max`). This let me reduce inference error.

Then **no**, it is not a black box at all.
