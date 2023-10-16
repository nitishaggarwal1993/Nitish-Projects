## Data Pickles files made for modelling are as follows:

1. Dataset_50_all_708.pb = Images scalled to 50 X 50, with all classes sampled with 708 observations and then augmented to 2124 observation per class.
2. Dataset_50_notall_708.pb = Images scalled to 50 X 50, with only class 3 and 4 classes sampled with 708 observations and then augmented to 2124 observation and rest of the classes sampled with 2124 observations per class.
3. Dataset_224_all_708.pb = Images scalled to 224 X 224, with all classes sampled with 708 observations and then augmented to 2124 observation per class.
4. Dataset_224_notall_708.pb = Images scalled to 224 X 224, with only class 3 and 4 classes sampled with 708 observations and then augmented to 2124 observation and rest of the classes sampled with 2124 observations per class.
5. Dataset_50_grayscale_all_708.pb = Images greyscaled, then scalled to 50 X 50, with all classes sampled with 708 observations and then augmented to 2124 observation per class.
6. Dataset_50_grayscale_notall_708.pb = Images greyscaled and then scalled to 50 X 50, with only class 3 and 4 classes sampled with 708 observations and then augmented to 2124 observation and rest of the classes sampled with 2124 observations per class.
7. Testset_50_grayscale_normalized.pb = Representative image set created for XAI. Scalled imges at 50 X 50 and greyscalled. Contains 2 images per class.
8. Testset_50_normalized.pb = Representative image set created for XAI. Scalled imges at 50 X 50. Contains 2 images per class.
9. Testset_224_normalized.pb = Representative image set created for XAI. Scalled imges at 224 X 224. Contains 2 images per class.
10. Testset_224_for_final_model_testing_normalized.pb = Test image set created for final model testing. Scalled at 224 X 224. Contains 1206 images per class.

For creating these data pickle files pre processed images of 512 pixel radius were used.

Dataset_A1_50X50_all_708.ipynb and Dataset_A1_50X50_notall_708.ipynb were used for creating data pickle files and final test set pickle file and Teset_Data_processing.ipynb was used for creating XAI representative image pickle files.

## Image Processing

Image processing was done for the following reasons:
1. Rescale the images to have the same radius – 512 pixels.
2. Subtracted the local average colour; the local average gets mapped to 50% grey. 
3. Clipped the images to 90% size to remove the boundary effects.

data_preprocessing.ipynb is the code for undergoing Image processing for the training set of the data.

## Models and data pickle files used corresponding to the same.

First set of Experiments were done using Intensity Histohram and 50 X 50 Raw images (using pixel values of unprocessed images which were scalled to 50 X 50).

50X50_raw_image_dataset.ipynb and Histogram_code_for_50X50.ipynb files explains the modeling (Random Forest and XGBoost) done using 50X50_raw_dataset.pb, and histogram_dataset.pb dataset pickle files.

## Modeling using SHALLOW ML Models

SHallow_modeling.ipynb was done using Random Forest and XGBoost on Datasets Dataset_50_all_708.pb, Dataset_50_notall_708.pb, Dataset_50_grayscale_all_708.pb , Dataset_50_grayscale_notall_708.pb. For every model 5 fold cross validation was performed for which cv_predictions function is written. Then cross validation confussion matric and classification report is printed using the validation set. Post that using the whole dataset final models are trained.

## Modelling using Deep Learning

##### Class DRImageDatasetNew () was written for dataset initialisation for each model. train_epoch and valid_epoch functions were written for training and validating while cross validation. After cross validation for each of the model a confusion matrix and classification report was printed and then full model training was done on whole dataset.

**For ANN** - Datasets Dataset_50_all_708.pb, Dataset_50_notall_708.pb were used. 8 Layer netwrok was designed. Within this model architecture, the input dimension was 7500, corresponding to the three
channelled 50 X 50 images. Leaky ReLu was used as the activation function for each layer, as
it has a small slope for the negative values instead of being flat as in the case of ReLu. Furthermore,
Batch Normalization was done to normalize the output of each layer before pushing the output as
input in the next layer. So after pushing the data through eight hidden layers with varying
dimensions, in the end, softmax activation was applied to get the final probabilities for the
classes as this is a multiclass classification problem, and this was given as the input to the cross
entropy Loss function, which was used as the loss function for the classification task. ANN_50_NOTALL_708.ipynb and ANN_50_ALL_708.ipynb are the respective code files for the same.

**For CustomCNN** - Datasets Dataset_50_all_708.pb, Dataset_50_notall_708.pb, Dataset_224_all_708.pb and 
Dataset_224_notall_708.pb were used for modeling using CustomCNN model. A 5 convolutional and 3 fully connected layered model architecture wad designed for this. In this first, the input dimensions of either image of size 224 X 224 or 50 X 50 of three channels were passed through the model depending on the dataset used. Then, within each layer,
the convolution layer is encountered with a fixed kernel size of 3 X 3 but the varying number
of kernels for each layer, then the output for the same is passed through the ReLu activation
function and then Batch Normalization was used to normalize the output. Then the
output becomes the input for the Max Pooling layer, the size and stride of which is fixed for
all the layers, i.e. 2 X 2 is the size and stride of 2 is used. So, first, the data passes through
the five convolution layers, where the featurisation is done and then through two fully connected
layers. Again, within a fully connected layer, Batch Normalization is used for normalizing the
output of each layer, and the ReLu activation function is used. At the end of the fully connected
layer, the softmax activation function is applied to get the probabilities of the classes before
passing it into the loss function. The loss function used here is the Cross-Entropy Loss function
for performing the classification task. Also, dropouts are introduced in every convolution and
fully connected layer so as to prevent the model from getting overfitted. The dropout for
convolution is kept at 0.1, and for fully connected, it is kept at 0.25. CNN_50_NOT_ALL_708.ipynb, CNN_50_ALL_708.ipynb, CNN_224_NOT_ALL_708.ipynb and CNN_224_ALL_708.ipynb are the respective code files for the same.

**For ResNet18** -  Datasets Dataset_50_all_708.pb, Dataset_50_notall_708.pb, Dataset_224_all_708.pb and 
Dataset_224_notall_708.pb were used for modeling using ResNet18 model. Transfer learning with fine tuning (All the weights and baises were trained using the initial weights and baises of the final pretrained model which was trained on ImageNet dataset) was used for the same in which pretrained ResNet18 model was used with 3 fully connected layers at the end. In this first, the input image of dimension 224 x 224 or 50 x 50 of three channels
was passed, and then it went through the whole Resnet18 block where the featurisation takes place,
and then using automatic average pooling of 1 x 1, 512 feature maps were collected at the
end of the convolution using Resnet18. Then, two fully connected layers of 512 and 256 neurons,
respectively, were placed, and the ReLu activation function was used after the output of each
of these fully connected layers. Also, again, as done in custom CNN, batch normalization was
used after each fully connected layer to normalize the output before pushing through as input of
the next layer. At the end of the fully connected layer, the softmax activation function is
applied to get the probabilities of the classes before passing it into the loss function. The loss
function used here is the Cross-Entropy Loss function for performing the classification task.
Also, a dropout of 0.25 is introduced in every fully connected layer so as to prevent the model
from getting overfitted. Resnet18_50_NOT_ALL_708.ipynb, Resnet18_50_ALL_708.ipynb, Resnet18_224_NOT_ALL_708.ipynb, and Resnet18_224_ALL_708.ipynb are the respective code files for the same.

**For ResNet50** - Datasets Dataset_50_all_708.pb, Dataset_50_notall_708.pb, Dataset_224_all_708.pb and 
Dataset_224_notall_708.pb were used for modeling using ResNet50 model. Transfer learning with fine tuning (All the weights and baises were trained using the initial weights and baises of the final pretrained model which was trained on ImageNet dataset) was used for the same in which pretrained ResNet50 model was used with 3 fully connected layers at the end. In this also, the input image dimensions of 244 x 244 or 50 x 50 of three
channels were passed through the Resnet50 block, where the major featurization takes place and
then, as done for Resnet18, using automatic average pooling of 1 x 1, 2048 feature maps
were collected at the end of the convolution using Resnet50. Then, two fully connected layers
of 2048 and 1024 neurons, respectively, were placed, and the ReLu activation function was
used after the output of each of these fully connected layers. Also, again, as done in custom CNN
and Resnet18, batch normalization was used after each fully connected layer to normalize
the output before pushing through as input of the next layer. At the end of the fully connected layer, the softmax activation function is applied to get the probabilities of the classes before
passing it into the loss function. The loss function used here is the Cross-Entropy Loss function
for performing the classification task. Also, a dropout of 0.25 is introduced in every fully
connected layer so as to prevent the model from getting overfitted. Resnet50_50_NOT_ALL_708.ipynb, Resnet50_50_ALL_708.ipynb, Resnet50_224_NOT_ALL_708.ipynb, and Resnet50_224_ALL_708.ipynb are the respective code files for the same.

## DeepSHAP on winning Resnet50 model

DeepSHAP was used for Explaining the predictions made by the ResNet50 model. For the same Testset_224_normalized.pb dataset of 10 total images were used and results for one correct prediction(185_left), one incorrect prediction(9_left) and one shortcoming(8_left) were explored. Also, to investigate further IDRID dataset (XAI_224_deepshap_idrid_normalized.pb) was used to find the predictions using ResNet50 model and then making the explainations of the predistion and then one image (IDRID_03) from the same was also explored to compare the DeepSHAP results with the groundtruth images.

## Datasets Used

Kaggle EyePACS dataset was used for training and testing: https://www.kaggle.com/competitions/diabetic-retinopathy-detection

IDRID Dataset was used (only Section A) for further exploration of XAI using DeepSHAP: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid


```python

```
