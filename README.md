<h2>TensorFlow-FlexUNet-Image-Segmentation-Ridge-in-Retinopathy-of-Prematurity (2025/10/26)</h2>
<!--
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
 -->
This is the first experiment of Image Segmentation for 
<b>Ridge-in-Retinopathy-of-Prematurity (HVDROPDB-RIDGE) </b>, based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
 and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1h-RYrq3zd69z0-FWnMrW-SIVAD6yZJQF/view?usp=sharing">
Augmented-HVDROPDB-Ridge-ImageMask-Dataset.zip</a>
, which was derived by us from <br><br> <b>HVDROPDB-RIDGE</b> subset of <b>HVDROPDB_RetCam_Neo_Segmentation</b> in  
<a href="https://data.mendeley.com/datasets/xw5xc7xrmp/3">
HVDROPDB Datasets for Classification and Segmentation for Research in Retinopathy of Prematurity, Ranjana Agrawal
</a>
<br>
<br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of <b>HVDROPDB-RIDGE</b> subset of <b>HVDROPDB_RetCam_Neo_Segmentation</b>, 
we used our offline augmentation tool <a href="https://github.com/sarah-antillia/ImageMask-Dataset-Offline-Augmentation-Tool"> 
ImageMask-Dataset-Offline-Augmentation-Tool</a> and 
<a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool <a> to augment the subset.
<br><br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
our dataset appear similar to the ground truth masks.<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1001_0.3_0.3_Neo_12.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1001_0.3_0.3_Neo_12.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1001_0.3_0.3_Neo_12.png" width="320" height="320"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1001_0.3_0.3_RetCam_10.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1001_0.3_0.3_RetCam_10.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1001_0.3_0.3_RetCam_10.png" width="320" height="320"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1002_0.3_0.3_RetCam_9.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1002_0.3_0.3_RetCam_9.png" width="320" height="320"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1002_0.3_0.3_RetCam_9.png" width="320" height="320"></td>
</tr>
</table>

<hr>
<br>

<h3>1. Dataset Citation</h3>
The dataset used here was obtained from the following 
<a href="https://data.mendeley.com/datasets/xw5xc7xrmp/3">
HVDROPDB Datasets for Classification and Segmentation for Research in Retinopathy of Prematurity, Ranjana Agrawal
</a>
<br><br>
<b>Contributors:</b><br>
Ranjana Agrawal,Sucheta Kulkarni
<br><br>

<b>Description</b><br>
HVDROPDB_RetCam_Neo_Segmentation and HVDROPDB_RetCam_Neo_Classification are the first datasets to be published for the retinal structure 
segmentation to identify the Retinopathy of Prematurity (ROP). <br>
They are prepared by screening the preterm infants visiting PBMA's H.V. Desai Eye Hospital, Pune with two diverse imaging systems 
RetCam and Neo.  The Segmentation dataset contains sub-datasets for the segmentation of optic disc, blood vessels, 
and demarcation line/ridge from the fundus images of preterm infants, annotated by a group of ROP experts. <br>
Each sub-dataset contains retinal fundus images of premature infants with the ground truths prepared manually to assist 
researchers in developing an explainable automated ROP screening system. <br>
The Classification sub-datasets contain ROP and Normal images.
<br>
<br>
<b>Related Identifiers*</b><br>
<a href="https://www.sciencedirect.com/science/article/pii/S2352340923009010?via%3Dihub">
HVDROPDB datasets for research in retinopathy of prematurity</a>
<br><br>

<b>Licence</b><br>
<a href="https://interoperable-europe.ec.europa.eu/licence/creative-commons-attribution-40-international-cc-40">
CC BY 4.0
</a>
<br>
<br>
<h3>
2 HVDROPDB Ridge ImageMask Dataset
</h3>
<h4>2.1 Download HVDROPDB-Ridge-PNG-ImageMask-Dataset</h4>
 If you would like to train this HVDROPDB-Ridge Segmentation model by yourself,
 please download  our dataset <a href="https://drive.google.com/file/d/1h-RYrq3zd69z0-FWnMrW-SIVAD6yZJQF/view?usp=sharing">
 Augmented-HVDROPDB-Ridge-PNG-ImageMask-Dataset.zip  </a> on the google drive
, expand the downloaded and put it under <b>./dataset</b> folder to be.<br>
<pre>
./dataset
└─HVDROPDB-Ridge
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>HVDROPDB-Ridge Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/HVDROPDB-Ridge_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not large  to use for a training set of our segmentation model.
<br>
<br>
<h4>2.2 HVDROPDB Ridge Dataset Derivation</h4>
The original dataset folder structure <b>HVDROPDB_RetCam_Neo_Segmentation</b> is the following.<br>
<pre>
./HVDROPDB_RetCam_Neo_Segmentation
├─HVDROPDB-BV
├─HVDROPDB-OD
└─HVDROPDB-RIDGE
</pre>
We derived the Augmented HVDROPDB-Ridge  2 classes (Neo and RetCam) dataset from the following <b>HVDROPDB-RIDGE</b> subset of <b>HVDROPDB_RetCam_Neo_Segmentation</b> .
<pre>
./HVDROPDB-RIDGE
├─Neo_Ridge_images
├─Neo_Ridge_masks
├─RetCam_Ridge_images
└─RetCam_Ridge_masks
</pre>

<h4>2.3 Train Dataset</h4>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained HVDROPDB-Ridge TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/HVDROPDB-Ridgeand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small base_filters=16 and large base_kernels=(9,9) for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 3

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learning_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for HVDROPDB-Ridge 1+2 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;                     Neo:yellow,   RetCam: cyan
rgb_map = {(0,0,0):0,(255,255,0):1, (0,255,255):2,}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 30,31 32)</b><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 61,62,63)</b><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was terminated at epoch 63.<br><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/train_console_output_at_epoch63.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/HVDROPDB-Ridge</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for HVDROPDB-Ridge.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/evaluate_console_output_at_epoch63.png" width="720" height="auto">
<br><br>Image-Segmentation-HVDROPDB-Ridge

<a href="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/evaluation.csv">evaluation.csv</a><br>

The loss (categorical_crossentropy) to this HVDROPDB-Ridge/test was low, and dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.0125
dice_coef_multiclass,0.9939
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/HVDROPDB-Ridge</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowUNet model for HVDROPDB-Ridge.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1001_0.3_0.3_Neo_27.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1001_0.3_0.3_Neo_27.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1001_0.3_0.3_Neo_27.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1001_0.3_0.3_Neo_48.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1001_0.3_0.3_Neo_48.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1001_0.3_0.3_Neo_48.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1002_0.3_0.3_Neo_27.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1002_0.3_0.3_Neo_27.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1002_0.3_0.3_Neo_27.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1001_0.3_0.3_RetCam_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1001_0.3_0.3_RetCam_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1001_0.3_0.3_RetCam_4.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1001_0.3_0.3_RetCam_23.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1001_0.3_0.3_RetCam_23.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1001_0.3_0.3_RetCam_23.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/images/barrdistorted_1002_0.3_0.3_RetCam_19.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test/masks/barrdistorted_1002_0.3_0.3_RetCam_19.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HVDROPDB-Ridge/mini_test_output/barrdistorted_1002_0.3_0.3_RetCam_19.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Assistive Framework for Automatic Detection of All the Zones in Retinopathy of Prematurity Using Deep Learning</b><br>
Ranjana Agrawal, Sucheta Kulkarni, Rahee Walambe & Ketan Kotecha<br>
<a href="https://link.springer.com/article/10.1007/s10278-021-00477-8">https://link.springer.com/article/10.1007/s10278-021-00477-8</a>
<br>
<br>
<b>2. Retinopathy of Prematurity</b><br>
Ranjana Agrawal, Sucheta Kulkarni, Rahee Walambe & Ketan Kotecha<br>
<a href="https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/retinopathy-prematurity">
https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/retinopathy-prematurity</a>
<br>
<br>
<b>3.HVDROPDB datasets for research in retinopathy of prematurity</b><br>
Ranjana Agrawal, Rahee Walambe, Ketan Kotecha, Anita Gaikwad, 
Col. Madan Deshpande, Sucheta Kulkarni
<br>
<a href="https://www.sciencedirect.com/science/article/pii/S2352340923009010?via%3Dihub">
https://www.sciencedirect.com/science/article/pii/S2352340923009010?via%3Dihub
</a>
<br>
<br>

