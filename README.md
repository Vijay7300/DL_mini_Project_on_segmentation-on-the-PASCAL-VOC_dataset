# DL_mini_Project_on_segmentation-on-the-PASCAL-VOC_dataset

PASCAL VOC Efficient Segmentation Challenge
The objective of this mini competition is to design, train, and evaluate a computationally lightweight
deep learning model for semantic segmentation. We will use the PASCAL VOC 2012 dataset. The
goal is not only to achieve high segmentation accuracy but also to develop a model that is
computationally efficient at inference time, highlighting the important trade-off between accuracy
and efficiency that arises in real-world deployment of deep learning systems. All implementations
must use the PyTorch framework exclusively.
You are required to submit an end-to-end semantic segmentation model that directly maps an
input RGB image to a pixel-wise segmentation mask. Given an input image from the PASCAL VOC
dataset with shape (3, 300, 300), the model must output a segmentation mask of shape (300, 300),
where each pixel is assigned an integer class label in the range [0, 20], corresponding to the 21
classes in the dataset (including background). The model must be fully end-to-end, meaning that the
segmentation mask must be produced directly by the neural network forward pass, without relying
on any external post-processing steps or auxiliary pipelines. The final output must be a pixel-wise
segmentation mask with integer class indices. You may design/ use any deep neural network model.
In addition to accuracy and efficiency, robustness is a mandatory requirement for this competition.
In real-world scenarios, input images are often affected by noise, compression artifacts, illumination
changes, or minor corruptions. Therefore, at inference time, your model must demonstrate
reasonable robustness when evaluated on noisy or corrupted images. These corruptions may
include additive Gaussian noise, salt-and-pepper noise, mild blur or compression artifacts, and
intensity or contrast variations.
Submissions will be evaluated using two metrics: segmentation accuracy and computational
efficiency. Segmentation quality will be measured using the Dice Similarity Coefficient (DSC), where
a higher Dice score indicates better performance. The dice score will be computed as a
macro-average over all 21 classes, including the background class, with equal weightage assigned to
every class. Computational efficiency will be measured by the number of floating-point operations
(FLOPs) required during a single forward pass (i.e., for a single test image) at inference time, with
lower FLOPs indicating a more efficient model. The final ranking will consider both Dice score and
FLOPs, and models achieving a better balance between accuracy and efficiency will rank higher.
Each team will have to submit its full codebase (including codes for training, validation, and testing).
For training, split the original PASCAL VOC 2012 training dataset into train and validation set using
80:20 ratio. You can also augment training data with its noisy/ corrupted versions. No validation
data from original PASCAL VOC 2012 should be used during any steps of training. It will be used as
test set for evaluating the final model. Use of test (i.e. original PASCAL VOC validation split) during
training (including hyperparameter tuning) will lead to disqualification of the team from the
competition and they will not be awarded any marks. The inference codebase should provide the
user with an option to specify the name of the folder containing the test images. Once the folder is

specified, the inference code must produce an output folder containing the segmented test images.
The name of the folder should be your group number_output. Each segmented image file within the
folder should have the name original image name_mask.
Important Note: Two separate leaderboards will be maintained to showcase the results of different
groups in terms of (i) segmentation performance in terms of DSC, and (ii) Flops per test image on
the PASCAL VOC 2012 official validation dataset. Results on the leaderboard will be just for
indication purposes. The leaderboard will not be used for final grading. Final grading will be
performed based on the performance of the models on a dataset, which will be released after the
submission deadline. This dataset may contain noisy or corrupted images.
Dataset download link (You can use other sources as well, but ensure that it contains all the images
from PASCAL VOC 2012 dataset).
