# Automated-classification-of-emphysema
This project focuses on the automated classification of emphysema using deep learning models. We employed VGG16 and DenseNet121 architectures to classify emphysema patterns from lung CT scans. To enhance the training process and improve model accuracy, we used data augmentation techniques.


# Abstract
In modern medical diagnosis also, the emphysema is still recognized by the computed tomography (CT) scans with a set of
defined patterns as a classification problem in computer vision. There were as many algorithms developed in the past that
attempt to classify the underlying patterns and their relevant associated clusters by modeling an automated system. And
this classification modeling approach is responsible for the benchmarking classification and quantification of various
emphysematous tissues from lung CT images on different scales in the literature. Hence, with the same motivation and
intents, this article put forth a multiscale residual network with data augmentation model (MS-ResNet-DA). First, a
generative adversarial network (GAN) is employed to augment the training samples and avoid the overfitting problem.
These images are again augmented based on different image processing methods. Then, the obtained images are learned by
MS-ResNet to categorize the emphysema. Still, the accuracies of categorizing the centrilobular emphysema (CLE) and
panlobular emphysema (PLE) are not satisfactory because they do not have spatial dependence. So, an enhanced MSResNet-DA (EMS-ResNet-DA) model is proposed, which applies an effective position estimation algorithm to measure
relative and absolute location data of emphysema pixels in the images. The relative location data give the current location
of the emphysema pixel by extracting the relative dislocation measures from CT images. Also, the absolute location
estimation model is based on the position encoding network to match the diseased image with the reference emphysema
images and validate whether location data are implicitly learned when trained on categorical labels. Moreover, these
location data of all pixels in the images are learned by the MS-ResNet for emphysema classification. Finally, the
experimental results demonstrated that the EMS-ResNet-DA achieves an overall classification accuracy of 94.6% that
outclasses the conventional models.

