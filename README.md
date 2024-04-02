Project Summary:

augmentations.py - contains the augmentations preformed on the images while training,
                                                            (we want the model to be indifferent to these augmentations)

models.py - contains the encoder and projector(for training) that comprise the VICReg model

example_VICReg.py - loads the models weights and preforms linear probing for classification, as well as choosing one
image at random from each class and retrieves the 5 closest imgs from the data

example_anomaly.py - loads the models weights and preforms anomaly detection and plots the ROC curve of the different
thresholds.

VICReg_Questions.py - contains different ablations on the VICReg model to understand the importance of each of its
three loss objectives (invariance, variance, covariance), as well as clustering tasks and evaluation.