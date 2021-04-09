# Covid-19-and-Pneumonia-Classification-with-Deep-Learning
Using neural network, to create a classification model that outputs a diagnosis based on a patient’s X-ray scan.

#
The data is collected from Kaggle, which is Covid-19 Image Dataset with total three classification.

Step1: Load the dataset and preprossing the image data.

Step2: Building the model, it adds 2 CNN that have 6 filters and the size of each is 3x3 and Maxpooling layers before Flatting.
For the compiling, I set Adam optimizer and categoricalcrossentropy for loss function. Also, the Categoricalaccuracy is used for metrics. Since the dataset is balanced, AUC also used for testing accuracy of the metric.
Earlystopping is also used to catch 'val_auc' stopping increase.


Step3: ploting the figure from model fitting, I simplely use the history.

Result: The best performance of the model show that - loss: 0.5190
                                                      categorical_accuracy: 0.8367
                                                      auc: 0.9372 - val_loss: 0.4409
                                                      val_categorical_accuracy: 0.8333
                                                      val_auc: 0.9549

You also can try F1-score metrics and confusion metrics to see the evaulation of this model.
