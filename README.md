This contains codes I used for the Pover-T competition. 

Notebooks contains a jupyter notebook overview of the project
Scripts contains the models as well as the cross validation scripts

Main takeway:

I was able to improve the log loss 67% from a benchmark Random Forests model.

However, because it is an unbalanced data set, I think the competition metric is not an appropriate one.

For example: running CV_B.py will run an xgboost model, without resampling, and will result in a log loss of ~0.28. 

But the confusion matrix and recall and precision will show that it is all predicting False, the majority class.

Running CV_B_resampled.py runs a downsampled xgboost model, but will return a log loss of ~0.67, which is worse than before.

However, the confusion matrix, and recall and precision values show that this is infact a more useful model. It will return ~0.85 recall, and ~0.12 precision.

