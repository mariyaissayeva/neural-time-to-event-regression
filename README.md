# Neural Time-to-Event Regression

Comparing a shallow network against a deeper one on a time-to-event target.

## What the notebook does

The notebook loads 300,000 rows of a processed tabular dataset with 134 columns, drops the timestamp and other text columns, and keeps 120 numeric features. The target is a time-to-event value, the length of study measured in time steps. Because the data is ordered in time, the split is chronological rather than random: the first 70% for training, the next 15% for validation, and the final 15% for testing (210k / 45k / 45k rows). Features are standardized with a scaler fitted on the training portion only.

Two Keras models are then trained head to head with identical settings (Adam, mean squared error loss, early stopping on validation loss): a shallow MLP with layers of 128, 64, and 32 units, and a deeper network that extends the same stack with 16 and 8 unit layers. Both end in a single linear output. After training, each model is evaluated on the untouched test set with MAE and RMSE, the training curves are plotted, and the shallow model is saved and reloaded to confirm it works end to end.

## Results

The shallow MLP won: it reached an MAE of 54.4 and an RMSE of 74.6 on the test set, while the deeper network came in at 63.1 and 83.6. Added depth hurt rather than helped here, a useful reminder that on tabular data more layers mostly add ways to overfit.

## Tech

Python, TensorFlow / Keras, scikit-learn, pandas

## Data & trained models

Large data files and model weights are hosted on Google Drive:
https://drive.google.com/drive/folders/1QHk6sbWbRPGKOpGKMo3FWEW9V5gGDTcC 

## Notes

The shallow model won because the extra layers gave the deeper network more room to overfit without adding useful signal. The biggest limitation is that only two architectures were compared, with no simpler baseline to beat. Adding dropout, tuning hyperparameters, and testing a gradient boosting model like XGBoost would make the comparison stronger. Predicting time to an event can be used for equipment failure forecasting and maintenance planning.
