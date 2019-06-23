# Peptic-Ulcer-Food-Recommendation
Basic Food recommendation system for peptic ulcer patients.

This a Flask powered Application, here we are predicting if a particular food is good or bad for ulcer patients.
For foods not trained or unknown sources, we save to the database and wait for when next to retrain our models.

KNN was used for the training of our models, Realtime prediction was achieved with loading of our saved trained model, application is powered by flask

Updates to make:
================
* Train demo dataset with other models eg (SVM, Naive Byes e.t.c)
* Evaluation metrics with other models, check (Confusion matrix, recall, precision, F1-Score)
* Model boosting with GridSearch

This project shows how Machine learning is used in Health and how its integratable with existing solutions.

## NOTE:
### THE DATA USED ARE FROM MULTIPLE ONLINE SOURCES, NONE ARE VERIFIED FROM NUTRITIONIST OR CERTIFIED DOCTORS.
