# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was built by Kaleb Cook in October, 2026 for the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project. It is a Random Forest classifier from scikit-learn 1.5.1 with 100 trees, min_samples_leaf=5 and random_state=42. Categorical features are one-hot encoded and the label is binarized before training.

## Intended Use
The model predicts whether a person's annual income is above $50K from census attributes. It is intended for learning and demonstrating ML deployment practices. It should not be used to make real decisions about individuals, such as hiring, lending or housing.

## Training Data
The data is the UCI Adult dataset extracted from the 1994 US Census. 80% of the rows (26,048) were used for training, using a split stratified on the label. About 24% of people earn more than $50K, so the classes are imbalanced.

## Evaluation Data
The remaining 20% of the data (6,513 rows) was held out as the test set. It was processed with the encoder and label binarizer fitted on the training data.

## Metrics
The model was evaluated with precision, recall and F1 score. On the test set it scored a precision of 0.7847, a recall of 0.6135 and an F1 of 0.6886. This means that when the model predicts an income above $50K it is right about 78% of the time, and it finds about 61% of the people who actually earn above $50K. Performance was also measured on every value of each categorical feature, saved in slice_output.txt. Performance varies noticeably across slices. For education, the model does well on people with a Masters degree (F1 0.8723) but much worse on high school graduates (F1 0.4097). For HS-grad its precision is high (0.8087) but its recall is only 0.2743, so it misses most high earners in that group.

## Ethical Considerations
The data includes sensitive attributes such as race, sex and native country, and the model's performance differs across these groups. Predictions could reinforce historical income gaps if used for real decisions. Some slices have very few rows, so their metrics are unreliable.

## Caveats and Recommendations
The data is over 30 years old and does not reflect current incomes or inflation. The model has not been tuned for hyperparameters or for fairness. Before any real use, retrain on recent data, tune the decision threshold, and audit performance across demographic groups.