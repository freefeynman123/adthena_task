# adthena_task

1. Description of the model and justification of its usage:

I chose a model from BERT () family, which is considered as a state-of-the-art model in many natural language processing tasks.
In order to compare the model with some baseline, we chose to perform the comparison in following manner:

* Try some simple, well-established solution (Doc2Vec embeddings + Logistic Regression)
* Compare it with BERT.

Since full dataset takes some time to preprocess and train on and due to the fact that LogisticRegression on full embedded
dataset had diffuculties with convergence, comparison on the custom dataset was performed, which consisted of labels, which
occured more than 600 times (arbitrary number based on EDA analysis). The results from validation set on those two datasets
are as followis:

Doc2Vec + Logit :

Validation accuracy 0.618
Validation F1 score: 0.621
Testing accuracy 0.619
Testing F1 score: 0.62

BERT with 2 unfrozen encoder layers:

The split was done in a 50:25:25 manner, in order to reflect the split that was performed on whole dataset. Justification
of this fact is presented in EDA.ipynb notebook and comes from found class imbalance.

2. Preprocessing for the BERT model was done in a two-state scenario:

*
*

3. Evaluation was performed with two metrics: accuracy and weighted f1score.

4. The model takes around 15 minutes to train 1 epoch on 1080Ti GPU, so 20 epoch training lasts about 5 hours. In inference
time on my local

5. Weaknesses:

* Improving the embedding - describe
* Sparsify and prune the model - describe
* Take into account self-supervised features
* Add data with similar queries to those found in trainSet.csv, especially for underrepresented classes.

TODO:
* Eval script for provided .txt file

PLANS:
* Dockerfile for train and eval
* Detailed task description.
* Unfix some of the hiperparameters.

Project adthena_task initialized with `prose`.
