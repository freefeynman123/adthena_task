# How to train and evaluate

install poetry:

`pip install poetry`

run following commands:

`poetry install`

`poetry shell`

to run training:

`python src/adthena_task/training/main.py`

to run evaluation:

`python src/adthena_task/eval/eval_on_test_file.py`

In order to change configuration paths and other hyperparameters you should change `config.py` file.
Specifically to train on new dataset one should change `DATA_DIR_TRAIN` variable. To evaluate on new dataset one
should change `DATA_DIR_TEST` variable. By default they are placed in `data` folder, which is ignored in commits.
By default checkpoints are saved in `eval` folder, path to the checkpoint that should be used during evaluation can
also be changed in `config.py` file.

You can also run evaluation of your custom queries by going to `eval` folder and running fastapi app with:

`uvicorn fast:app --port 9999`

and run the application by going to:

`http://127.0.0.1:9999/docs`

or predict directly from browser by typing `http://127.0.0.1:9999/predict/{sentence}`
where `{sentence}` should be replaced with custom text.

# Points description

1.    Description of the model and justification of its usage:

I chose a model from BERT (Biderectional Encoder Representations) family, which is considered as a state-of-the-art model
in many natural language processing tasks. In order to compare it with some baseline,
the following procedure was chosen:

* Try some simple, well-established solution (Doc2Vec embeddings + Logistic Regression).
* Compare it with BERT.

Since full dataset takes some time to preprocess and train on and due to the fact that LogisticRegression on full embedded
dataset had difficulties with convergence, comparison on the custom dataset was performed, which consisted of labels, which
occurred more than 600 times (arbitrary number based on EDA analysis). The results from validation set on those two datasets
are as follows:

Doc2Vec + Logit :

| Metric      | Value |
| ----------- | ----------- |
| Validation accuracy      | 0.618       |
| Validation F1 score   | 0.621        |
| Testing accuracy      | 0.619       |
| Testing F1 score   | 0.620        |


BERT with 2 unfrozen encoder layers:

| Metric      | Value |
| ----------- | ----------- |
| Validation accuracy      | 0.850       |
| Validation F1 score   | 0.804        |
| Testing accuracy      | 0.852       |
| Testing F1 score   | 0.807        |

The split was done in a `50:25:25` manner, in order to reflect the split that was performed on whole dataset. Justification
of this fact is presented in EDA.ipynb notebook and comes from found class imbalance.

Moreoever using BERT gives us an access to pretrained embeddings and encoders, allowing us to perform transfer learning,
which is not possible with custom Doc2Vec embedding, which additionally could be prone to overfitting. Moreoever, Doc2Vec
has inherent randomness, which can influence reproducibility of results for this dataset.

<span>2.</span> Preprocessing for the BERT model was done in a two-stage scenario

* Basic preprocessing which consisted of operations such as getting rid of numbers, interpunction and some special symbols
like ampersand.
* Utilizing BERT tokenizer with pretrained embedding. Since the whole sentence is considered as a one vector and the hidden
state of the first token is taken to represent the whole sentence, an additional token must be added, which is usually
named CLS and placed at the beginning.

Max length of tokens needs to be decided

<span>3.</span> Methods for evaluation.

Evaluation was performed with two metrics: accuracy and weighted F1score. First of the mentioned metrics is a standard
method to obtain performance of our model, actually quite well suited in case of our dataset, since there is no single class that dominates the
rest of the data. Due to the fact that some part of labels are less present in our data, weighed F1 score was used,
however when one class in completely misclassified, the result is equal to 0, which was the case
for quite a lot of training and validation epochs. It tells us that there are some harder classes that the model has problem with
and it is an issue to address later.

<span>4.</span> Runtime overhead.

The model takes around 15 minutes to train 1 epoch on 1080Ti GPU with 16GB of RAM and 12 CPU cores,
so 20 epoch training lasts about 5 hours. In inference
time on test set with about 60K records it takes around 5 minutes to successfully run the model, however quite a lot of it
is taken to run preprocessing. In order to see training memory usage and other training metrics you can visit
Weights and Biases project and System tab for each run that are described later.

<span>5.</span> Weaknesses of the model and possible improvements.

Weaknesses:

* The model was trained on 50% of training data only due to imbalanced classes.
* It may have be too large for given problem, which results in increased time of inference and resources overhead.

Improvements:

* Improving the embedding - take into account specific words that are found in the dataset.
* Sparsify and prune the model - use frameworks such as rasa https://github.com/RasaHQ/rasa/tree/compressing-bert to
speed up the inference with similar quality of predictions.
* Take into account self-supervised features - improvements such as adding self-supervised attention to BERT have shown
that for example in case of "obvious" sentences the performance is improved (such as in this paper:
https://arxiv.org/pdf/2004.03808.pdf)
* Add data with similar queries to those found in trainSet.csv, especially for underrepresented classes.
* Add tests and Dockerfile to the project.
* Perform hyperparameter tuning - find optimal parameters such as learning rate or weight decay that are the best for
given task. It may be accomplished by such frameworks as Weights and Biases sweeps.

## Weights and biases runs

In the link below there are two runs present for BERT model:

https://wandb.ai/freefeynman123/adthena_task?workspace=user-freefeynman123

`bert_600_plus` shows results from training on smaller dataset that was used as a baseline comparison.
`bert_full_data` shows results of training on full data.


Project adthena_task initialized with `prose`.
