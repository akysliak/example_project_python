These code, graphs (see report) and report were created in 2020 to approach 2 tasks in NLP.
The tasks in general (more details are omitted here) were as follows:

**Task 1**: 

implement Deep Learning models and a full pipeline (from reading data for
		training till evaluation of the trained models) to perform Slot Filling and
		Intent Classification for the given data; use 2 different types of pre-trained
		word embeddings, compare results.

**Task 2**: 

perform token classification in a very low-resource scenario: 
		a small amount of annotated in-domain training examples are available
		together with a bigger number of unannotated in-domain utterances,
		plus some annotated and unannotated data from several other domains can be used.
		(A classical Machine Learning method combined with word embeddings was used.) 

Data 

  Task1: train/dev/test sets in the format:
  - one utterance per line;
  - each example in the form:

			token1:<slot> token2:<slot>... tokenN:<slot> <=> <intent>

  Task2: annotated and unannotated in- and out-of-domain train data plus annotated test
		data, in the format:
  - one utterance per line;
  - annotated utterances in the the form:

			token1||<slot> token2||<slot>... tokenN||<slot><=><domain>

The folder contains the following files:

**task1.py** - an implementation for Task 1: contains the neural model and functions to perform the whole
pipeline from reading training/development/test data to the final evaluation of the trained models.
When run, performs all steps in the pipeline with 2 different pre-trained embeddings (GloVe and FastText),
outputs the evaluation results on the development set after each training epoch and final evaluation
results on the test set (after 10 epochs), stores corresponding predictions for the test set in a file
"task1_test_pred_[GloVe|FastText].txt".

**task2.py** - an implementation for Task 2: contains functions to build models which showed best results in
preliminary 10-fold cross-validation experiments as well as corresponding baselines.
When run, performs all the required steps from data reading to final evaluation, outputs
evaluation results on the test set and stores corresponding predictions for the test set in files
"task2_test_pred_[GloVe|FastTex]_[no_augm|with_augm].txt".

**task2_all_experiments.py** - contains implementation of the 10-fold cross-validation and other functions used
also by task2.py to build models and run experiments. When run, performs experiments for all considered parameter
combinations and kinds of experiments (cross-validation vs training on the full training set), for K=1 within kNN
(this can be changed in line 19).

**data_utils.py** - contains functions for reading and writing data as well as data analysis, used within task1.py,
task2.py, and task2_all_experiments.py. Can be run by itself to obtain data analysis.

**make_graphs.py** - contains functions to create plots for Task 1 and Task 2. When run, shows the plots.

All py-files can be executed with "python3 filename" command in the prompt. They do not require any arguments.
But the data folders have to be in the same directory.

**report_Kysliak.pdf** - the report for the assignment, for both tasks.
