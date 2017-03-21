# TensorFlow Sentiment Analysis

### Brief
This code is an attempt at sentiment analysis. It takes a dataset of book reviews from Amazon along with the latent out of five star ratings. The model takes the review and tries to predict the rating. The words are stripped of the html tags, tokenised and padded before being passed to an embedding in the form of a word vector. The sentance of vectors is then passed to a LSTM, then to a fully connected layer, then to softmax for classification.

_It should be said that the model defined in model.py is based off the lstm found [here](https://github.com/tflearn/tflearn/tree/master/examples). There is also a tflearn lstm found in the tflearn_attempt.py, again based on the examples. Annoyingly, this scores much better accuracy after less training than my handcrafted tensorflow implementation._

### Usage
The first time you run the code the program will output:
```
Creating dataset and saving Pickle files.
```
This only happens once but will take a long time. It takes the datasets found in the data folder and turns html reviews and stars into preprocessed tokens and one-hot values. It also massively reduces the size of the dataset - it ensures there are an equal amount of reviews for every possible rating to reduce bias for a given five star rating.

I'm presuming that TFLearn is not desirable (as it is too high level,) but feel free to run that for the best results (~65% accuracy) by simply running:
```
$ python tflearn_attempt.py
```

**To train:**
```
$ python main.py --train
```
When you are finished training just quit the program. Early stopping is implemented in a vague manner in that it saves the model with the best accuracy.

**To test:**
```
$ python main.py --test
```
The checkpoint of the session with the highest accuracy will be loaded.

### Thoughts
This was my first attempt at Deep Learning within the context of NLP. The CNN found in the failed_attempts folder was based on [this tutorial](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/). Unfortunately it refused to learn. A little better is my implementation but it seems to stop learning at around the 40% accuracy mark. The best as priorly mentioned is the TFLearn example.
