# Session 9 - Neural Word Embedding

[![Website](https://img.shields.io/badge/Website-blue.svg)](http://face-operations.s3-website-us-east-1.amazonaws.com/)

The goal of this assignment is to perform sentiment analysis on movie reviews i.e. for a given review, the model will predict whether it is a positive review or a negative review. The model is deployed and can be tested [here](http://face-operations.s3-website-us-east-1.amazonaws.com).

## Different Approaches to Sentiment Analysis

We first try out different approaches and then choose the best one among them for deployment. We use this [repository](https://github.com/bentrevett/pytorch-sentiment-analysis) as reference for testing different approaches.

### 1 - Simple Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S9/1%20-%20Simple%20Sentiment%20Analysis.ipynb)

This tutorial covers the workflow of a PyTorch with TorchText project. We'll learn how to: load data, create train/test/validation splits, build a vocabulary, create data iterators, define a model and implement the train/evaluate/test loop. The model will be simple and achieve poor performance, but this will be improved in the subsequent tutorials. In this model words are encoded as one-hot vector and fed to a single RNN layer. This model attains test accuracy of just **47%**.

### 2 - Upgraded Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S9/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb)

Now we have the basic workflow covered, this tutorial will focus on improving our results. We'll cover: using packed padded sequences, loading and using pre-trained word embeddings, different optimizers, different RNN architectures, bi-directional RNNs, multi-layer (aka deep) RNNs and regularization. The script uses the approach to encodes  words using 100-dimensional GloVe embeddings and feeds them to a LSTM network. The model obtains a test accuracy of **85%**.

### 3 - Faster Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S9/3%20-%20Faster%20Sentiment%20Analysis.ipynb)

After we've covered all the fancy upgrades to RNNs, we'll look at a different approach that does not use RNNs. More specifically, we'll implement the model from Bag of Tricks for Efficient Text Classification. This simple model achieves comparable performance as the Upgraded Sentiment Analysis, but trains much faster.
In this approach, the following steps are performed:

1. Generate bigrams for each input sentence and append it to the end of the tokenized list.
2. Encode the words using 100-dimensional GloVe embeddings.
3. Use 2D average pooling with a filter size of `sentence_length x 1` on the embedding matrix.
4. Feed the output of the above step to a Fully Connected layer.

The model trains very fast and obtains a test accuracy of **85%**.

### 4 - Convolutional Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S9/Convolutional%20Sentiment%20Analysis.ipynb)

This approach uses Convolutional Neural Networks (CNNs) on word embeddings. CNNs can help the model to look at bi-grams (a 1x2 filter), tri-grams (a 1x3 filter) and/or n-grams (a 1x*n* filter) within the text. The intuition here is that the appearance of certain bi-grams, tri-grams and n-grams within the review will be a good indication of the final sentiment. The model obtains a test accuracy of **85%**.

### 5 - Transformers Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S9/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb)

In this approach, we use a pre-trained transformer model, specifically the BERT (Bidirectional Encoder Representations from Transformers) model to provide embeddings for the text. These embeddings are then fed to a GRU for making predictions. The model obtains a test accuracy of **92%**. Since the model is huge in size, it cannot be deployed on AWS lambda.

## 6 - Multi-Class Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S9/5%20-%20Multi-class%20Sentiment%20Analysis.ipynb)

Then we'll cover the case where we have more than 2 classes, as is common in NLP. We'll be using the CNN model from the previous notebook and a new dataset which has 6 classes.


## Results

We choose the `Upgraded Sentiment Analysis` model for depoyment. The code for deployment can be found [here](deployment).

Model output samples
| Input Text | Prediction |
| :----: | :----: |
| This film is great | positive |
| This film is terrible | negative |

## Refrences
  -  [Pytorch Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)
