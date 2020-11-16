# Session 11 - GRU, Attention Mechanism & Transformers

[![Website](https://img.shields.io/badge/Website-blue.svg)](http://face-operations.s3-website-us-east-1.amazonaws.com)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S11/Translation_by_attention_using_encoder_decoder.ipynb)

This assignment is targeted to understand and implement Attention mechanism in Encoder/Decoder networks.
The original paper on [Bahdanau attention](https://arxiv.org/pdf/1409.0473.pdf).
The original notebook which forms the basis for this notebook can be found [here](https://bastings.github.io/annotated_encoder_decoder/)

All the code for deployment can be found in the [deployment](deployment) directory.

## Seq2Seq Translation with Bahdnau Attention

The model consists of following elements:

1. Encoder Class: Implemented as a bi-directional GRU
1. Decoder Class: Implemented as a conditional GRU, where the initial hidden state of the decoder is conditioned on the encoder's final vector
1. Attention Mechanism: An additive attention is used at every time-step of the decoder.  
1. Embeddings: PyTorch's nn.Embedding class is used to derive embeddings from the vocabulary.

PyTorch's IWSLT German-English Translation dataset has been used to train the model.
Adam optimizer with default settings (β1=0.9, β2=0.999 and ϵ=10−8). Cross-Entropy loss is used as the cost-function.

Training code can be found in this [notebook](https://colab.research.google.com/github/pankaj90382/TSAI-2/blob/master/S11/Translation_by_attention_using_encoder_decoder.ipynb)

### Parameters and Hyperparameters

- Loss Function: NLLLoss
- Epochs: 10
- Learning Rate: 0.0003
- Optimizer: Adam
- Batch Size: 64

## Results

Since the model was trained on a very small dataset like IWSLT (International Workshop on Spoken Language) Data Set(due to limitations on google colab), the results are not very accurate.

### Input Sentence

```
jeden tag nahmen wir einen anderen weg, sodass niemand erraten konnte, wohin wir gingen.
```

### Output Sentence

```
and every day every day , we took another one that could n't go anywhere . where we went .
```

## Attention Mechanism

A neural network is considered to be an effort to mimic human brain actions in a simplified manner. Attention Mechanism is also an attempt to implement the same action of selectively concentrating on a few relevant things, while ignoring others in deep neural networks.

An attention mechanism allows the model to focus on the currently most relevant part of the source sentence. In this project we implemented additive attention that was used in _Bahdanau et al._

<p align='center'>
    <img src="./attention.jpeg" width="430px" alt="attention mechanism" />  
</p>

## Refrences

 - [Pytorch Seq2Seq Tutorial Deployment](https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html)
 - [Annonated Encoder & Decoder](https://bastings.github.io/annotated_encoder_decoder/)
 - [Bastings GithubLink for Annoated Encoder and Decoder](https://github.com/bastings/annotated_encoder_decoder)
 - [NLP Basics Pytorch](https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html)
