# ML Fundamentals

Hands-on implementations of core machine learning and NLP techniques — built to understand what's happening underneath higher-level tools like LangChain/RAG pipelines, not just how to call an API.

This sits alongside [MomDigital](https://github.com/yogitamittal93/momdigital), where these fundamentals are applied in a production RAG system (embeddings, semantic search, LLM integration). These notebooks are the "from scratch" layer underneath that work.

---

## Notebooks

| Notebook | Focus |
|---|---|
| `CIFAR100_CNN.ipynb` | Image classification with a convolutional neural network (TensorFlow), trained on the CIFAR-100 dataset |
| `digits.ipynb` | Digit classification — core supervised learning fundamentals |
| `NLP_simpleSemanticAnalysis.ipynb` | A basic semantic similarity/meaning-comparison pipeline in NLP |
| `sentiment_analysis.ipynb` | Sentiment classification on text data |
| `twitterSentimentAnalysis.ipynb` | Sentiment analysis applied to tweet-style short text |
| `tensorflow.ipynb` | General TensorFlow workflow, using a public UCI dataset |

## Why this repo exists

Production AI work (like MomDigital's RAG pipeline) leans on embeddings, vector similarity, and LLM APIs — but it's easy to use those as black boxes. This repo is where I worked through the underlying mechanics directly: training a CNN instead of just fine-tuning one, building a semantic similarity pipeline by hand instead of only calling a hosted embedding API, and classifying sentiment with traditional NLP approaches before layering LLMs on top.

## Stack

TensorFlow, scikit-learn, standard NLP preprocessing — developed in Google Colab.
