# RNN Models
The RNN_models directory contains scripts and models related to Recurrent Neural Networks (RNNs) for sequence-based tasks. These models are designed to handle time-series data or sequential information, such as text or speech.

I’m a grade 12 student, and this project showcases my understanding of RNNs and their variants, including LSTMs and GRUs.

**Directory Structure**


## RNN_Model.py:
A script that implements a simple RNN model for sequence prediction. It demonstrates how to build and train a basic RNN using PyTorch.

## LSTM_Model.py:
A script showcasing the use of Long Short-Term Memory (LSTM) networks, a more advanced version of RNNs designed to handle long-term dependencies in data.

## GRU_Model.py:
A script for Gated Recurrent Units (GRUs), which are an alternative to LSTMs, offering a simpler architecture while maintaining similar performance in certain tasks.

## Difference Between RNN, LSTM, and GRU

### RNN (Recurrent Neural Network):

A basic RNN is designed to handle sequential data by maintaining a hidden state over time. However, RNNs struggle to capture long-term dependencies because of issues like the vanishing gradient problem. As a result, they may not perform well on tasks where the relationship between distant time steps is important.

### LSTM (Long Short-Term Memory):

LSTMs are a type of RNN designed to overcome the limitations of basic RNNs. They use a more complex architecture with gates (input, forget, and output gates) to control the flow of information. This allows them to retain information over longer sequences and capture long-term dependencies more effectively. LSTMs are commonly used in tasks like language modeling and machine translation.

### GRU (Gated Recurrent Unit):

GRUs are similar to LSTMs but have a simpler structure with fewer gates. They combine the input and forget gates into a single update gate, which makes them computationally more efficient. GRUs can perform similarly to LSTMs on many tasks, but their simpler architecture often makes them faster to train.

## Usage
Each script demonstrates different types of RNN architectures and their application in sequence-based problems. These models can be integrated into various projects involving time-series forecasting, natural language processing, or other sequential data tasks.
