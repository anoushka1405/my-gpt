# My GPT : A Transformer Language Model Built from Scratch

A full implementation of a GPT-style language model built from first principles, including neural network primitives, attention mechanisms, tokenizer, training pipeline, and text generation.

## Overview

This project is a ground-up reconstruction of a GPT-style transformer model, built to deeply understand how modern large language models work internally.

Instead of using high-level abstractions, every major component is implemented manually — from gradient descent and backpropagation to self-attention and text generation.

The system follows a full progression:

**Math Foundations → Neural Networks → NLP Pipeline → Transformer Architecture → GPT Model**

## Key Features

- Neural network primitives implemented from scratch (forward + backprop)
- Gradient descent training system
- Activation functions (ReLU, Sigmoid, Softmax)
- Byte-Pair Encoding (BPE) tokenizer
- Full dataset + batching pipeline
- Self-attention and multi-head attention
- Transformer block implementation
- GPT-style autoregressive language model
- KV-cache for faster inference
- Grouped Query Attention (GQA)
- Text generation with sampling


## Project Structure

```
model/          Attention, Transformer, GPT architecture
  attention.py             Self-attention head
  multi_head_attention.py  Multi-headed attention
  transformer.py           Transformer block
  gpt.py                   GPT model
  normalization.py         Layer normalization
  batch_normalization.py   Batch normalization
  rms_normalization.py     RMS normalization
  embeddings.py            Word embeddings
  positional_encoding.py   Positional encoding
  kv_cache.py              KV-Cache for fast inference
  grouped_query_attention.py  Grouped query attention

data/           Data pipeline
  tokenizer.py                BPE tokenizer
  vocab.py                    Character-level vocabulary
  loader.py                   Batched training data loader
  dataset.py                  GPT dataset preparation
  nlp_preprocessing.py        NLP preprocessing
  tokenizer_utils.py          Tokenization edge cases

train.py        GPT training loop
generate.py     Text generation

foundations/    Neural network primitives built from scratch
  neuron.py, backprop.py, mlp.py, activations.py, loss.py,
  training_loop.py, dead_relu_detector.py, ...
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py
python generate.py
```

## Limitations
- Not trained on large-scale datasets
- Not optimized for production use
- No distributed training support
- Simplified compared to industrial LLM systems

  
## Future Work
- Flash attention optimization
- GPU acceleration improvements
- Larger dataset training
- Web-based demo interface
- RLHF-style fine-tuning
