# README

This repository contains a implementation of our "Hybrid Language Model and Diffusion based Neural ODE for Progressive Diagnosis Prediction in Healthcare Data" .

## Environment Setup

1. Pytorch 1.12.1
2. Python 3.7.15

### Run

The implementation of language model-based embedding initialization for EHR data (```emb.py```) and model (```ProHealth.py```); 

## Example to run the codes

### step 1:fine-tune language model and generate embeddings

```python
python emb.py
```

### step 2:train model

```python
python ProHealth.py
```

