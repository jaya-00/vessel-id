# Vessel Origins

**Vessel Origins** is an experimental computer vision project that predicts the likely cultural origin of an ancient vessel from an uploaded photograph.

The project combines museum collection data, machine learning, and an interactive web interface to explore how computer vision can be applied to cultural heritage objects.

## Current MVP

The current model classifies vessels into four cultural categories:

- East Asian
- Egyptian
- Greek
- Roman

The classifier is currently trained using vessel images and object information collected from The Metropolitan Museum of Art's online collection.

This is an early-stage MVP and is intended as an experimental classification tool rather than an authoritative method of cultural attribution.

## How It Works

1. A user uploads a photograph of an ancient vessel.
2. The image is sent to a FastAPI backend.
3. A PyTorch image classification model processes the image.
4. The model returns probabilities for each supported cultural category.
5. The website displays the most likely cultural origin and the model's confidence scores.

## Model

The current model uses a ResNet-based image classification architecture built with PyTorch.

## Current Categories:

The current MVP can identify vessels across four cultural categories:

- **East Asian**
- **Egyptian**
- **Greek**
- **Roman**

The dataset is currently limited and imbalanced, so predictions should be interpreted as experimental results rather than definitive identifications.

Future versions will expand the training dataset and the number of supported cultural categories.

## Tech Stack:
Machine Learning
- Python
- PyTorch
- Torchvision
- ResNet

## Backend
- FastAPI
- Uvicorn

## Frontend
- HTML
- CSS
- JavaScript

## Data
- The Metropolitan Museum of Art Open Access collection

## Project Structure
```text
ANCIENT_VESSEL/
├── assets/
├── back.ipynb
├── best_pottery_model_cpu.pt
├── full_df_backup.csv
├── index.html
├── requirements.txt
├── server.py
└── README.md
```

## Goals
The long-term goal of Vessel Origins is to explore how computer vision can assist with the discovery and preliminary classification of historical objects while maintaining transparency about model uncertainty and dataset limitations.

## Future development may include:
- Additional cultural categories
- A larger and more balanced training dataset
- Improved model evaluation
- Object metadata integration
- More detailed prediction explanations
- Expanded museum collection sources

## Disclaimer
Vessel Origins is an experimental educational project. Model predictions are probabilistic and should not be treated as professional archaeological, historical, or curatorial attributions.

Museum collection images and associated metadata remain subject to the terms and rights information provided by their respective institutions.

## Creator
Created by Jael Ultimo.
Built as an independent exploration of computer vision, human-computer interaction, cultural heritage, and museum collections.