# Movie Recommender System

## Overview
**movie_recommender** is a recommender system using collaborative filtering. It utilizes matrix factorization to learn latent features for users and movies from a dataset of 20 million ratings. Leveraging TensorFlow, the system efficiently handles large datasets and provides accurate movie recommendations based on user preferences.

## Dataset
1. Download the MovieLens dataset.
2. Place the `ratings.csv` file in the `data/raw/` directory.

## Usage
To train the recommender system, adjust hyperparameters like embedding dimensions (`K`), batch size, and learning rate in the script as needed.


## Technology Stack

- **Framework**: This project utilizes PyTorch for building and training the recommender system.
- **Training**: The training process is currently configured to run on CPU. For optimal performance, consider running on GPU if available.


## Evaluation
The evaluation script calculates metrics such as Mean Squared Error (MSE) on a test set to assess model performance.

## Project Structuree

```
movie-recommender/
├── data/
│   ├── raw/
│   │   └── ratings.csv # Raw dataset
│   └── processed/
│       └── edit_ratings.csv # Processed dataset after preprocessing
├── module/
│   ├── model.py # Recommender model definition
│   ├── setup.py # Dataset setup and DataLoader
│   ├── process.py # Data preprocessing functions
│   ├── engine.py # Training and evaluation engine
│   ├── train.py # Script for training the model
```

## traing plot:

![Plot](movie-recommender/img/plot.png)




