##movie_recommender
a recommender system using collaborative filtering. It uses matrix factorization to learn latent features for users and movies from a dataset of 20 million ratings. By leveraging TensorFlow, the system efficiently handles large datasets and provides accurate movie recommendations based on user preferences.


##Dataset:
1-Download the MovieLens dataset It uses matrix factorization to learn latent features for users and movies from a dataset of 20 million
2-Place the ratings.csv file in the data/raw/ directory.



##Usage :
To train the recommender system ,Adjust hyperparameters like embedding dimensions (K), batch size, and learning rate in the script as needed



##Evaluation:
This script calculates metrics such as Mean Squared Error (MSE) on a test set to assess model performance

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



