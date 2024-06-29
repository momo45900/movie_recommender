import pandas as pd
from sklearn.utils import shuffle

def dataframe_process_create(raw_dir):
    # Load the data
    df = pd.read_csv(raw_dir)

    # Make the user IDs go from 0...N-1
    df.userId = df.userId - 1

    # Create a mapping for movie IDs
    unique_movie_ids = sorted(set(df.movieId.values))
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}

    # Map the movie IDs to indices
    df['movie_idx'] = df['movieId'].map(movie2idx)

    # Drop the 'timestamp' column
    df = df.drop(columns=['timestamp'])

    # Save the processed DataFrame
    df.to_csv('data/processed/edit_ratings.csv', index=False)

    print("The processed data has been created.")

def train_test_spliter(df):
    num_users = df.userId.max() + 1  # number of users
    num_movies = df.movie_idx.max() + 1  # number of movies
    mu = df.rating.mean()

    # Split into train and test
    df = shuffle(df, random_state=42)  # Add a random_state for reproducibility
    cutoff = int(0.8 * len(df))
    df_train = df.iloc[:cutoff]
    df_test = df.iloc[cutoff:]

    return df_train, df_test, num_users, num_movies, mu

# Example usage:
# dataframe_process_create('data/raw/ratings.csv')
# df_train, df_test, num_users, num_movies, mu = train_test_spliter(pd.read_csv('data/processed/edit_ratings.csv'))
