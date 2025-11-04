# Importing necessary dependecies
from fastai.collab import *
from fastai.tabular.all import *
from fastai.learner import load_learner
import pickle

# Loading the model
learn_inf = load_learner(r"C:\Users\Dan\Desktop\Movielens\model\PMF_collab_model_1.pkl", pickle_module=pickle)

# Reading relevant files
ratings = pd.read_csv(r"C:\Users\Dan\Desktop\Movielens\data\ratings.csv", sep=",")
movies = pd.read_csv(r"C:\Users\Dan\Desktop\Movielens\data\movies.csv", sep=",")
ratings = ratings.merge(movies)

# Turning csv into dataframe
ratings = pd.DataFrame(ratings)

# Creating dataloaders
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)

# Creating user input
user_input = input("Enter movie name: ")

# Finding movies with similar embedding distances
movie_name = f"{user_input}"
movie_factors = learn_inf.model.movie_factors.weight
idx = dls.classes['title'].o2i[movie_name]
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
idx = distances.argsort(descending=True)[1:6]
print(f"Users who Liked {movie_name} also liked:")
for t in dls.classes['title'][idx]:
    print(f"\n{t}")