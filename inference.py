# Importing necessary dependecies
from fastai.collab import *
from fastai.tabular.all import *
from fastai.learner import load_learner

# Loading the model
model_path = Path("/home/duncan/fastai-project/venv/model/PMF_collab_model_1.pkl")
learn_inf = load_learner(model_path)

# Reading relevant files
ratings = pd.read_csv("/home/duncan/fastai-project/venv/data/ratings.csv", sep=",")
movies = pd.read_csv("/home/duncan/fastai-project/venv/data/movies.csv", sep=",")
ratings = ratings.merge(movies)

# Turning csv into dataframe
ratings = pd.DataFrame(ratings)

# Creating dataloaders
dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)

# Creating user input

print("\n=====================================================\n")
user_input = input("\nEnter movie name: ")

# Finding movies with similar embedding distances
movie_name = f"{user_input}"
movie_factors = learn_inf.model.movie_factors.weight
idx = dls.classes['title'].o2i[movie_name]
distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
idx = distances.argsort(descending=True)[1:6]
print(f"\nUsers who Liked {movie_name} also liked:")
for t in dls.classes['title'][idx]:
    print(f"\n{t}")
print("\n=====================================================\n")