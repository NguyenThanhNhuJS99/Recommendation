import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv("toy_dataset.csv",index_col=0)
ratings.fillna(0, inplace=True)
ratings
print(ratings)
print('------------------------------------------------')
def standardize(row):
    new_row = (row - row.mean())/(row.max()-row.min())
    return new_row
df_std = ratings.apply(standardize).T
print(df_std)
print('------------------------------------------------')
sparse_df = sparse.csr_matrix(df_std.values)
corrMatrix = pd.DataFrame(cosine_similarity(sparse_df),index=ratings.columns,columns=ratings.columns)
print(corrMatrix)
print('------------------------------------------------')
corrMatrix = ratings.corr(method='pearson')
print(corrMatrix.head(6))
print('------------------------------------------------')
def get_similar(movie_name,rating):
    similar_score = corrMatrix[movie_name]*(rating-2.5)
    #similar_score = similar_score.sort_values(ascending=False)
    return similar_score
action_lover = [("action1",5),("romantic2",1),("romantic3",1)]
similar_scores = pd.DataFrame()
for movie,rating in action_lover:
    similar_scores = similar_scores.append(get_similar(movie,rating),ignore_index = True)
print(similar_scores.head(10))
print('------------------------------------------------')
print(similar_scores.sum().sort_values(ascending=False))
