import pandas as pd
from scipy import sparse
from math import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
class User:
    rated = {}
    books = {}
    standardize = {}
    def __init__(self,id, location, age):
        self.id = id
        self.location = location
        self.age = age
    def RatedBook(self,books,bookratings): #Tạo dataframe điểm đánh giá cho sách
        self.rated = {}
        self.books = {}
        for key,value in books.items():
            self.books[str(value.isbn)] = value
            self.rated[str(value.isbn)] = BookRating(self.id,value.isbn,0)
        for key,value in bookratings.items():
            if(value.userId == self.id):
                self.rated[str(value.bookIsbn)] = value
        return self.rated
    def Standardize(self, bookratings,users): #chuẩn hóa điểm đánh giá
        self.standardize = {}
        self.RatedBook(self.books,bookratings)
        for key,value in self.rated.items():
            for key1,value1 in self.books.items():
                if(key == key1):
                    meanrate = value1.MeanRating(bookratings,users)
                    self.standardize[str(key)] = (value.rating - meanrate) / (max(value1.ratings) - min(value1.ratings))                   
        return self.standardize


class Book: 
    tempcategories = []
    categories = []
    ratings = []
    meanrate = 0
    standardize = 0
    def __init__(self,isbn, bookTitle, bookAuthor, yearOfPublication, publisher, image):
        self.isbn = isbn
        self.bookTitle = bookTitle
        self.bookAuthor = bookAuthor
        self.yearOfPublication = yearOfPublication
        self.publisher = publisher
        self.image = image
    def Categories(self, bookcategories, categories): #Lấy ra thể loại sách
        self.categories = []
        self.tempcategories = []
        for key,value in bookcategories.items():
            if(self.isbn == value.bookIsbn):
                self.tempcategories.append(value.categoryId)
        for key,value in categories.items():
            for i in range(len(self.tempcategories)):
                if(value.id == self.tempcategories[i]):
                    self.categories.append(value.name)
        return self.categories
    def MeanRating(self, bookratings,users): #tính điểm đánh giá trung bình cho sách
        self.ratings = []
        self.meanrate = 0
        count = 0
        for key,value in bookratings.items():
            if(value.bookIsbn == self.isbn):
                count+=1
                self.ratings.append(value.rating)
        for i in range(len(users) - count):
            self.ratings.append(0);
        self.meanrate = sum(self.ratings) / len(self.ratings)
        return self.meanrate
class Category: 
    meanrate = 0
    books = []
    rated = []
    standardize = []
    def __init__(self, id, name):
        self.id = id
        self.name = name
    def Books(self,books, bookcategories):
        self.books = []
        for key,value in books.items():
            for key1,value1 in bookcategories.items():
                if(value1.categoryId == self.id and value.isbn == value1.bookIsbn):
                    self.books.append(value)
        return self.books
    
class BookCategory:
    def __init__(self, bookIsbn, categoryId):
        self.bookIsbn = bookIsbn
        self.categoryId = categoryId
class BookRating:
    def __init__(self, userId, bookIsbn, rating):
        self.userId = userId
        self.bookIsbn = bookIsbn
        self.rating = rating
class Cosine(object):
    def __init__(self,row):
        self.row = row
        return
    def Standardize(self):
        new_row = (self.row - self.row.mean()) / (self.row.max() - self.row.min())
        return new_row
class CollaborativeFiltering:
    corrMatrix = pd.DataFrame()
    similar_scores = pd.DataFrame()
    def __init__(self, df_ustd):
        self.df_std = df_ustd.apply(lambda x: Cosine(x).Standardize()).T #Chuẩn hóa giá trị của từng dòng
        self.df = df_ustd
    def CosineCalculate(self):
        sparse_df = sparse.csr_matrix(self.df_std.values)
        self.corrMatrix = pd.DataFrame(cosine_similarity(sparse_df),index=self.df_std.T.columns,columns=self.df_std.T.columns)
        print('------------------------------------------------')
        print(self.corrMatrix)
        print('-------')
    def PearsonCalculate(self): #Su dng pearson correclation similartiy
        self.corrMatrix = self.df.corr(method='pearson')
    def SpearmanCalculate(self): #Su dung spearman correclation similarity
        self.corrMatrix = self.df.corr(method='spearman')
    def KendallCalculate(self): #Su dung kendall correclation similarity
        self.corrMatrix = self.df.corr(method='kendall')
    def Get_Similar_Score(self, isbn,rating):
        similar_score = self.corrMatrix[isbn]*(rating - 2.5)
        return similar_score
    def Recommend(self, ratings):
        self.similar_scores = pd.DataFrame()
        for isbn,rating in ratings:
            self.similar_scores = self.similar_scores.append(self.Get_Similar_Score(isbn,rating),ignore_index = True)
        return self.similar_scores.sum().sort_values(ascending=False).head()
    def MAE(self):
        temp = self.similar_scores.T
        sparse_df = sparse.csr_matrix(temp.values)
        temp2 = pd.DataFrame(cosine_similarity(sparse_df),index=temp.T.columns,columns=temp.T.columns)
        return mean_absolute_error(self.corrMatrix, temp2)
    def RMSE(self):
        temp = self.similar_scores.T
        sparse_df = sparse.csr_matrix(temp.values)
        temp2 = pd.DataFrame(cosine_similarity(sparse_df),index=temp.T.columns,columns=temp.T.columns)
        return mean_squared_error(self.corrMatrix, temp2, squared=False)

dataframeUser = pd.read_json("user.json")
dataframeBook = pd.read_json("book.json")
dataframeRating = pd.read_json("book-ratings.json")
dataframeCategory = pd.read_json("category.json")
dataframeBookCate = pd.read_json("book-category.json")

dataframeRating.fillna(0, inplace=True)
user_ratings = pd.merge(dataframeUser, dataframeRating, on="User-ID")
user_ratings_data = pd.merge(user_ratings,dataframeBook, on="ISBN")

test = pd.DataFrame(dataframeRating.groupby('ISBN')['Book-Rating'].mean());
test['rating_count'] = pd.DataFrame(dataframeRating.groupby('ISBN')['Book-Rating'].count());
book_data = pd.merge(test, dataframeBook,on='ISBN')


user_book_rating = user_ratings_data.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating')
user_book_rating.fillna(0, inplace=True);
# size_Train = int(len(user_book_rating)*0.8)
# train_model = user_book_rating[:size_Train]
# test_model = user_book_rating[size_Train:]
# print(test_model)


# ,("0061076031",5),("1567407781",5),("1881320189",5)
user = [("0002005018",5),("0061076031",5),("1567407781",5),("1881320189",5)]

CF = CollaborativeFiltering(user_book_rating);
CF.CosineCalculate();
print(user)
print('Dự đoán với độ tương đồng cosine')
print('---------------------------')
print(CF.Recommend(user));
print("RMSE: " + str(CF.RMSE()))
print("MAE: " + str(CF.MAE()))
print('------------------------------')
CF.PearsonCalculate();
print('Dự đoán với độ tương đồng PCC(Pearson Correclation Coefficient)')
print('---------------------')
print(CF.Recommend(user));
print("RMSE: " + str(CF.RMSE()))
print("MAE: " + str(CF.MAE()))
CF.SpearmanCalculate();
print('Dự đoán với độ tương đồng SRC(Spearman Rank Coefficient)')
print('---------------------')
print(CF.Recommend(user));
print("RMSE: " + str(CF.RMSE()))
print("MAE: " + str(CF.MAE()))
CF.KendallCalculate();
print('Dự đoán với độ tương đồng KCC(Kendall Tau correlation coefficient)')
print('---------------------')
print(CF.Recommend(user));
print("RMSE: " + str(CF.RMSE()))
print("MAE: " + str(CF.MAE()))