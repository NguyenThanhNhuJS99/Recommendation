import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

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
    def Standardize(self, bookratings): #chuẩn hóa điểm đánh giá
        self.standardize = {}
        self.RatedBook(books,bookratings)
        for key,value in self.rated.items():
            for key1,value1 in self.books.items():
                if(key == key1):
                    meanrate = value1.MeanRating(bookratings)
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
    def MeanRating(self, bookratings): #tính điểm đánh giá trung bình cho sách
        self.ratings = []
        for key,value in bookratings.items():
            if(value.bookIsbn == self.isbn):
                self.ratings.append(value.rating)
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

class CollaborativeFiltering:
    corrMatrix = pd.DataFrame()
    def __init__(self, df_std):
        self.df_std = df_std
    def Calculate(self):
        sparse_df = sparse.csr_matrix(self.df_std.values)
        self.corrMatrix = pd.DataFrame(cosine_similarity(sparse_df),index=self.df_std.columns,columns=self.df_std.columns)
        print(self.corrMatrix)
        print('------------------------------------------------')
        self.corrMatrix = self.df_std.corr(method='pearson')
        print(self.corrMatrix.head(6))
        print('-------')
    def Get_Similar_Score(self,isbn,rating):
        book = Book(isbn,'','','','','')
        tempcategories = book.Categories(bookcategories, categories) 
        similar_score = self.corrMatrix[isbn]*(rating-2.5)
        return similar_score
    def Recommend(self, ratings):
        similar_scores = pd.DataFrame()
        for isbn,rating in ratings:
            similar_scores = similar_scores.append(amen.Get_Similar_Score(isbn,rating),ignore_index = True)
        return similar_scores.sum().sort_values(ascending=False)
users = {}
books = {}
ratings = {}
categories = {}
bookcategories = {}

dataframeUser = pd.read_json("user.json")
dataframeBook = pd.read_json("book.json")
dataframeRating = pd.read_json("book-ratings.json")
dataframeCategory = pd.read_json("category.json")
dataframeBookCate = pd.read_json("book-category.json")
# for index in list(dataFrameUser.index):
#     print(dataFrameUser[index]['Location'])
dataframeUser = dataframeUser.T #Lật dataframe để đưa các key của json thành cột
dataframeBook = dataframeBook.T
dataframeRating = dataframeRating.T
dataframeCategory = dataframeCategory.T
dataframeBookCate = dataframeBookCate.T

for key,value in dataframeUser.items():
    users[str(value['User-ID'])] = User(value['User-ID'],value['Location'],value['Age'])

for key,value in dataframeBook.items():
    books[str(value['ISBN'])] = Book(value['ISBN'], value['Book-Title'], value['Book-Author'], value['Year-Of-Publication'], value['Publisher'], value['Image-URL-L'])

for key,value in dataframeRating.items():
    ratings[str(value['User-ID']) + ' ' + str(value['ISBN'])] = BookRating(dataframeRating[key]['User-ID'],dataframeRating[key]['ISBN'],dataframeRating[key]['Book-Rating'])

for key,value in dataframeCategory.items():
    categories[str(value['Category-ID'])] = Category(dataframeCategory[key]['Category-ID'],dataframeCategory[key]['Category-Name'])

for key,value in dataframeBookCate.items():
    bookcategories[str(value['ISBN']) + ' ' + str(value['Category-ID'])] = BookCategory(dataframeBookCate[key]['ISBN'],dataframeBookCate[key]['Category-ID'])


dicts = []
for key,value in users.items():
    value.RatedBook(books, ratings)
    dicts.append(value.Standardize(ratings))
    
super_dict = {}
for d in dicts:
    for k, v in d.items():  # d.items() in Python 3+
        super_dict.setdefault(k, []).append(v)


test = pd.DataFrame().from_dict(super_dict)

amen = CollaborativeFiltering(test)
amen.Calculate()

truyen_lover = [("0452264464",5),("0393045218",5),("0671870432",5),("0002005018",5),("0425176428",5)]

print(amen.Recommend(truyen_lover).head(8))

