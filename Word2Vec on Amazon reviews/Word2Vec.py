import gzip
import keras
import gensim
import numpy as np 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image
from keras.callbacks import EarlyStopping
from collections import defaultdict
from sklearn.preprocessing import scale
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

tknzr = TweetTokenizer()
reviewRating = []
reviewText = []
for l in readGz("/home/ayush/assignment1/train.json.gz"):
	text = tknzr.tokenize(l['reviewText'].lower())
	rating = l['rating']
	if (len(text) > 10) and (rating != 3):
		reviewText.append(text)
		reviewRating.append(int(rating > 3))

vector_size = 200
model = gensim.models.Word2Vec(iter=1)
model.build_vocab(reviewText)
model = gensim.models.Word2Vec(reviewText, window = 5, min_count = 40, size=vector_size)

vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform(reviewText)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

def buildWordVector(tokens, vector_size):
    vec = np.zeros(vector_size).reshape((1, vector_size))
    count = 0.
    for word in tokens:
        try:
            vec += model[word].reshape((1, vector_size)) * tfidf[word]
            count += 1.
        except KeyError: #Word was not present in training corpus
            continue
    if count != 0:
        vec /= count
    return vec


x_train, x_test, y_train, y_test = train_test_split(reviewText,np.array(reviewRating), test_size=0.25)
train_vecs_w2v = np.concatenate([buildWordVector(text, vector_size) for text in x_train])
train_vecs_w2v = scale(train_vecs_w2v)
test_vecs_w2v = np.concatenate([buildWordVector(text, vector_size) for text in x_test])
test_vecs_w2v = scale(test_vecs_w2v)


model_nn = Sequential()
model_nn.add(Dense(32, activation='relu', input_dim=vector_size))
model_nn.add(Dense(1, activation='sigmoid'))
model_nn.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model_nn.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)

score = model_nn.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print "Test Accuracy: ", score[1]*100
 

