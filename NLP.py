#Natural Language Processing

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
#importando o dataset
dataset = pd.read_csv('data.tsv', delimiter = '\t', quoting = 3)

#limpando os textos
import re   #biblioteca para remoção de letras e caracteres
import nltk #biblioteca para remoção de stopwords (palavras sem peso)
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] 
for i in range(0, 1000): 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #mantém todas as letras de a-z, os espaços. Retira todo o resto
    review = review.lower() #transforma todas as letras em minúsculas
    review = review.split() #separa o texto em valores de uma lista
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] #subtrai do texto as palavras "Stopwords"
                                                                                               # e pega a raiz das palavras (Loved = Love)     
    review = ' '.join(review) #transforma o array numa string de novo.
    corpus.append(review) #implementa na variável os resultados.

#Criando a Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer #Converte uma coleção de texto em uma matris 
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() #aplicou como array em Corpus
Y = dataset.iloc[:, 1].values          # Criou uma variável apenas com comentários (1,0) positivos e negativos
"""
Criando o modelo de aprendizagem de máquina usando classificação de Naive Bases
"""
#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#Fitting classifier to the Training set
# Create your classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test) 

#Making the Confuion Matrix  Essa Matriz mostra a quantidade de predições certas e corretas. 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
