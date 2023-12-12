import numpy as np
import pandas as pd
import ast
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import streamlit as st

st.title("Restaurant Review Classification")

data=pd.read_csv("C:\\Users\\manoj\\OneDrive\\Desktop\\reviews\\Restaurant_Reviews.csv")

corpus=[]
for i in range(0,1000):
  review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=data['Review'][i])
  review= review.lower()
  review_words=review.split()
  review_words = [word for word in review_words if not word in set(stopwords.words('english'))]
  ps=PorterStemmer()
  review=[ps.stem(word) for word in review_words]
  review = ' '.join(review)
  corpus.append(review)

print(corpus[:10])


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
nv=GaussianNB()
nv.fit(x_train,y_train)

def predict_review(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ',string=sample_review)
  sample_review = sample_review.lower()
  sample_review_words=sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  ps=PorterStemmer()
  final_review=[ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)
  temp = cv.transform([final_review]).toarray()
  return nv.predict(temp)

try:
  text = st.text_input("Enter")
  if predict_review(text):
    print("The review given is a POSITIVE REVIEW")
  else:
    print("The review given is a NEGATIVE REVIEW")
except:
  st.markdown("Please enter a review")