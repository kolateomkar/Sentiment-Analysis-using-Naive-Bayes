#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sklearn
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB , MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("dataset-file-path")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.sentiment.value_counts()


# In[7]:


df.sentiment.replace('positive', 1 , inplace = True)
df.sentiment.replace('negative', 0 , inplace = True)


# In[8]:


df.head()


# In[9]:


df.review[0]


# In[10]:


def clean_data(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)


# In[11]:


text_with_tags = "<p>This is an example<b>with</b> HTML tags.</p>"
cleaned_text = clean_data(text_with_tags)
print(cleaned_text)


# In[12]:


df.review = df.review.apply(clean_data)
df.review[0]


# In[13]:


def clean_special_characters(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem


# In[14]:


text_with_special_characters = "Hello@hh world! How are you?"
cleaned_text = clean_special_characters (text_with_special_characters)
print(cleaned_text)


# In[15]:


df.review = df.review.apply(clean_special_characters)
df.review[0]


# In[16]:


def convert_to_lowercase(text):
    return text.lower()


# In[17]:


sentence = "This is a SENTENCE with UPPERCASE letters"
output_sentence = convert_to_lowercase(sentence)
print(output_sentence)


# In[18]:


df.review = df.review.apply(convert_to_lowercase)
df.review[0]


# In[19]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [word for word in words if word.lower() not in stop_words]


# In[20]:


sentence = "They are right, as this is exactly what happend with me."
output_sentence = remove_stopwords(sentence)
print (output_sentence)


# In[21]:


get_ipython().run_cell_magic('time', '', 'df.review = df.review.apply(remove_stopwords)')


# In[22]:


def stemmer (text):
    stemmer_object = SnowballStemmer('english')
    return " ".join([stemmer_object.stem(w) for w in text])


# In[23]:


text = "The cats are running"
stemmed_text = stemmer(text.split())
print (stemmed_text)


# In[24]:


get_ipython().run_cell_magic('time', '', 'df.review = df.review.apply(stemmer)')


# In[25]:


x = np.array(df.iloc[:,0].values)


# In[26]:


y = np.array(df.sentiment.values) 


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[28]:


type(x_train)


# In[29]:


vectorizer = CountVectorizer(max_features = 1000)


# In[30]:


x_train_final = vectorizer.fit_transform(x_train).toarray()


# In[31]:


x_test_final = vectorizer.fit_transform(x_test).toarray()


# In[32]:


print("x_train_final :", x_train_final.shape)
print("y_train :", y_train.shape)


# In[33]:


print(x_train_final)


# In[34]:


model_v1 = GaussianNB()


# In[35]:


model_v1.fit(x_train_final, y_train)


# In[36]:


model_v2 = MultinomialNB()


# In[37]:


model_v2.fit(x_train_final, y_train)


# In[38]:


model_v3 = BernoulliNB(alpha = 1.0, fit_prior = True)
model_v3.fit(x_train_final, y_train)


# In[39]:


y_pred_v1 = model_v1.predict(x_test_final)
y_pred_v2 = model_v2.predict(x_test_final)
y_pred_v3 = model_v3.predict(x_test_final)


# In[40]:


print("Acuuracy of GaussianNB : ", accuracy_score(y_test, y_pred_v1)*100)
print("Acuuracy of MultinomialNB: ", accuracy_score(y_test, y_pred_v2)*100)
print("Acuuracy of BernoulliNB : ", accuracy_score(y_test, y_pred_v3)*100)


# In[41]:


from sklearn.metrics import roc_auc_score


# In[42]:


y_proba = model_v1.predict_proba(x_test_final)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("AUC of GaussianNB :",auc)

y_proba = model_v2.predict_proba(x_test_final)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("AUC of MultinomialNB :",auc)

y_proba = model_v3.predict_proba(x_test_final)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("AUC of BernoulliNB :",auc)


# In[43]:


with open('model_v3.pkl', 'wb') as file:
    pickle.dump(model_v3, file)
    


# In[44]:


with open('model_v3.pkl', 'rb') as file:
    final_model = pickle.load(file)


# In[45]:


review_text = " movie is very good"


# In[46]:


task1 = clean_data(review_text)
task2 = clean_special_characters(task1)
task3 = convert_to_lowercase(task2)
task4 = remove_stopwords(task3)
task5 = stemmer(task4)


# In[47]:


print (task5)


# In[48]:


type(task5)


# In[49]:


task5_array = np.array(task5)


# In[50]:


type(task5_array)


# In[51]:


final_review = vectorizer.transform(np.array([task5_array])).toarray()


# In[52]:


type(final_review)


# In[53]:


prediction = final_model.predict(final_review.reshape(1, 1000))
print (prediction)


# In[54]:


if prediction == 1:
    print ("The Text Indicates Positive Sentiment")
    
else :
    print ("The Text Indicates Negative Sentiment")


# In[55]:


final_model.predict(final_review.reshape(1, 1000))


# In[ ]:




