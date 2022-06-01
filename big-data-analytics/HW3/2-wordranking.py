#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


import re
import pyspark
from pyspark import SparkConf
from pyspark import SparkContext


# In[3]:


conf  = SparkConf().setMaster("local[*]").setAppName("2-wordranking") 
sc = SparkContext.getOrCreate(conf)


# In[4]:


# Reads in the csv file
rdd = sc.textFile("Amazon_Comments.csv")
data = rdd.map(lambda x: x.split("^"))
#data.collect()


# In[5]:


cleaned_data = data.map(lambda x: (x[6], re.sub('\W+',' ', x[5]).strip().lower()))
#cleaned_data.collect()


# In[6]:


import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
#print(stopwords)


# In[7]:


words = cleaned_data.flatMapValues(lambda x: x.split())
words_cleaned = words.filter(lambda x: x[1] not in stopwords)
#words_cleaned.collect()


# In[8]:


one_rating_words = words_cleaned.filter(lambda x: x[0]=="1.00")
#one_rating_words.collect()
two_rating_words = words_cleaned.filter(lambda x: x[0]=="2.00")
#two_rating_words.collect()
three_rating_words = words_cleaned.filter(lambda x: x[0]=="3.00")
#three_rating_words.collect()
four_rating_words = words_cleaned.filter(lambda x: x[0]=="4.00")
#four_rating_words.collect()
five_rating_words = words_cleaned.filter(lambda x: x[0]=="5.00")
#five_rating_words.collect()


# In[9]:


one_words_list = one_rating_words.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
one_common = one_words_list.takeOrdered(10, lambda x: -x[1])
#one_common


# In[10]:


two_words_list = two_rating_words.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
two_common = two_words_list.takeOrdered(10, lambda x: -x[1])
#two_common


# In[11]:


three_words_list = three_rating_words.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
three_common = three_words_list.takeOrdered(10, lambda x: -x[1])
#three_common


# In[12]:


four_words_list = four_rating_words.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
four_common = four_words_list.takeOrdered(10, lambda x: -x[1])
#four_common


# In[13]:


five_words_list = five_rating_words.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
five_common = five_words_list.takeOrdered(10, lambda x: -x[1])
#five_common


# In[14]:


one = [x[0][1] for x in one_common]
two = [x[0][1] for x in two_common]
three = [x[0][1] for x in three_common]
four = [x[0][1] for x in four_common]
five = [x[0][1] for x in five_common]
print("top 10 common words")
print("1 star rating:",one)
print("2 star rating:",two)
print("3 star rating:",three)
print("4 star rating:",four)
print("5 star rating:",five)

