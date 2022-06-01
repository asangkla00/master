#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


import re
import pyspark
from pyspark import SparkConf, SparkContext


# In[3]:


conf  = SparkConf().setMaster("local[*]").setAppName("1-length") 
sc = SparkContext.getOrCreate(conf=conf)

# Reads in the csv file
rdd = sc.textFile("Amazon_Comments.csv")
data = rdd.map(lambda x: x.split("^"))

# Turn to (Key, Value) where it is (Rating, Review) where review is all lower cases 
cleaned_data = data.map(lambda x: (x[6], re.sub('\W+',' ', x[5]).strip().lower().split()))
#cleaned_data.collect()


# In[6]:


# The total reviews of each rating
count = cleaned_data.groupByKey().mapValues(lambda x: len(x))
#count.collect()

# The total reviewed words of each rating
word_count = cleaned_data.mapValues(lambda x: len(x)).reduceByKey(lambda x,y: x + y)
#word_count.collect()


# In[7]:


# Join the total reviews and words of each rating
counts = count.leftOuterJoin(word_count)
#counts.collect()

# Calculate the average number of words in each rating
average = counts.mapValues(lambda x: (x[1]/x[0])).sortByKey(ascending=True)
#average.collect()


# In[8]:


print("1 star rating: average length of comments {}".format(average.values().collect()[0]))
print("2 star rating: average length of comments {}".format(average.values().collect()[1]))
print("3 star rating: average length of comments {}".format(average.values().collect()[2]))
print("4 star rating: average length of comments {}".format(average.values().collect()[3]))
print("5 star rating: average length of comments {}".format(average.values().collect()[4]))

