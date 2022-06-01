#!/usr/bin/env python
# coding: utf-8

# # Final project 
from pyspark.context import SparkContext, SparkConf
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import LogisticRegression,LinearSVC
from pyspark.ml.feature import HashingTF,  IDF, RegexTokenizer,  StopWordsRemover,NGram,CountVectorizer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf,explode, split
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as f
from pyspark.sql.functions import when
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

schema = StructType([
StructField("business_id", StringType()),
StructField("name", StringType()),
StructField("neighborhood", StringType()),
StructField("address", StringType()),
StructField("city", StringType()),
StructField("state", StringType()),
StructField("postal_code", StringType()),
StructField("latitude", StringType()),
StructField("longitude", StringType()),
StructField("stars", StringType()),
StructField("review_count", StringType()),
StructField("is_open", DoubleType()),
StructField("categories", StringType())
])

#1. Start Spark Context with Spark Session 
# local
#sc = SparkContext("local[*]")
# cluster
sc = SparkContext("yarn")
spark = SparkSession(sc)

#2. Read the csv file
# Google cloud
df1 = spark.read.format("csv").option("header", "true").option("multiline", "true").load("gs://msa8050/yelp_review.csv")
df2 = spark.read.format("csv").schema(schema).option("header", "true").option("multiline", "true").option("mode", "DROPMALFORMED").load("gs://msa8050/yelp_business.csv")
# Local
#df1 = spark.read.format("csv").option("header", "true").option("multiline", "true").load("yelp_review.csv")
#df2 = spark.read.format("csv").schema(schema).option("header", "true").option("multiline", "true").option("mode", "DROPMALFORMED").load("yelp_business.csv")
selected = df1.select("stars","text")
cleaned = selected.dropna()


# ## Task 1: Top rating by users

filtered = selected.filter(selected.stars.isin('1','2','3','4','5'))
countrating = filtered.groupBy("stars").count().orderBy("stars")
countrating.show()
#Plot
#df_pandas = countrating.toPandas()
#plt.bar(df_pandas['stars'], df_pandas['count'], color =['C0', 'C1', 'C2', 'C3', 'C4', 'C5'],width = 0.5)
#plt.xlabel('Stars')
#plt.ylabel('Number of reviews')
#plt.title('Top rating by users')
#plt.show()

# ## Task 2 : The average length of words for each rating

data = cleaned.withColumn("rating", df1["stars"].cast("double").cast("string"))
countTokens = udf(lambda words: len(words), IntegerType())
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+")
regexTokenized = regexTokenizer.transform(data)
word = regexTokenized.select('rating','words')
numword = word.withColumn("numwords", countTokens(col("words")))
nword = numword.filter(numword.rating.isin('1.0','2.0','3.0','4.0','5.0'))
avgword = nword.groupBy("rating").agg(f.avg("numwords")).orderBy("rating")
avgword.show()
#df_pandas = avgword.toPandas()
#plt.bar(df_pandas['rating'], df_pandas['avg(numwords)'], color =['C0', 'C1', 'C2', 'C3', 'C4', 'C5'],width = 0.5)
#plt.ylabel('Average number of words')
#plt.xlabel('Rating')
#plt.title('The average length of reviews each rating')
#plt.show()


# ## Task 3: Top 10 most used words
cleaned_text = selected.dropna()
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words_token", pattern="\\W+")
regexTokenized = regexTokenizer.transform(cleaned_text).select('stars','words_token')
remover = StopWordsRemover(inputCol='words_token', outputCol='words_clean')
data_clean = remover.transform(regexTokenized)
result = data_clean.withColumn('word', f.explode(f.col('words_clean')))
select = result.select('stars', 'word')
filtered5 = select.filter(select.stars.isin('5'))
groupword5 = filtered5.groupBy('word').count().sort('count', ascending=False)
filtered4 = select.filter(select.stars.isin('4'))
groupword4 = filtered4.groupBy('word').count().sort('count', ascending=False)
filtered3 = select.filter(select.stars.isin('3'))
groupword3 = filtered3.groupBy('word').count().sort('count', ascending=False)
filtered2 = select.filter(select.stars.isin('2'))
groupword2 = filtered2.groupBy('word').count().sort('count', ascending=False)
filtered1 = select.filter(select.stars.isin('1'))
groupword1 = filtered1.groupBy('word').count().sort('count', ascending=False)
print("5 star rating: top 10 common words:\n")
groupword5.show(10)
print("4 star rating: top 10 common words:\n")
groupword4.show(10)
print("3 star rating: top 10 common words:\n")
groupword3.show(10)
print("2 star rating: top 10 common words:\n")
groupword2.show(10)
print("1 star rating: top 10 common words:\n")
groupword1.show(10)
#df_pandas1 = groupword1.limit(10).toPandas()
#df_pandas2 = groupword2.limit(10).toPandas()
#df_pandas3 = groupword3.limit(10).toPandas()
#df_pandas4 = groupword4.limit(10).toPandas()
#df_pandas5 = groupword5.limit(10).toPandas()
#df_pandas1.sort_values('count', ascending= True).plot(x = "word", y = "count", kind  = "barh", color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], legend = False)
#plt.ylabel('Words')
#plt.xlabel('Frequency')
#plt.title('Top 10 common words used for 1-star reviews')
#plt.show()

#df_pandas2.sort_values('count', ascending= True).plot(x = "word", y = "count", kind  = "barh", color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], legend = False)
#plt.ylabel('Words')
#plt.xlabel('Frequency')
#plt.title('Top 10 common words used for 2-star reviews')
#plt.show()

#df_pandas3.sort_values('count', ascending= True).plot(x = "word", y = "count", kind  = "barh", color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], legend = False)
#plt.ylabel('Words')
#plt.xlabel('Frequency')
#plt.title('Top 10 common words used for 3-star reviews')
#plt.show()

#df_pandas4.sort_values('count', ascending= True).plot(x = "word", y = "count", kind  = "barh", color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], legend = False)
#plt.ylabel('Words')
#plt.xlabel('Frequency')
#plt.title('Top 10 common words used for 4-star reviews')
#plt.show()

#df_pandas5.sort_values('count', ascending= True).plot(x = "word", y = "count", kind  = "barh", color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], legend = False)
#plt.ylabel('Words')
#plt.xlabel('Frequency')
#plt.title('Top 10 common words used for 5-star reviews')
#plt.show()


# ## Task 4: Top 10 Categories
cleaned = df2.dropna()
category = cleaned.select('categories')
indcategory = category.select(explode(split('categories', ';')).alias('category'))
grouped_category = indcategory.groupby('category').count()
top_category = grouped_category.sort('count',ascending=False)
top_category.show(10)
#df_pd = top_category.limit(10).toPandas()
#df_pd.sort_values('count', ascending= True).plot(x = "category", y = "count", kind  = "barh", color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], legend = False)
#plt.xlabel('Count')
#plt.title('Top 10 category')
#plt.show()


# ## Task 5: Top cities having the most reviews
loc = df2.select('business_id','city')
review = df1.select('business_id')
merge_city = loc.join(review,'business_id','inner')
grouped_review = merge_city.groupby('city').count()
reviewed_city = grouped_review.groupby('city').sum().sort('sum(count)',ascending=False)
reviewed_city.show(10)
#df_pd = reviewed_city.limit(10).toPandas()
#df_pd.sort_values('sum(count)', ascending= True).plot(x = "city", y = "sum(count)", kind  = "barh", color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'], legend = False)
#plt.title('Top 10 cities having the most reviews')
#plt.show()


# ## Task 6: Classify positive and negative reviews
newdf = df1.withColumn("rating", df1["stars"].cast("double"))
df = newdf.select("rating", "text")
cleaned_df = df.dropna()
binary = cleaned_df.withColumn("label", when(df['rating'] > 3.0, 1).otherwise(0))
#binary.groupBy('label').count().show()
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W+")
remover = StopWordsRemover(inputCol= "words", outputCol="filtered")
cv = CountVectorizer(inputCol="filtered", outputCol="rawfeatures")
idf = IDF(inputCol="rawfeatures", outputCol="features")
svm = LinearSVC(maxIter=30)
pipeline = Pipeline(stages=[tokenizer, remover, cv, idf,svm])
(trainingData, testData) = binary.randomSplit([0.7, 0.3])


# model fitting and prediction
model = pipeline.fit(trainingData)
prediction = model.transform(testData)
#result = prediction.select("label","words", "prediction").show()

# model evaluation
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = evaluator.evaluate(prediction) 
print("F1 score of this model is %.2f" % f1)

getword = prediction.select("filtered","prediction")
wordsplit = getword.withColumn('word', f.explode(f.col('filtered')))
result = wordsplit.select("prediction", "word")
filtered_positive = result.filter(result.prediction.isin('1.0'))
groupword1 = filtered_positive.groupBy('word').count().sort('count', ascending=False)
filtered_negative = result.filter(result.prediction.isin('0.0'))
groupword2 = filtered_negative.groupBy('word').count().sort('count', ascending=False)
print("Top 10 positive words:\n")
groupword1.show(10)
print("Top 10 negative words:\n")
groupword2.show(10)

#df_pandas1 = groupword1.limit(50).toPandas()
#df_pandas2 = groupword2.limit(50).toPandas()
#text = " ".join(word for word in df_pandas1.word.astype(str))
#wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)
#plt.axis("off")
#plt.figure( figsize=(20,10))
#plt.tight_layout(pad=0)
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.show()

#text = " ".join(word for word in df_pandas2.word.astype(str))
#wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)
#plt.axis("off")
#plt.figure( figsize=(20,10))
#plt.tight_layout(pad=0)
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.show()

