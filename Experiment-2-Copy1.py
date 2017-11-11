
# coding: utf-8

# <h3>Experiment 2</h3>
# <h4>Team members: Sweetin Paul, Vishaka Patel</h4>

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode
from array import array
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.ml.tuning import *
import matplotlib.pyplot as plt


#files to use for processing
file_user_details="users_libraries_test.txt"
# initialise Spark Session
spark = SparkSession.builder.appName("Experiment2_test").getOrCreate()
sc = spark.sparkContext

# Schema for reading the user_like table
userSchema = StructType([
    #  name, dataType, nullable
    StructField("user_id_string", StringType(), False),
    StructField("paper_hash", StringType(), False)  
])
user_like_hash = spark.read.csv(file_user_details,sep=';',schema=userSchema)




# In[2]:


# We shall create a UDF for inserting a unique integer id for each user_id in user_like_hash table
#  and then select the integer_id and string_id and store it as a mapping table

# the user ids are small in size so it can be collected in the driver and mapped with an int id in a dictionary
array_of_string_ids=user_like_hash.select('user_id_string').distinct().collect()
user_id_dictionary={}
#enumerate through the array of string_ids ,storing it as keys and the counter as the value
for i,value in enumerate(array_of_string_ids):
    user_id_dictionary[value.user_id_string]=i+10
#bradcast the user_id dictionary to all nodes    
broadcasted_user_idmapping=sc.broadcast(user_id_dictionary)

# Udf function to read string_id and return the corresponding integer_id from the dictionary. 

udf_for_userid= udf(lambda string_id: 
                    broadcasted_user_idmapping.value[string_id]
                    , IntegerType())

user_like_hash=user_like_hash.withColumn('id',
                                         udf_for_userid(user_like_hash.user_id_string))

#mapping table for user_id to integer id
user_id_mapping=user_like_hash.select('user_id_string','id')


# In[3]:


#separating each row of user_id and paper_id_array (u_id,[p_1,p_2,p-3])
#into separate rows..
#.. into (u_id,p_1),(u_id,p_2),(u_id,p_3)
user_likes=user_like_hash.select(user_like_hash.id.alias('user_id')
                                 ,explode( split('paper_hash',','))
                                        .alias('paper_id'))

#casting paper_ids to integer
user_likes= user_likes.select('user_id',
                              user_likes.paper_id.cast(IntegerType())
                                        .alias('paper_id'))

#Exercise 2.1: Sparsity ratio: No of empty ratings divided by total ratings
#to calculate this ratio, we need to find:
                        #1. total ratings is total possible entries in ..
                        #  ..rating matrix = no of distinct users x no of distint papers
                        #2. empty ratings= total ratings- available ratings(i.e no of rows in user_likes table)
paper_count=user_likes.select('paper_id').distinct().count()
total_ratings=user_like_hash.count()*paper_count
print("total ratings NxM",total_ratings)
print("Sparsity ratio of rating matrix is:",(total_ratings-user_likes.count())/total_ratings)

#Ex 2.1

item_popularity=user_likes.groupBy('paper_id')
                            .count()
                            .withColumnRenamed('count','papers_likecount')

print("plotting")
plt.hist(user_likes.select('user_id').rdd.map(lambda row: row.user_id).collect(), bins=28416)
plt.show()
plt.hist(user_likes.select('paper_id').rdd.map(lambda row: row.paper_id).collect(), bins=172079)
plt.show()

#ex 2.2
# Each row in the user_likes table represent a paper which has been rated 1 by that user. This our table of 1 ratings 
# To do this ,We will append a column to user_likes with default 1.

rating_1_table=user_likes.select('user_id','paper_id')
                            .withColumn('paper_ratings',lit(1))

# the ratings matrix conceptually contains rating of one user corresponding to all papers, 
# to get such a matrix , first we have to do a cross join of user_ids with paper_ids
# .We will default all values with a 0 rating

#We have to then subtract those rows which are already present in ratin_1 table and again..
#..union this result with rating_1 table

'''update 7-nov-17:For each user (u), add q unknown ratings, 
where q equals to the number of positive ratings the user (u) has. 
The papers which will receive the unknown ratings 
should be chosen randomly from the set of papers unrated by (u). '''

#So we will not do cross join instead union x no. of rows for each user into rating_1_table. 
#x is the no. of '1' ratings for each user.

table_with_all_ratings=user_likes.select('user_id').distinct()
                                .crossJoin(item_popularity.select('paper_id'))
                                .withColumn('paper_ratings',lit(0))
#rating_1_table.show() table_with_all_ratings.show() user_likes.show()
rating_matrix=table_with_all_ratings.subtract(rating_1_table
                                              .select('user_id','paper_id')
                                                        .withColumn('paper_ratings',lit(0)))
                                                .union(rating_1_table)#.show()
rating_matrix.show(10)


# <h4>Exercise 2. 3 ALS algorithm </h4>
# - a)Write a python program that employs ALS to fit a model from the dataframe you generated in the last task.
# - b)Use the learned model to generate top 10 recommendations for all users using the recommendForAllUsers method. 
# - c)In addition to that, output the top 10 recommendations for user with user hash id = 1eac022a97d683eace8815545ce3153f

# In[6]:


import pyspark.ml.recommendation as ml 
import pyspark.ml.evaluation as E

#We instantiate an als object bsed on default parametrs
als = ml.ALS(rank=10, 
             maxIter=5,
             seed=0,
             userCol="user_id", 
             itemCol="paper_id",
             ratingCol="paper_ratings")


# In[7]:


# Ex 2.3....find top 10 papers for 1eac022a97d683eace8815545ce3153f

# Part a: we fit a model on the entire dataset. 
model = als.fit(rating_matrix)

# Part b:select 10 recommendations for each user and filter the user_id we need
top_10_results=model.recommendForAllUsers(10).select('*').filter(condition).first().recommendations

        
#Part c: REcommendation for user:1eac022a97d683eace8815545ce3153f

    # FInd internal integer ID of the user searching in the mapping table of user_string-user_int_id.
    # Collect the result as an array of row objects.Get the 1st row from the row object
    # and return its integer_id to user_x variable
user_x= user_id_mapping.select('*').filter('user_id_string = "d45c0800b19e92ed7ee2db27ab2aae21"').collect()[0].id

    # we now create an SQL where condition and then use this where condition clause inside the filter
condition='user_id='+str(user_x)



    #we get a result in the form of an array of row objects 
    #  [Row(paper_id=9, rating=8.5), Row(paper_id=1, rating=8.45)..]
    #Print the paper_ids by iterating through the array
print("papers are",[i.paper_id for i in top_10_results])


# <h4>Exercise 2. 4 :Recommender System Evaluation</h4>
# 
# - (a) Split the ratings randomly into two sets: training set (contains 70% of the ratings) and test set (with
# 30% of the ratings)
# - (b)Fit a model on the training set.
# - (c)Calculate the Root Mean Squared Error (RMSE) over the test set.
# - (d)Generate top 10 recommendations over the test set as follows:
#     - Apply the learned model over the test set.
#     - For each user, order the papers by the predicted ratings descendingly.
#     - Choose the top 10 papers to be the recommendations for that user.

# In[8]:


# part a: use the inbuilt randomsplit function to split entire data into training and test
training_data,test_data = rating_matrix.randomSplit([70.0, 30.0])

# part b: We shall now fit the model on training dataframe:
#         we shall reuse the previous ALS object and fit our model on training dataset:
trained_model = als.fit(training_data)

#part c: To find RMSE over test data, we first transform our test data using the trained model

#        Transform test dataset using previous trained model
prediction_object=trained_model.transform(test_data)

#        the prediction_object has a 'prediction' column containing the predicted ratings.
#        Use rmse error calculator from the library Evaluation.RegressionCalculator 

#        Instatntiate the Evaluator object
evaluator = E.RegressionEvaluator(labelCol="paper_ratings",predictionCol="prediction",metricName="rmse")
#       Evaluate the RMSE of predicted dataset and print
rmse=evaluator.evaluate(prediction_object)
print("RMSE of the model on test data is:",rmse)

# part d: Top 10 recommendation for each user.Since it was not mentioned what to return, we return the paper_ids


#       creating a window for ranking the papers for each user based on prediction. 
window = Window.partitionBy("user_id")        .orderBy(desc("prediction"))
#       we add a row_no column based on our window specification. Method row_no() adds a seq no. starting from 1
top10_for_test_users=prediction_object                    .withColumn("recommendation_rank"                                ,row_number().over(window))
#       display the first 10 recommendations for each user
top10_for_test_users.filter(top10_for_test_users.recommendation_rank<10)                    .select('user_id','recommendation_rank','paper_id','prediction')                    .show(500)


# <h4>Exercise 2.5: Hyperparameter tuning </h4>
# - (a) search for the optimal value in the following set: rank ∈ {10, 25, 50}. Report the best value you find along with the corresponding RMSE.
# - (b)increase the maxIter, do you get better results? report your observation.

# In[11]:


#instantiate als object with default maxIter.
als_for_k=ml.ALS(maxIter=5,userCol="user_id", itemCol="paper_id",ratingCol="paper_ratings")

# part a:Create a parameter map for different values of "rank" parameter
paramGrid_for_rank = ParamGridBuilder()
                                        .addGrid(als_for_k.rank,
                                                 [10, 25, 50])
                                        .build()

#        Create cross validator object with your estimator algorithms,paprameter set and the evaluator. 
#         We use the previosly declared Regression evaluator
crossval_for_rank = CrossValidator(estimator=als_for_k,
                          estimatorParamMaps=paramGrid_for_rank,
                          evaluator=evaluator,
                          numFolds=2)

#         We fit our dataset in the above crossvalidator
crossval_ranked_model=crossval_for_rank.fit(rating_matrix)

#         Get the "rank" hyperparameter from the best model( returns an ALS model)
print("Least RMSE was found for rank=",      crossval_ranked_model.bestModel.rank)
print("the RMSE error was",crossval_ranked_model.avgMetrics[2])


# In[23]:


# part b: Experiment with maxIter hyperparametrs
#        instantiate als object with with a default rank.
als_for_iter=ml.ALS(rank=5,userCol="user_id", itemCol="paper_id",ratingCol="paper_ratings")

# Create a separate parameter map for maxIter
maxiter_values=[5,10,11,12,15,25]
    
paramGrid_for_iter = ParamGridBuilder()
                                    .addGrid(als_for_iter.maxIter, 
                                             maxiter_values)
                                .build()
crossval_iter = CrossValidator(estimator=als_for_iter,
                          estimatorParamMaps=paramGrid_for_iter,
                          evaluator=evaluator,
                          numFolds=2)
#we now fit our dataset in the above crossval
crossval_iter_model=crossval_iter.fit(rating_matrix)


# <h4> Observation on max-Iter </h4>
# Plotting the RMSE against max-iter parameters gives us an idea how its contribution decreases after a certain thresholdd
# We can see there is no muh significant decrease in RMSE after incresing iterations beyond 10

# In[29]:


# maxIter_rmse_map=zip(maxiter_values,crossval_iter_model.avgMetrics)
plt.axis(x=maxiter_values)
plt.plot(crossval_iter_model.avgMetrics)
plt.show()

