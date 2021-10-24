#to transform csv to json
import csv
import json

csvfile = open('train_titanic_suvorova.csv', 'r')
jsonfile = open('train_titanic_suvorova.json', 'w')

reader = csv.DictReader(csvfile)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')

# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.7

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StringType, IntegerType, DecimalType
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

kafka_brokers = "bigdataanalytics2-worker-shdpt-v31-1-2:6667"
topic = "topic_suvorova910_10_ml"
model_dir = "ml_data/models"
pipeline_dir = "ml_data/my_pipeline"

 #DataFrame in the batch mode #
# =========================== #
raw_data = spark\
    .read\
    .format("kafka")\
    .option("kafka.bootstrap.servers", kafka_brokers)\
    .option("subscribe", topic)\
    .option("startingOffsets", "earliest")\
    .load()

# Check on records structure
raw_data.show()

# Change 'value' column type to STRING
value_data = raw_data \
    .select(F.col("value").cast("String"), "offset")

value_data.printSchema()

# schema of the json struct
schema = StructType() \
    .add("PassengerId", StringType()) \
    .add("Survived", StringType()) \
    .add("Pclass", StringType()) \
    .add("Name", StringType()) \
    .add("Sex", StringType()) \
    .add("Age", StringType()) \
    .add("SibSp", StringType()) \
    .add("Parch", StringType()) \
    .add("Ticket", StringType()) \
    .add("Fare", StringType()) \
    .add("Cabin", StringType()) \
    .add("Embarked", StringType())


# extract json from string column 'value' and rename it as 'value'
from_json_data = value_data \
    .select(F.from_json(F.col("value"), schema).alias("data"), "offset")


# Flat the value structure (alike SELECT t.* FROM table t)
parsed_data = from_json_data.select("data.*", "offset")
parsed_data.show()

# change types
data = parsed_data \
    .withColumn("PassengerId", F.col("PassengerId").cast(IntegerType())) \
    .withColumn("Age", F.col("Age").cast(IntegerType())) \
    .withColumn("Fare", F.col("Fare").cast(DecimalType()))


data.printSchema()
data.show()

#Delete an empty row
data = data.where(F.col("Name").isNotNull())

#Replace mean for null on only age column
mean_value_age = data.agg({"Age": "mean"}).collect()[0][0]
print(mean_value_age)
data = data.na.fill(value=mean_value_age, subset=["Age"])

data = data.na.fill(value="S", subset=["Embarked"])
data = data.withColumn("Embarked", \
       F.when(F.col("Embarked")=="", "S") \
          .otherwise(F.col("Embarked")))

data = data.withColumn("Pclass", \
       F.when(F.col("Pclass")=="", "3") \
          .otherwise(F.col("Pclass")))

data = data.withColumn("Sex", \
       F.when(F.col("Sex")=="", "male") \
          .otherwise(F.col("Sex")))

data = data.withColumn("SibSp", \
       F.when(F.col("SibSp")=="", "0") \
          .otherwise(F.col("SibSp")))

data = data.withColumn("Parch", \
       F.when(F.col("Parch")=="", "0") \
          .otherwise(F.col("Parch")))

# model evaluator
evaluator = BinaryClassificationEvaluator() \
        .setMetricName("areaUnderROC") \
        .setLabelCol("label") \
        .setRawPredictionCol("prediction")

data.show()

# use Category Indexing, One-Hot Encoding and VectorAssembler
categoricalColumns = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + "Index")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = "Survived", outputCol = "label")
stages += [label_stringIdx]
numericColumns = ["Age", "Fare"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericColumns
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

#choose columns
df = data.select("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
df.show(1000)

#pipeline
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(df)
pipelineModel.write().overwrite().save(pipeline_dir + "/pipeline_model_1")

df = pipelineModel.transform(df)
selectedCols = ['label', 'features']
df = df.select(selectedCols)


#split data into train and test sets
train, test = df.randomSplit([0.7, 0.3], seed = 2018)

#use Gradient-Boosted Tree Classifier
gbt = GBTClassifier(maxIter=1)
gbtModel = gbt.fit(train)
predictions = gbtModel.transform(test)
predictions.show(5)
evaluation_result = evaluator.evaluate(predictions)
print("Evaluation result: {}".format(evaluation_result))
gbtModel.write().overwrite().save(model_dir + "/model_1")










