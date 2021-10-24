# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.7

import json

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
from pyspark.sql.types import StructType, StringType, IntegerType, DecimalType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

checkpoint_location = "tmp/ml_checkpoint"

# upload the best model
kafka_brokers = "bigdataanalytics2-worker-shdpt-v31-1-2:6667"
topic = "topic_suvorova910_10_ml"
model_dir = "ml_data/models"
model = GBTClassificationModel.load(model_dir + "/model_1")
pipeline_dir = "ml_data/my_pipeline"
pipelineModel = PipelineModel.load(pipeline_dir + "/pipeline_model_1")
target_row_topic = "predictions_modified_topic_row"

# stream mode
raw_data = spark.readStream. \
    format("kafka"). \
    option("kafka.bootstrap.servers", kafka_brokers). \
    option("subscribe", topic). \
    option("startingOffsets", "earliest"). \
    option("maxOffsetsPerTrigger", "5"). \
    load()

# Change 'value' column type to STRING
value_data = raw_data \
    .select(F.col("value").cast("String"), "offset")

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


def prepare_data(df):
    data = df \
        .withColumn("PassengerId", F.col("PassengerId").cast(IntegerType())) \
        .withColumn("Age", F.col("Age").cast(IntegerType())) \
        .withColumn("Fare", F.col("Fare").cast(DecimalType()))
    data = data.where(F.col("Name").isNotNull())
    mean_value_age = data.agg({"Age": "mean"}).collect()[0][0]
    data = data.na.fill(value=mean_value_age, subset=["Age"])
    data = data.na.fill(value="S", subset=["Embarked"])
    data = data.withColumn("Embarked", F.when(F.col("Embarked") == "", "S").otherwise(F.col("Embarked")))
    data = data.withColumn("Pclass", F.when(F.col("Pclass") == "", "3").otherwise(F.col("Pclass")))
    data = data.withColumn("Sex", F.when(F.col("Sex") == "", "male").otherwise(F.col("Sex")))
    data = data.withColumn("SibSp", F.when(F.col("SibSp") == "", "0").otherwise(F.col("SibSp")))
    data = data.withColumn("Parch", F.when(F.col("Parch") == "", "0").otherwise(F.col("Parch")))
    data_model = data.select("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
    data_model = pipelineModel.transform(data_model)
    return data_model.select('features')

def process_batch(df, epoch):
    model_data = prepare_data(df)
    prediction = model.transform(model_data)
    prediction.show()


def foreach_batch_output(df):
    from datetime import datetime as dt
    date = dt.now().strftime("%Y%m%d%H%M%S")
    return df \
        .writeStream \
        .trigger(processingTime='%s seconds' % 10) \
        .foreachBatch(process_batch) \
        .option("checkpointLocation", checkpoint_location + "/" + date) \
        .start()


stream = foreach_batch_output(parsed_data)
stream.stop()

#save to Kafka
def process_batch_for_kafka(df, epoch):
    model_data = prepare_data(df)
    prediction = model.transform(model_data)
    return prediction.selectExpr("CAST(null AS STRING) as key",
                                                          "CAST(struct(*) AS STRING) as value") \
        .write \
        .format("kafka") \
        .option("topic", "predictions_modified_topic_row") \
        .option("kafka.bootstrap.servers", kafka_brokers) \
        .save()


def foreach_batch_kafka(df):
    from datetime import datetime as dt
    date = dt.now().strftime("%Y%m%d%H%M%S")
    return df \
        .writeStream \
        .trigger(processingTime='%s seconds' % 10) \
        .foreachBatch(process_batch_for_kafka) \
        .option("checkpointLocation", checkpoint_location + "/" + date) \
        .start()

stream = foreach_batch_kafka(parsed_data)
stream.stop()


