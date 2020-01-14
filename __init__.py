import os
from pyspark.sql import SparkSession

try:
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    spark.sparkContext.addPyFile("hdfs://datalakeprod/prod/RiskandRevenue/pypackages/l2cmodel.zip")
except:
    if os.environ.get("UNIT_TEST") == "1":
        pass