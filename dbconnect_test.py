from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("maven_e2emlops_dbw_stan").getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)