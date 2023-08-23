# PySpark Version to Implement PSM & DML

Open source python packages such as [EconML](https://github.com/microsoft/EconML) and [CausalML](https://github.com/uber/causalml) are
excellent and comprehensive for people to fulfill causal inference. And the combination for PSM and DML often performed better results.
For industry and large dataset, spark is a common engine for engineering. So this project is a simple revision to implement psm and dml
by PySpark. 

## Example

### Propensity Score Matching

```
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .getOrCreate()
    
def add_id(df):
    schema = df.schema.add(StructField("id", LongType()))
    rdd = df.rdd.zipWithIndex()
    
    def flat(l):
        for k in l:
            if not isinstance(k, (list, tuple)):
                yield k
            else:
                yield from flat(k)

    rdd = rdd.map(lambda x: list(flat(x)))
    df_with_id = spark.createDataFrame(rdd, schema).fillna(0)
    
    return df_with_id
```
```
from psm import PSM

treatment_group_ = spark.read.parquet('path to treatment file')
control_group_ = spark.read.parquet('path to control file')
all_group = treatment_group_.union(control_group_)
psm_df = add_id(all_group)

psm = PSM(spark, psm_df)

treatment_group, control_group = psm.fit(T='treatment column name')
```
#### Standard Mean Difference (SMD) Table
```
smd_table = psm.get_smd_table()
smd_table
```
#### Density Plot of Propensity Score
```
psm.get_propensity_plot(False)
psm.get_propensity_plot(True)
```

### Double Machine Learning
```
from dml import LinearDML

df = spark.read.parquet('path to data')
df = add_id(df)

est = LinearDML(spark, df, model_y='rf', model_t='rf', discrete_treatment=True, cv=2)
est.fit(Y='outcome column name', T='treatment column name')
```
#### Average Treatment Effect
```
est.get_ate(decimals=4)
```
#### Individual Treatment Effect
```
est.get_ite(decimals=4)
```
