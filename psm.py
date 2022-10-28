from warnings import warn

import numpy as np
import pandas as pd

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from dml import get_estimator

vec2arr = F.udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
arr2vec = F.udf(lambda v: Vectors.dense(v), VectorUDT())

class PSM():

    def __init__(self,
                 spark,
                 data,
                 model='xgb',
                 discrete_treatment=True,
                 categories='auto',
                 test_ratio=0.2,
                 sdcal=0.2,
                 random_state=None):

        self.spark = spark
        self.data = data
        self.model = model
        self.discrete_treatment = discrete_treatment
        self.categories = categories
        self.test_ratio = test_ratio
        self.sdcal = sdcal
        self.random_state = random_state

        if self.discrete_treatment == False:
            raise ValueError("Parameter `discrete_treatment` cannot be true in PSM calculating")

    # override only so that we can update the docstring to indicate support for `LinearModelFinalInference`
    def fit(self, T, X=None, id='id', sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):

        data_cols = self.data.columns
        X = [x for x in data_cols if x != T and x != id] if X is None else X

        self.T = T
        self.X = X
        self.id_col = id

        self.n_obs = self.data.count()

        data = self.data if id == 'id' else self.data.withColumnRenamed(id, 'id')
        assembler = VectorAssembler(
            inputCols=X,
            outputCol="X")

        data = assembler.transform(data)

        train_data, test_data = data.randomSplit([1 - self.test_ratio, self.test_ratio])

        estimator = get_estimator(self.model, T, self.discrete_treatment, self.random_state)

        model = estimator.fit(train_data)
        predict_train = model.transform(train_data)
        predict_test = model.transform(test_data)
        evaluator = BinaryClassificationEvaluator(labelCol=T, rawPredictionCol="rawPrediction")
        print('Train AUC：', evaluator.evaluate(predict_train))
        print('Test AUC：', evaluator.evaluate(predict_test))

        data_pre = model.transform(data).withColumn('%s_prediction' % T, vec2arr('%s_prediction' % T)[1])
        mean_hat, sttdev_hat = data_pre.select(F.mean('%s_prediction' % T), F.stddev('%s_prediction' % T)).first()
        data_pre = data_pre.withColumn('%s_scaled' % T, (F.col('%s_prediction' % T) - mean_hat) / sttdev_hat).cache()
        self.data_pre = data_pre

        treatment_pre = data_pre.where('%s = 1' % T).select(id, '%s_scaled' % T).toDF('id1', 'scale1')
        control_pre = data_pre.where('%s = 0' % T).select(id, '%s_scaled' % T).toDF('id0', 'scale0')

        nearest_neighbors = treatment_pre.crossJoin(control_pre).withColumn('distance', F.abs(F.col('scale1') - F.col('scale0'))) \
            .withColumn('rank', F.row_number().over(Window.partitionBy('id1').orderBy(F.asc('distance')))).where(
            'rank=1 and distance<=%f' % self.sdcal).cache()

        self.treatment_set = nearest_neighbors.withColumnRenamed('id1', 'id').select('id').join(data, ['id'], 'left').cache()
        self.control_set = nearest_neighbors.withColumnRenamed('id0', 'id').select('id').join(data, ['id'], 'left').cache()

        return self.treatment_set.select([id]+X+[T]).cache(), self.control_set.select([id]+X+[T])


    def get_smd_table(self):
        treatment_df = self.treatment_set.select(self.X).describe().toPandas().T[[1, 2]][1:].rename(
            columns={1: 'treatment_mean', 2: 'treatment_std'}).apply(pd.to_numeric)
        control_df = self.control_set.select(self.X).describe().toPandas().T[[1, 2]][1:].rename(
            columns={1: 'control_mean', 2: 'control_std'}).apply(pd.to_numeric)
        combine_df = pd.concat([treatment_df, control_df], axis=1)
        combine_df['SMD'] = (combine_df.treatment_mean - combine_df.control_mean) / np.sqrt(
            0.5 * (np.square(combine_df.treatment_std) + np.square(combine_df.control_std)))

        return combine_df

    def get_propensity_plot(self, after_fitting=True):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        plt.style.use('fivethirtyeight')
        rcParams['figure.figsize'] = 8, 8

        if after_fitting == True:
            control_group = self.control_set.select('id').join(self.data_pre, ['id'], 'left')
            treatment_group = self.treatment_set.select('id').join(self.data_pre, ['id'], 'left')
        else:
            control_group = self.data_pre.where('%s = 0' % self.T)
            treatment_group = self.data_pre.where('%s = 1' % self.T)

        ax = sns.distplot(
            control_group.select('%s_prediction' % self.T).toPandas())
        sns.distplot(
            treatment_group.select('%s_prediction' % self.T).toPandas())
        ax.set_xlim(0, 1)
        ax.set_xlabel("propensity scores")
        ax.set_ylabel('density')
        ax.legend(['Control', 'Treatment'])

        plt.show()