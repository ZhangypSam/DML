import warnings

import numpy as np
import pandas as pd

from scipy.stats import norm
from functools import reduce
from collections import OrderedDict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from sparkxgb import XGBoostClassifier
from sparkxgb import XGBoostRegressor

from econml.grf import CausalForest, MultiOutputGRF
from econml.utilities import cross_product
from utilities import expand_treatments

vec2arr = F.udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))
arr2vec = F.udf(lambda v: Vectors.dense(v), VectorUDT())


def get_estimator(model_name, target, discrete_treatment, random_state):
    if model_name == 'rf':
        if discrete_treatment == False:
            estimator = RandomForestRegressor(featuresCol='X', labelCol=target, predictionCol='%s_prediction' % target,
                                              seed=random_state, numTrees=300)
        else:
            estimator = RandomForestClassifier(featuresCol='X', labelCol=target,
                                               probabilityCol='%s_prediction' % target, seed=random_state, numTrees=300)
    elif model_name == 'xgb':
        if discrete_treatment == False:
            estimator = XGBoostRegressor(missing=float(0.0), featuresCol='X', labelCol=target,
                                         predictionCol='%s_prediction' % target, numRound=25, seed=random_state)
        else:
            estimator = XGBoostClassifier(missing=float(0.0), featuresCol='X', labelCol=target,
                                          probabilityCol='%s_prediction' % target, numRound=25, seed=random_state)
    else:
        if discrete_treatment == False:
            estimator = LinearRegression(featuresCol='X', labelCol=target, predictionCol='%s_prediction' % target,
                                         regParam=0.2, elasticNetParam=1)
        else:
            estimator = LogisticRegression(featuresCol='X', labelCol=target, probabilityCol='%s_prediction' % target,
                                           regParam=0.2, elasticNetParam=1)

    return estimator


def cv_fit_predict(splits, cv, estimator):
    train_sets, test_sets = [], []
    for i in range(cv):
        test_sets.append(splits[i])
        train_sets.append(reduce(DataFrame.unionAll, splits[0:i] + splits[i + 1:cv]))

    def fit_predict(data):
        model = estimator.fit(data[0])
        predict = model.transform(data[1])
        return predict

    predicts = list(map(fit_predict, (train_sets, test_sets)))

    return reduce(DataFrame.unionAll, predicts)


class LinearDML():

    def __init__(self,
                 spark,
                 data,
                 model_y='auto',
                 model_t='auto',
                 model_final='linear',
                 linear_first_stages=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 random_state=None):

        self.spark = spark
        self.data = data
        self.model_y = model_y
        self.model_t = model_t
        self.model_final = model_final
        self.linear_first_stages = linear_first_stages
        self.discrete_treatment = discrete_treatment
        self.categories = categories
        self.cv = cv
        self.random_state = random_state

    # override only so that we can update the docstring to indicate support for `LinearModelFinalInference`
    def fit(self, Y, T, X=None, id='id', ite_stat=True, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):

        data_cols = self.data.columns
        X = [x for x in data_cols if x != Y and x != T and x != id] if X is None else X

        self.Y = Y
        self.T = T
        self.X = X
        self.id_col = id
        self.ite_stat = ite_stat

        self.n_obs = self.data.count()

        data = self.data if id == 'id' else self.data.withColumnRenamed(id, 'id')
        assembler = VectorAssembler(
            inputCols=X,
            outputCol="X")
        data = assembler.transform(data)

        splits = [data.where(F.col('id') - F.lit(self.cv) * F.floor(F.col('id') / F.lit(self.cv)) == F.lit(i)) for i in
                  range(self.cv)]

        estimator_t = get_estimator(self.model_t, T, self.discrete_treatment, self.random_state)
        estimator_y = get_estimator(self.model_y, Y, False, self.random_state)

        predict_t = cv_fit_predict(splits, self.cv, estimator_t).select('id', 'X', T, '%s_prediction' % T)
        if self.discrete_treatment:
            predict_t = predict_t.withColumn('%s_prediction' % T, vec2arr('%s_prediction' % T)[1])
        predict_y = cv_fit_predict(splits, self.cv, estimator_y).select('id', 'X', Y, '%s_prediction' % Y)

        result = predict_t.join(predict_y, ['id', 'X']) \
            .withColumn("u_hat", F.col(Y) - F.col("%s_prediction" % Y)) \
            .withColumn("v_hat", F.col(T) - F.col("%s_prediction" % T)) \
            .withColumn("x_add1", F.concat(F.array(F.lit(1)), vec2arr(F.col("X")))) \
            .withColumn("fts", arr2vec(F.expr("transform(x_add1, x -> x * v_hat)"))).cache()

        assembler = VectorAssembler(
            inputCols=['fts'],
            outputCol="X_1")

        lr = LinearRegression(featuresCol='X_1', labelCol='u_hat', predictionCol='pred', regParam=0.0)

        data = assembler.transform(result)
        model = lr.fit(data)
        self._param = model.coefficients.toArray()
        param_str = 'array(' + ','.join(list(map(str, self._param))) + ')'
        self.spark_data = data.withColumn('ite', F.expr(
            'aggregate(zip_with(%s, x_add1, (x, y) -> x * y), 0D, (x, y) -> (x + y))' % param_str)).cache()

        if self.ite_stat:
            self.id = np.squeeze(np.array(data.select("id").collect()))
            self._pred = np.squeeze(np.array(data.select("ite").collect()))
            WX = np.squeeze(np.array(data.select("x_add1").collect()))
            Tres = np.squeeze(np.array(data.select("v_hat").collect()))
            WXT = cross_product(WX, Tres)
            wy = np.squeeze(np.array(data.select("u_hat").collect()))

            sigma_inv = np.linalg.pinv(np.matmul(WXT.T, WXT))
            var_i = (wy - np.matmul(WXT, self._param)) ** 2
            df = len(self._param)
            if self.n_obs <= df:
                warnings.warn("Number of observations <= than number of parameters. Using biased variance calculation!")
                correction = 1
            else:
                correction = (self.n_obs / (self.n_obs - df))

            weighted_sigma = np.matmul(WXT.T, WXT * var_i.reshape(-1, 1))
            self._param_var = correction * np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv))

            self._pred_se = np.sqrt(np.clip(np.sum(np.matmul(WX, self._param_var) * WX, axis=1), 0, np.inf))

    def get_ate(self, alpha=0.05, decimals=3):

        if self.ite_stat:
            self.pred = np.mean(self._pred, axis=0)
            self.pred_se = np.sqrt(np.mean(self._pred_se ** 2, axis=0))
            self.zstat = self.pred / self.pred_se
            self.pvalue = norm.sf(np.abs(self.zstat), loc=0, scale=1) * 2
            self.ci_lower = norm.ppf(alpha / 2, loc=self.pred, scale=self.pred_se)
            self.ci_upper = norm.ppf(1 - alpha / 2, loc=self.pred, scale=self.pred_se)

            to_include = OrderedDict()
            to_include['point_estimate'] = self.pred
            to_include['stderr'] = self.pred_se
            to_include['zstat'] = self.zstat
            to_include['pvalue'] = self.pvalue
            to_include['ci_lower'] = self.ci_lower
            to_include['ci_upper'] = self.ci_upper

            res = pd.DataFrame(to_include, index=pd.Index(['ATE'])).round(decimals)

        else:
            res = self.spark_data.select('ite').summary()

        return res

    def get_ite(self, alpha=0.05, decimals=3):

        if self.ite_stat:
            self._zstat = self._pred / self._pred_se
            self._pvalue = norm.sf(np.abs(self._zstat), loc=0, scale=1) * 2
            self._ci_lower = norm.ppf(alpha / 2, loc=self._pred, scale=self._pred_se)
            self._ci_upper = norm.ppf(1 - alpha / 2, loc=self._pred, scale=self._pred_se)

            to_include = OrderedDict()
            to_include['point_estimate'] = self._pred
            to_include['stderr'] = self._pred_se
            to_include['zstat'] = self._zstat
            to_include['pvalue'] = self._pvalue
            to_include['ci_lower'] = self._ci_lower
            to_include['ci_upper'] = self._ci_upper

            res = pd.DataFrame(to_include, index=pd.Index(self.id)).round(decimals)

        else:
            res = self.spark_data.select('id', 'X', self.T, self.Y, 'ite')

        return res


class CausalForestDML():

    def __init__(self,
                 spark,
                 data,
                 model_y='auto',
                 model_t='auto',
                 model_final='linear',
                 linear_first_stages=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 random_state=2021):

        self.spark = spark
        self.data = data
        self.model_y = model_y
        self.model_t = model_t
        self.model_final = model_final
        self.linear_first_stages = linear_first_stages
        self.discrete_treatment = discrete_treatment
        self.categories = categories
        self.cv = cv
        self.random_state = random_state

    # override only so that we can update the docstring to indicate support for `LinearModelFinalInference`
    def fit(self, Y, T, X=None, id='id', sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):

        data_cols = self.data.columns
        X = [x for x in data_cols if x != Y and x != T and x != id] if X is None else X

        self.Y = Y
        self.T = T
        self.X = X
        self.id_col = id

        self.n_obs = self.data.count()

        data = self.data if id == 'id' else self.data.withColumnRenamed(id, 'id')
        assembler = VectorAssembler(
            inputCols=X,
            outputCol="X")
        data = assembler.transform(data).cache()
        data.count()

        splits = [data.where(F.col('id') - F.lit(self.cv) * F.floor(F.col('id') / F.lit(self.cv)) == F.lit(i)) for i in
                  range(self.cv)]

        estimator_t = get_estimator(self.model_t, T, self.discrete_treatment, self.random_state)
        estimator_y = get_estimator(self.model_y, Y, False, self.random_state)

        predict_t = cv_fit_predict(splits, self.cv, estimator_t).select('id', 'X', T, '%s_prediction' % T)
        if self.discrete_treatment:
            predict_t = predict_t.withColumn('%s_prediction' % T, vec2arr('%s_prediction' % T)[1])
        predict_y = cv_fit_predict(splits, self.cv, estimator_y).select('id', 'X', Y, '%s_prediction' % Y)

        result = predict_t.join(predict_y, ['id', 'X']) \
            .withColumn("u_hat", F.col(Y) - F.col("%s_prediction" % Y)) \
            .withColumn("v_hat", F.col(T) - F.col("%s_prediction" % T)).cache()

        self.id = np.squeeze(np.array(result.select("id").collect()))
        WX = np.squeeze(np.array(result.select(vec2arr("X")).collect()))
        Y_res = np.squeeze(np.array(result.select("u_hat").collect()))
        T_res = np.squeeze(np.array(result.select("v_hat").collect()))

        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))

        self.cfdml = MultiOutputGRF(CausalForest(n_estimators=100,
                                                 criterion="mse",
                                                 max_depth=None,
                                                 min_samples_split=10,
                                                 min_samples_leaf=5,
                                                 min_weight_fraction_leaf=0.,
                                                 min_var_fraction_leaf=None,
                                                 min_var_leaf_on_val=False,
                                                 max_features="auto",
                                                 min_impurity_decrease=0.,
                                                 max_samples=.45,
                                                 min_balancedness_tol=.45,
                                                 honest=True,
                                                 inference=True,
                                                 fit_intercept=True,
                                                 subforest_size=4,
                                                 n_jobs=-1,
                                                 random_state=self.random_state,
                                                 verbose=0,
                                                 warm_start=False))

        self.cfdml.fit(WX, T_res, Y_res, sample_weight=sample_weight)

        WX, T0, T1 = expand_treatments(WX, 0, 1)
        dT = T1 - T0
        if dT.ndim == 1:
            dT = dT.reshape((-1, 1))
        pred, pred_var = self.cfdml.predict_projection_and_var(WX, dT)

        self._pred = pred.reshape((-1,))
        self._pred_se = np.sqrt(pred_var.reshape((-1,)))

    def _get_new_X_pred_and_var(self, data_X):

        data = data_X if self.id_col == 'id' else data_X.withColumnRenamed(id, 'id')
        assembler = VectorAssembler(
            inputCols=self.X,
            outputCol="X")
        data = assembler.transform(data)
        id = np.squeeze(np.array(data.select("id").collect()))
        WX = np.squeeze(np.array(data.select(vec2arr("X")).collect()))

        WX, T0, T1 = expand_treatments(WX, 0, 1)
        dT = T1 - T0
        if dT.ndim == 1:
            dT = dT.reshape((-1, 1))
        pred, pred_var = self.cfdml.predict_projection_and_var(WX, dT)

        pred_X = pred.reshape((-1,))
        pred_se_X = np.sqrt(pred_var.reshape((-1,)))

        return pred_X, pred_se_X, id

    def get_ate(self, data_X=None, alpha=0.05, decimals=3):

        if data_X != None:
            data_X = data_X.cache()
            data_X.count()
            pred_X, pred_se_X, _ = self._get_new_X_pred_and_var(data_X)
            self.pred = np.mean(pred_X, axis=0)
            self.pred_se = np.sqrt(np.mean(pred_se_X ** 2, axis=0))
        else:
            self.pred = np.mean(self._pred, axis=0)
            self.pred_se = np.sqrt(np.mean(self._pred_se ** 2, axis=0))

        self.zstat = self.pred / self.pred_se
        self.pvalue = norm.sf(np.abs(self.zstat), loc=0, scale=1) * 2
        self.ci_lower = norm.ppf(alpha / 2, loc=self.pred, scale=self.pred_se)
        self.ci_upper = norm.ppf(1 - alpha / 2, loc=self.pred, scale=self.pred_se)

        to_include = OrderedDict()
        to_include['point_estimate'] = self.pred
        to_include['stderr'] = self.pred_se
        to_include['zstat'] = self.zstat
        to_include['pvalue'] = self.pvalue
        to_include['ci_lower'] = self.ci_lower
        to_include['ci_upper'] = self.ci_upper

        res = pd.DataFrame(to_include, index=pd.Index(['ATE'])).round(decimals)

        return res

    def get_ite(self, data_X=None, alpha=0.05, decimals=3):

        to_include = OrderedDict()

        if data_X != None:
            data_X = data_X.cache()
            data_X.count()
            pred_X, pred_se_X, id_X = self._get_new_X_pred_and_var(data_X)
            to_include['point_estimate'] = pred_X
            to_include['stderr'] = pred_se_X
            self._zstat = pred_X / pred_se_X
            self._ci_lower = norm.ppf(alpha / 2, loc=pred_X, scale=pred_se_X)
            self._ci_upper = norm.ppf(1 - alpha / 2, loc=pred_X, scale=pred_se_X)
        else:
            to_include['point_estimate'] = self._pred
            to_include['stderr'] = self._pred_se
            self._zstat = self._pred / self._pred_se
            self._ci_lower = norm.ppf(alpha / 2, loc=self._pred, scale=self._pred_se)
            self._ci_upper = norm.ppf(1 - alpha / 2, loc=self._pred, scale=self._pred_se)

        self._pvalue = norm.sf(np.abs(self._zstat), loc=0, scale=1) * 2

        to_include['zstat'] = self._zstat
        to_include['pvalue'] = self._pvalue
        to_include['ci_lower'] = self._ci_lower
        to_include['ci_upper'] = self._ci_upper

        if data_X != None:
            res = pd.DataFrame(to_include, index=pd.Index(id_X)).round(decimals)
        else:
            res = pd.DataFrame(to_include, index=pd.Index(self.id)).round(decimals)

        return res
