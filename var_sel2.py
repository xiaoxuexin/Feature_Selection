import warnings
import subprocess
import uuid
from pyspark.sql import SparkSession
warnings.filterwarnings('ignore')
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

from l2cmodel.l2cTrainer import *
from l2cmodel.l2cTrainer import writeToJsonRenameOldFile
import random
import numpy as np
import pandas as pd
from rocktrain.ptrEvaluator import PTREval
from l2cmodel.util import getData, getGlobals, globalAttr, setGlobals, getSaveResult

VARIABLE_COLS = ['banker_performance', 'borrowerage', 'cashout', 'creditscore', 'dti', 'fulldoc',
                 'hasjointprimaryopencount',
                 'haspropertyinspectwaiver', 'hassuspense', #'hedgegroup___garm_30yr',
                 #                   'income',
                 'highltv', 'interestrate', 'iscondopropertytype', 'isforward', 'ispudpropertytype',
                 'ispurchaseloan', 'isqlms', 'isrelo', 'issingleborrower', 'issinglepropertytype', 'istenureover10',
                 'loanamount', 'loantype___fhafixed', 'lock_age', 'ltv', 'nocashamt', 'nothasappraisalvalue',
                 'numberoflockextensions',
                 'numberstatus',
                 'requiredpts', 'retentionclient', 'selfemployflg', 'status_age_businessdays',
                 'status_age_zeronotsuspense',# 'status_age_zerosuspense_businessdays',
                 'status_split___approved_appraised', 'status_split___prefolder',  # 'status_split___signoff',
                 # 'status_split___suspense_ana', 'status_split___suspense_collateral',
                 'status_split___suspense_other',
                 'statusgroup___folder', 'statusgroup___prefolder', 'statusgroup___signoff', 'statusgroup___suspense',
                 'statusgroup___approved',
                 # 'statusgroup_split35___approved_appraised', 'statusgroup_split35___folder',
                 # 'statusgroup_split35___prefolder', 'statusgroup_split35___signoff', 'statusgroup_split35___suspense',
                 'suspensereason_ana', 'suspensereason_ana_zero', 'suspensereason_collateral',
                 'suspensereason_collateral_zero', 'suspensereason_title_zero', 'tenure', 'timesinsuspense',
                 #                   'totinsuramt',
                 'log_income', 'log_totinsuramt'
                 ]

GENERAL_COLS = ["observation_dt", "loannumber", "loanamount", "hedgestatus", "submodel", "loanlevel_score"]

TOTAL_COLS = VARIABLE_COLS + GENERAL_COLS
BACK_TEST_PERIOD = None
TRAIN_PERIOD = '2017-01-01', '2018-12-31'
FORWARD_TEST_PERIOD = '2019-01-01', '2019-07-15'
_, _, _, LABEL_COL = globalAttr()
VERSION = '_v0.1_var_selection_xin'
SUBMODEL_NAME = 'GOVT_15YR_FHA'

spark.catalog.clearCache()
setGlobals(TRAIN_PERIOD=TRAIN_PERIOD, FORWARD_TEST_PERIOD=FORWARD_TEST_PERIOD,
           BACK_TEST_PERIOD=BACK_TEST_PERIOD)

uuid_4 = str(uuid.uuid4().int)
train_tb = "var_sel_" + SUBMODEL_NAME + uuid_4 + "_train"
test_tb = "var_sel_" + SUBMODEL_NAME + uuid_4 + "_test"
path_train = f"hdfs://datalakeprod/prod/RiskandRevenue/model_hedge/temp/{train_tb}"
path_test = f"hdfs://datalakeprod/prod/RiskandRevenue/model_hedge/temp/{test_tb}"
TRAIN_DF = getData(TOTAL_COLS, SUBMODEL_NAME, TRAIN_PERIOD).dropna(subset=VARIABLE_COLS)
TRAIN_DF.write.saveAsTable(f'lock2close_conformed.{train_tb}', mode='overwrite', path=path_train,
                           format='orc', partitionBy="observation_dt")
TEST_DF = getData(TOTAL_COLS, SUBMODEL_NAME, FORWARD_TEST_PERIOD).dropna(subset=VARIABLE_COLS)
TEST_DF.write.saveAsTable(f'lock2close_conformed.{test_tb}', mode='overwrite', path=path_test, format='orc',
                           partitionBy="observation_dt")

try:
    spark.catalog.clearCache()
    spark.catalog.refreshTable("lock2close_conformed.first_train_xin")
    spark.catalog.refreshTable("lock2close_conformed.first_test_xin")
    print("DEBUG TABLE REFRESH")
except:
    pass


def get_initial_coor(dimension):
    # index of the randomly selected starting point
    initial_index = random.sample(range(0, dimension), int(np.ceil(dimension / 2)))
    # coordinate of the starting point
    initial_coordinate = np.zeros(dimension)
    for i in initial_index:
        initial_coordinate[i] = 1
    return initial_coordinate


def get_neighbor(coordinate, dimension):
    # nearest neighbors of the point using coordinate
    nearest_neighbor = [coordinate.copy() for i in range(dimension)]
    for i in range(dimension):
        nearest_neighbor[i][i] = 1 - nearest_neighbor[i][i]
    return nearest_neighbor


def not_vis_neighbor(dimension, next_step_coordinate, have_visited):
    nearest_neighbor = []
    for i in range(0, dimension):
        current_neighbor = next_step_coordinate.copy()
        current_neighbor[i] = 1 - current_neighbor[i]
        if len(current_neighbor) != dimension:
            print("error in the size of current neighbor")
            return 0
        else:
            if not any((current_neighbor == x).all() for x in have_visited):
                nearest_neighbor.append(current_neighbor)
    return nearest_neighbor


def coordinate2attribute(coor, col_var):
    selected_variable = []
    # coor = coor.copy()
    # col_var = col_var.copy()
    for i in range(len(coor)):
        if coor[i] == 1:
            selected_variable.append(col_var[i])
    return selected_variable


def attribute2coordinate(attr_list, col_var):
    coordinate = np.zeros(len(col_var))
    for i in range(len(col_var)):
        if col_var[i] in attr_list:
            coordinate[i] = 1
    return coordinate


def get_score(var_col, training_pred, test_pred):
    getdata_cols = var_col + GENERAL_COLS
    TRAIN_DF_temp = spark.table(f"lock2close_conformed.{train_tb}")
    TEST_DF_temp = spark.table(f"lock2close_conformed.{test_tb}")
    tr = TrainerModel(data=TRAIN_DF_temp, labelCol=LABEL_COL)
    tr.vectorAssembler(inputCols=var_col)
    tr.logisticRegression(customEvaluatorClass=PTREval, regParam=0.0, maxIter=100)
    transformedData = tr.buildPipeline()
    trainEval = tr.evaluationResult
    pred_transform, pred_score = tr.predict(TEST_DF_temp)
    return pred_score[0]['score']


def start_score_list(coordinate, var_col, TRAIN_PERIOD, FORWARD_TEST_PERIOD):
    col = coordinate2attribute(coordinate, var_col)
    score_list = [get_score(col, TRAIN_PERIOD, FORWARD_TEST_PERIOD)]  # list of scores
    return score_list


def append_score_list(score_list, nearest_neighbor, var_col, TRAIN_PERIOD, FORWARD_TEST_PERIOD):
    # apply score to every component in nearest neighbor to get score list
    for coordinate in nearest_neighbor:
        col = coordinate2attribute(coordinate, var_col)
        score_list.append(get_score(col, TRAIN_PERIOD, FORWARD_TEST_PERIOD))
    return score_list


def buildDataFrame(next_step_coordinate, var_col, score, step):
    pandas_df = pd.DataFrame([[coordinate2attribute(next_step_coordinate, var_col)]], columns=['Variables list'])
    pandas_df['score'] = score
    pandas_df['length'] = sum(next_step_coordinate)
    pandas_df['step'] = step
    return pandas_df


def variable_selection(var_col, TRAIN_PERIOD, FORWARD_TEST_PERIOD):
    step = 0
    dimension = len(var_col)  # number of variables in our consideration
    if dimension <= 1:
        return 0
    min_score_dic = {}
    initial_coordinate = get_initial_coor(dimension)
    score_list = start_score_list(initial_coordinate, var_col, TRAIN_PERIOD, FORWARD_TEST_PERIOD)
    have_visited = [initial_coordinate]  # points have been visited

    # nearest neighbors of the initial point
    nearest_neighbor = get_neighbor(initial_coordinate, dimension)

    score_list = append_score_list(score_list, nearest_neighbor, var_col, TRAIN_PERIOD,
                                   FORWARD_TEST_PERIOD)

    have_visited = have_visited + nearest_neighbor  # append the nearest neighbor to have visited list
    next_step_coordinate = initial_coordinate

    pandas_df = buildDataFrame(next_step_coordinate, var_col, score_list[0], 0)

    while np.argmin(score_list) != 0: # (3 step ago result - result) < 0.001
        min_score_dic[step] = score_list[0]
        if step > 2:
            if (min_score_dic[step - 3] - min_score_dic[step]) < 0.001:
                return pandas_df

        next_step_coordinate = nearest_neighbor[np.argmin(score_list) - 1]  # first one in score list is current point
        nearest_neighbor = not_vis_neighbor(dimension, next_step_coordinate, have_visited)
        score_list = [min(score_list)]
        # score_list = start_score_list(next_step_coordinate, var_col, TRAIN_PERIOD, FORWARD_TEST_PERIOD)
        score_list = append_score_list(score_list, nearest_neighbor, var_col, TRAIN_PERIOD,
                                       FORWARD_TEST_PERIOD)
        have_visited = have_visited + nearest_neighbor  # update the have visited list
        step += 1

        pandas_temp_df = buildDataFrame(next_step_coordinate, var_col, score_list[0], step)
        pandas_df = pandas_df.append(pandas_temp_df, ignore_index=True)

    local_optimum = next_step_coordinate
    return pandas_df


def processor():
    try:
        pandas_df = variable_selection(VARIABLE_COLS, TRAIN_PERIOD, FORWARD_TEST_PERIOD)
        spark_df = spark.createDataFrame(pandas_df)
        path = "hdfs://datalakeprod//uat/DataScience/revenue_and_risk/var_selection_test"
        writeToJsonRenameOldFile(spark_df, path, 'govt15_fha')
    except Exception as e:
        try:
            app_id = spark.conf.get("spark.app.id")
            pd_df = pd.DataFrame({"msg": [str(e)], "app_id": [str(app_id)]})
            df = spark.createDataFrame(pd_df)
            writeToJsonRenameOldFile(df, path, "error___" + app_id)
        except:
            pass
        raise
    finally:
        try:
            spark.sql(f"drop table lock2close_conformed.{train_tb}")
            spark.sql(f"drop table lock2close_conformed.{test_tb}")
            subprocess.call(["hdfs", "dfs", "-rm", "-r", path_train])
            subprocess.call(["hdfs", "dfs", "-rm", "-r", path_test])
        except:
            pass