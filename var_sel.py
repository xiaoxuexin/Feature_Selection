import warnings
import subprocess
import uuid
from pyspark.sql import SparkSession

warnings.filterwarnings('ignore')
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

from l2cmodel.l2cTrainer import *
from l2cmodel.l2cTrainer import GENERAL_COLS, writeToJsonRenameOldFile
import random
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType
import pyspark.sql.functions as f
from rocktrain.ptrEvaluator import PTREval
from l2cmodel.util import getData, getGlobals, globalAttr, setGlobals, getSaveResult

VARIABLE_COLS = ['banker_performance', 'borrowerage', 'creditscore', 'dti',
                  'haspropertyinspectwaiver',
                  'highltv', 'interestrate', 'iscondopropertytype',   'ispudpropertytype',
                  'issingleborrower', 'issinglepropertytype', 'istenureover10',
                  'ltv', 'nocashamt', 'nothasappraisalvalue',
                  'requiredpts', 'retentionclient', 'selfemployflg', 'status_age_businessdays',
                  'status_age_zeronotsuspense','hassuspense', 'appraisalvalue',
                  'isqlms',"ispurchaseloan",'isforward','isrelo','cashout',

                  # 'status_split___approved_appraised', 'status_split___prefolder', 'status_split___signoff',
                  # 'status_split___suspense_ana', 'status_split___suspense_collateral', 'status_split___suspense_other',
                  # 'statusgroup___folder', 'statusgroup___prefolder', 'statusgroup___signoff', 'statusgroup___suspense',

                  'statusgroup_split35___approved_appraised', 'statusgroup_split35___folder', #'statusgroup___approved',
                  'statusgroup_split35___prefolder', 'statusgroup_split35___signoff', 'statusgroup_split35___suspense',
                  'suspensereason_ana', 'suspensereason_ana_zero', 'suspensereason_collateral',
                  'suspensereason_collateral_zero', 'suspensereason_title_zero','suspensereason_income',
                  'tenure', 'timesinsuspense',
                  'log_income', 'log_totinsuramt',
                  'loantype_va','streamlineflag','loantype___fhafixed',
                  #'clean35','status36', 'status34'
                  ]


# EPI_PRE_VARIABLE_COLS = ["banker_performance", "borrowerage",
#                  "cashoutindicator", "creditscore", "dti", "ltv", "fulldoc",
# #             "haspropertyinspectwaiver", "targetprofit","statusgroup_split35___approved_appraised", "noappraisalrequired",
#                 "highltv", "interestrate", "iscondopropertytype",
#                 "ispudpropertytype", "ispurchaseloan", "isqlms", "isrelo",
#                 "issingleborrower", "issinglepropertytype", "istenureover10", "loanamount",# "loantype___fhafixed",
#                 "lock_age",  "requiredpts", "retentionclient", "selfemployflg",
#                 "status_age_businessdays","tenure", "nothasappraisalvalue",'appraisalvalue',
#                 # "suspensereason_ana", "suspensereason_collateral",  "suspensereason_income","timesinsuspense",
# #                 'weightedzeropointrate','suspensereason_collateral_zero','daysinsuspense', 'suspensereason_ana_zero',
#                 "log_income", "log_totinsuramt",
#                 "tottaxamt", "streamlineflag",
#                  'libor1mo',# 'hasjointprimaryopencount',
#                 'currentfannie30yryield']

NEW_FEATURES_FORCED = []
ALL_NEW_FEATURES = []#'ARM_GRAM_CONV30_Ratio', 'ARM_GARM_CONV30_Spread']
# GENERAL_COLS = ["observation_dt", "loannumber", "hedgestatus", "submodel", "loanlevel_score"]

SUBMODEL_NAME = "GOVT_15YR_1"
OTHERCOND = "hedgegroup = 'GOVT_15YR' and hassuspense = 0 and nothasappraisalvalue = 0 and retentionclient = 1"
# OTHERCOND = "hedgegroup = 'GOVT_15YR' and hassuspense = 0 and nothasappraisalvalue = 1"
# OTHERCOND = "hedgegroup = 'GOVT_15YR' and hassuspense = 1"
# OTHERCOND = "hedgegroup = 'GOVT_15YR' and submodel = 'GOVT_15YR_VA'"

# OTHERCOND = "submodel = 'ARM' and hedgegroup in ('GARM_30YR', 'ARM_30YR') and statusgroup_split35___prefolder = 1"
# OTHERCOND = "submodel = 'ARM' and hedgegroup in ('GARM_30YR', 'ARM_30YR') and statusgroup_split35___folder = 1"
# OTHERCOND = "submodel = 'ARM' and hedgegroup in ('GARM_30YR', 'ARM_30YR') and statusgroup_split35___suspense = 1"
# OTHERCOND = "submodel = 'ARM' and hedgegroup in ('GARM_30YR', 'ARM_30YR') and statusgroup_split35___signoff = 1"
# OTHERCOND = "submodel = 'ARM' and hedgegroup in ('GARM_30YR', 'ARM_30YR') and status = '35' AND statusgroup_split35___approved_appraised = 0 and noappraisalrequired = 0"
# OTHERCOND =  "submodel = 'ARM' and hedgegroup in ('GARM_30YR', 'ARM_30YR') and status = '35' AND (statusgroup_split35___approved_appraised = 1 or noappraisalrequired = 1)"

TOTAL_COLS = VARIABLE_COLS + GENERAL_COLS
BACK_TEST_PERIOD = None
TRAIN_PERIOD = '2018-01-01', '2019-08-31'
FORWARD_TEST_PERIOD = '2019-01-01', '2020-04-10'
_, _, _, LABEL_COL = globalAttr()
VERSION = '_v0.1_var_selection_xin'
hive_db="lock2close_conformed"
hive_tb="l2c"

spark.catalog.clearCache()
setGlobals(TRAIN_PERIOD=TRAIN_PERIOD, FORWARD_TEST_PERIOD=FORWARD_TEST_PERIOD,
           BACK_TEST_PERIOD=BACK_TEST_PERIOD)

# the following lines are used for uploading data file from data lake which contain new features

# new_feature_df = spark.read.format("csv").option("header", "true").load("hdfs:/prod/RiskandRevenue/SandBox/Sample_Xin/ARM_CONV30_FINAL_02_12.csv")
# new_feature_df=spark.read.json("hdfs:/prod/RiskandRevenue/SandBox/Sample_Xin/benefit_partner.json")
# new_feature_df = strToDoubleSchema(new_feature_df,ALL_NEW_FEATURES)
# print(new_feature_df)
# print(new_feature_df.count())
# print(new_feature_df.show(5))
# change the format of observation_dt when join the feature dataframe and training data
# lists = new_feature_df.select(f.collect_list('observation_dt')).first()
# func = udf(lambda x: datetime.strptime(x, '%m-%d-%Y'), DateType())
# new_feature_df = new_feature_df.withColumn('observation_dt', func(col('observation_dt')))

"""
TRAIN_DF = new_feature_df.filter(
    (new_feature_df['observation_dt'] >= TRAIN_PERIOD[0]) & (new_feature_df['observation_dt'] <= TRAIN_PERIOD[1]))
print(TRAIN_DF.count())
print('\n')

TEST_DF = new_feature_df.filter(
    (new_feature_df['observation_dt'] >= FORWARD_TEST_PERIOD[0]) & (new_feature_df['observation_dt'] <= FORWARD_TEST_PERIOD[1]))
print(TEST_DF.count())
"""

uuid_4 = str(uuid.uuid4().int)
train_tb = "var_sel_" + SUBMODEL_NAME + uuid_4 + "_train"
test_tb = "var_sel_" + SUBMODEL_NAME + uuid_4 + "_test"
path_train = f"hdfs://datalakeprod/prod/RiskandRevenue/model_hedge/temp/{train_tb}"
path_test = f"hdfs://datalakeprod/prod/RiskandRevenue/model_hedge/temp/{test_tb}"
# TRAIN_DF = spark.sql(f"""
#             SELECT {','.join(TOTAL_COLS)}
#             FROM {hive_db}.{hive_tb}
#             WHERE observation_dt BETWEEN '{TRAIN_PERIOD[0]}' AND '{TRAIN_PERIOD[1]}'
#             AND {where_clause}
#         """).dropna(subset=VARIABLE_COLS)

TRAIN_DF = getData(TOTAL_COLS, SUBMODEL_NAME, TRAIN_PERIOD, otherConditions = OTHERCOND).dropna(subset=VARIABLE_COLS)
# TRAIN_DF.repartition(1).write.saveAsTable(
#             f"{hive_db}.{train_tb}", partitionBy="observation_dt", mode="overwrite", format="orc", path=path_train
#         )

# following for new feature table joining

# TRAIN_DF = TRAIN_DF.join(TRAIN_DF_new[ALL_NEW_FEATURES +['loannumber', 'observation_dt']], ['loannumber', 'observation_dt'])
print('TRAIN_DF num', TRAIN_DF.count())

# TEST_DF = spark.sql(f"""
#             SELECT {','.join(TOTAL_COLS)}
#             FROM {hive_db}.{hive_tb}
#             WHERE observation_dt BETWEEN '{FORWARD_TEST_PERIOD[0]}' AND '{FORWARD_TEST_PERIOD[1]}'
#             AND {where_clause}
#         """).dropna(subset=VARIABLE_COLS)
TEST_DF = getData(TOTAL_COLS, SUBMODEL_NAME, FORWARD_TEST_PERIOD, otherConditions = OTHERCOND).dropna(subset=VARIABLE_COLS)

# following for new feature table joining

# TEST_DF = TEST_DF.join(TEST_DF_new[ALL_NEW_FEATURES +['loannumber', 'observation_dt']], ['loannumber', 'observation_dt'])
print('TEST_DF num', TEST_DF.count())


TRAIN_DF.repartition(1).write.saveAsTable(f'lock2close_conformed.{train_tb}', mode='overwrite', path=path_train,
                           format='orc', partitionBy="observation_dt")
TEST_DF.repartition(1).write.saveAsTable(f'lock2close_conformed.{test_tb}', mode='overwrite', path=path_test,
                            format='orc', partitionBy="observation_dt")


# try:
#     spark.catalog.clearCache()
#     spark.catalog.refreshTable("lock2close_conformed.first_train_xin")
#     spark.catalog.refreshTable("lock2close_conformed.first_test_xin")
#     print("DEBUG TABLE REFRESH")
# except:
#     pass


def get_initial_coor(num, dimension):
    # index of the randomly selected starting point
    initial_index = random.sample(range(0, dimension), int(np.ceil((dimension - num) / 3)))
    # coordinate of the starting point
    initial_coordinate = np.zeros(dimension)
    for i in initial_index:
        initial_coordinate[i] = 1
    return initial_coordinate


def get_neighbor(new_feature, coordinate, dimension):
    new_len = len(new_feature)
    # nearest neighbors of the point using coordinate
    nearest_neighbor = [coordinate.copy() for i in range(dimension - new_len)]
    for i in range(new_len, dimension):
        nearest_neighbor[i - new_len][i] = 1 - nearest_neighbor[i - new_len][i]
    return nearest_neighbor


def not_vis_neighbor(new_feature, dimension, next_step_coordinate, have_visited):
    new_len = len(new_feature)
    nearest_neighbor = []
    for i in range(new_len, dimension):
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
    # TRAIN_DF_table = spark.table(f"lock2close_conformed.{train_tb}")
    train_query = f"""select {','.join(getdata_cols)} from lock2close_conformed.{train_tb}
                where observation_dt between '{training_pred[0]}' and '{training_pred[1]}'"""

    TRAIN_DF_temp = spark.sql(train_query)
    print('train_data', TRAIN_DF_temp.count(), TRAIN_DF_temp.head(5))
    # TEST_DF_table = spark.table(f"lock2close_conformed.{test_tb}")
    test_query = f"""select {','.join(getdata_cols)} from lock2close_conformed.{test_tb}
                where observation_dt between '{test_pred[0]}' and '{test_pred[1]}'"""
    TEST_DF_temp = spark.sql(test_query)
    print('test_data', TEST_DF_temp.count(), TEST_DF_temp.head(5))

    tr = TrainerModel(data=TRAIN_DF_temp, labelCol=LABEL_COL)

    tr.vectorAssembler(inputCols=var_col)
    tr.logisticRegression(customEvaluatorClass=PTREval)
    transformedData = tr.buildPipeline()
    print('after pipeline', transformedData)
    # trainEval = tr.evaluationResult
    pred_transform, pred_score = tr.predict(TEST_DF_temp)
    return pred_score[0]['score']


def start_score_list(coordinate, var_col, TRAIN_PERIOD, FORWARD_TEST_PERIOD):
    col = coordinate2attribute(coordinate, var_col)
    print('column is ', col)
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


def variable_selection(new_features, var_col, TRAIN_PERIOD, FORWARD_TEST_PERIOD):
    step = 0
    dimension = len(var_col)  # number of variables in our consideration
    new_num = len(new_features)
    if dimension <= 1:
        return 0
    min_score_dic = {}
    # randomly initialize
    initial_coordinate = get_initial_coor(new_num, dimension)
    # prod_var_fha = ["banker_performance", "borrowerage", "cashout", "haspropertyinspectwaiver", "creditscore",
    #                     "hassuspense", "highltv" ,  "interestrate", "iscondopropertytype" , "isforward" ,
    #                     "ispurchaseloan" ,"isqlms","isrelo" ,"issingleborrower", "issinglepropertytype" ,
    #                     "loanamount", "lock_age" ,"log_income", "nocashamt" , "requiredpts" ,
    #                     "status_age_businessdays" , "status_split___suspense_other",
    #                     "statusgroup___approved", "status_split___signoff","statusgroup_split35___suspense",
    #                     "suspensereason_ana",]
    # prod_var_va = ["borrowerage","creditscore","interestrate","ispurchaseloan","issingleborrower" ,"issinglepropertytype",
    #                "istenureover10", "loanamount" ,"lock_age", "log_totinsuramt","ltv" ,"nocashamt" ,  "nothasappraisalvalue" ,
    #                "requiredpts" ,"selfemployflg" , "status_age_businessdays" , "statusgroup___approved",
    #                "statusgroup_split35___approved_appraised", 'statusgroup_split35___folder',
    #                'statusgroup_split35___prefolder', 'statusgroup_split35___suspense', "suspensereason_collateral",]
    # prod_var_streamline = ["borrowerage", "dti","highltv" , 'interestrate', 'iscondopropertytype' , 'issingleborrower' ,
    #                'issinglepropertytype', 'loantype___fhafixed', 'requiredpts' , 'retentionclient',
    #                'selfemployflg' , 'status_age_businessdays' , "statusgroup___approved",
    #                "statusgroup_split35___approved_appraised", 'statusgroup_split35___signoff',
    #                'suspensereason_collateral',]
    # initial_coordinate = attribute2coordinate(prod_var, var_col)
    # assign the initial point
    # assigned_ini_attr = []
    # initial_coordinate = attribute2coordinate(assigned_ini_attr, var_col)
    # set the new feature coordinate as 1
    # new_coor = np.ones(new_num)
    # initial_coordinate[0:new_num] = new_coor
    print('\n')
    print('initial coor ', initial_coordinate)
    print('\n')
    score_list = start_score_list(initial_coordinate, var_col, TRAIN_PERIOD, FORWARD_TEST_PERIOD)
    have_visited = [initial_coordinate]  # points have been visited

    # nearest neighbors of the initial point
    nearest_neighbor = get_neighbor(new_features, initial_coordinate, dimension)

    score_list = append_score_list(score_list, nearest_neighbor, var_col, TRAIN_PERIOD,
                                   FORWARD_TEST_PERIOD)

    have_visited = have_visited + nearest_neighbor  # append the nearest neighbor to have visited list
    next_step_coordinate = initial_coordinate

    pandas_df = buildDataFrame(next_step_coordinate, var_col, score_list[0], 0)

    while np.argmin(score_list) != 0: # (3 step ago result - result) < 0.001
        min_score_dic[step] = score_list[0]
        if step > 2:
            if (min_score_dic[step - 3] - min_score_dic[step]) < 0.0005:
                return pandas_df

        next_step_coordinate = nearest_neighbor[np.argmin(score_list) - 1]  # first one in score list is current point
        nearest_neighbor = not_vis_neighbor(new_features, dimension, next_step_coordinate, have_visited)
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
        pandas_df = variable_selection(NEW_FEATURES_FORCED, ALL_NEW_FEATURES + VARIABLE_COLS, TRAIN_PERIOD, FORWARD_TEST_PERIOD)
        spark_df = spark.createDataFrame(pandas_df)
        path = "hdfs://datalakeprod//uat/DataScience/revenue_and_risk/var_selection_test"
        writeToJsonRenameOldFile(spark_df, path, 'govt15_1_2_0630')
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