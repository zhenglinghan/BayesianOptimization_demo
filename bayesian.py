
import numpy as np
#import pandas as pd
import lightgbm as lgb
from sklearn.metrics import   f1_score
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
#################################

def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
        return 'f1', f1_score(y_true, y_hat,average='macro'), True

########################################



def bayes_parameter_opt_lgb(lgb_train, lgb_val,val_data, val_lbl,  init_round=15, opt_round=25,  random_seed=61, num_boostround=3200):
    def lgb_eval(num_leaves,max_depth,max_bin,  min_data_in_leaf, bagging_freq, bagging_fraction,feature_fraction, reg_alpha, reg_lambda, min_sum_hessian_in_leaf, min_split_gain, learning_rate):
        params = {'application':'binary','verbosity' : -2, 'boosting':'gbdt',  'seed':123 , 'metric':'None'}
        params["num_leaves"] = int(round(num_leaves))
        params['max_depth'] = int(round(max_depth))
        params['max_bin'] = int(round(max_bin))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['bagging_freq'] = int(round(bagging_freq))
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        params['min_split_gain'] = min_split_gain
        params['learning_rate'] = learning_rate
        model = lgb.train(params, lgb_train, valid_sets=lgb_val, num_boost_round=3200, early_stopping_rounds=64,verbose_eval=False, feval=lgb_f1_score) 
        y_pred = model.predict(val_data)
        m=(y_pred*2.0).astype(int)
        chk=f1_score( val_lbl.astype(int), m ,average='macro')
        ###Bayesian优化的目标是chk的数值
        return chk
    # range 
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (13, 27),
                                            'max_depth': (4, 8.99),
                                            'max_bin': (13, 168),
                                            'min_data_in_leaf':(8,28),
                                            'bagging_freq': (3, 9),
                                            'bagging_fraction': (0.5, 1),
                                            'feature_fraction': (0.3, 0.9),
                                            'reg_alpha': (0, 5.1),
                                            'reg_lambda': (0, 3.2),
                                            'min_sum_hessian_in_leaf':(0.001, 16.7),
                                            'min_split_gain': (0.001, 0.8),
                                            'learning_rate': (0.01, 0.07) })
    ##保存中间每步的结果
    lgbBO.subscribe(Events.OPTMIZATION_STEP, logger)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    print(lgbBO.max)
    # return best parameters
    return lgbBO.res['max']['max_params']


##############################################

npy_trn=np.load('trn_km.npy')

total_trn_num = len(npy_trn)
trn_list = np.arange(total_trn_num)
val_num = int( total_trn_num *0.1)
 
np.random.shuffle(trn_list)
 
val_data = npy_trn[trn_list[:val_num]]
trn_data = npy_trn[trn_list[val_num:]]
del npy_trn

val_lbl  = val_data[:, -1]
val_data = val_data[:, :-1]
trn_lbl  = trn_data[:, -1]
trn_data = trn_data[:, :-1]
print(len(val_lbl),len(trn_lbl ) )
 

lgb_train = lgb.Dataset(trn_data ,trn_lbl, free_raw_data=False)
lgb_val = lgb.Dataset(val_data ,val_lbl, reference=lgb_train,free_raw_data=False)

logger = JSONLogger(path="/mnt/oss/df2019conrt/logs.json")



opt_params = bayes_parameter_opt_lgb(lgb_train,  lgb_val,val_data, val_lbl, init_round=16, opt_round=1024,  random_seed=61, num_boostround=3200)

print('*********************************')
print('*********************************')
print('result:')
print(opt_params)
