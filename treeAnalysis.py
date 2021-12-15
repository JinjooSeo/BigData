import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from hipe4ml.analysis_utils import train_test_generator
import matplotlib.pyplot as plt

from hipe4ml import analysis_utils, plot_utils
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler


promptH = TreeHandler('data/prompt.root','treeMLDplus')
dataH = TreeHandler('data/data.root','treeMLDplus')
bkgH = dataH.get_subset('inv_mass < 1.82 or 1.92 < inv_mass < 2.00', size=promptH.get_n_cand()*3)

train_test_data = train_test_generator([promptH, bkgH], [1,0], test_size=0.5, random_state=42)

vars_to_draw = promptH.get_var_names()

leg_labels = ['background', 'signal']

plot_utils.plot_distr([bkgH, promptH], vars_to_draw, bins=100, labels=leg_labels, log=True, density=True, figsize=(12, 7), alpha=0.3, grid=False)
plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
plt.savefig('sigvsbkg')

plot_corr = plot_utils.plot_corr([bkgH, promptH], vars_to_draw, leg_labels)
plot_corr[0].savefig('bkgCorr')
plot_corr[1].savefig('sigCorr')


features_for_train = vars_to_draw.copy()
features_for_train.remove('inv_mass')
features_for_train.remove('pt_cand')

# TRAINING AND TESTING

INPUT_MODEL = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
model_hdl = ModelHandler(INPUT_MODEL, features_for_train)

# hyperparams optimization
hyper_pars_ranges = {'n_estimators': (200, 1000), 'max_depth': (2, 4), 'learning_rate': (0.01, 0.1)}
model_hdl.optimize_params_bayes(train_test_data, hyper_pars_ranges, 'roc_auc', nfold=5, init_points=5, n_iter=5, njobs=-1)

# train and test the model with the updated hyperparameters
model_hdl.train_test_model(train_test_data)
y_pred_train = model_hdl.predict(train_test_data[0], False)
y_pred_test = model_hdl.predict(train_test_data[2], False)

# Calculate the BDT efficiency as a function of the BDT score
plt.rcParams["figure.figsize"] = (10, 7)

ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, False, leg_labels, True, density=True)
plt.savefig(f'ml_output_{pT}')
roc_train_test_fig = plot_utils.plot_roc_train_test(train_test_data[3], y_pred_test, train_test_data[1], y_pred_train, None, leg_labels)
plt.savefig(f'roc_train_test_{pT}')

efficiency, threshoold = analysis_utils.bdt_efficiency_array(train_test_data[3], y_pred_test, n_points=10)
precision_recall = plot_utils.plot_precision_recall(train_test_data[3], y_pred_test)
plt.savefig('precisionRecall')
BDT_eff = plot_utils.plot_bdt_eff(threshoold, efficiency)
plt.savefig('BDTeff')

# Features importance
feature_importance = plot_utils.plot_feature_imp(train_test_data[2], train_test_data[3], model_hdl)
feature_importance[0].savefig('feature1')
feature_importance[1].savefig('feature2')


# Apply model
dataH.apply_model_handler(model_hdl, False)
dataH.write_df_to_root_files('results_data')
selected_data_hndl = dataH.get_subset('model_output>0.98')

promptH.apply_model_handler(model_hdl, False)
selected_mc_hndl = promptH
selected_mc_hndl.write_df_to_root_files('results_prompt')


labels_list = ["after selection","before selection"]
colors_list = ['orangered', 'cornflowerblue']
plot_utils.plot_distr([selected_data_hndl, dataH], column='inv_mass', bins=200, labels=labels_list, colors=colors_list, density=True,fill=True, histtype='step', alpha=0.5)
ax = plt.gca()
ax.set_xlabel(r'm(K$^-\pi^+\pi^+$) (GeV/$c^2$)')
ax.margins(x=0)
ax.xaxis.set_label_coords(0.9, -0.075)
plt.savefig('Inv_Mass')
#plt.show()