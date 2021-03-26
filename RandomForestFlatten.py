import numpy as np
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_curve,roc_auc_score,f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import itertools
np.random.seed(0)

def load_datasets(dataset):
    newX = np.load(file=dataset + "_Combine_Matrix.npy")
    label = np.load(file=dataset + "_Label.npy")

    print('Combine (sizes * features * time points): ', newX.shape)
    print('Label size: ', len(label))

    return newX, label


data_x, data_y = load_datasets('knat')
size=data_x.shape[0]   #number of examples
data_x=np.reshape(data_x,(data_x.shape[0],-1))
kf = StratifiedKFold(n_splits=5)
mean_fpr = np.linspace(0, 1, 100)
params_dict = {'n_estimators': [30,100],
              'max_depth': [4,8],
                 'min_samples_leaf': [2,4]}
outer_f1s = [];outer_aucs=[];outer_tprs = [] 
inner_f1s = [];inner_params=[]


for i in range(10):
    for train, test in kf.split(data_x, data_y):
        #inner 5-flod cv
        clf = GridSearchCV(RandomForestClassifier(), param_grid=params_dict,scoring='f1',cv=5)
        clf.fit(data_x[train], data_y[train])
        inner_f1s.append(clf.best_score_)

        #use best estimator attained from inner cv
        y_pred_prob = clf.predict(data_x[test]) 
        y_pred_label = (y_pred_prob>[0.5]).astype(int)
        f1=f1_score(y_true=data_y[test], y_pred=y_pred_prob)
        fpr, tpr, thresholds = roc_curve(data_y[test], y_pred_prob)
        auc = roc_auc_score(data_y[test], y_pred_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr) 
        interp_tpr[0] = 0.0
        outer_tprs.append(interp_tpr) 
        outer_f1s.append(f1)
        outer_aucs.append(auc)

mean_f1=np.mean(outer_f1s)
mean_auc=np.mean(outer_aucs)
mean_f1_train=np.mean(inner_f1s)
print("mean_f1_train: ",mean_f1_train)
print("mean_f1: ",mean_f1)
print("mean_auc: ",mean_auc)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
mean_tpr = np.mean(outer_tprs, axis=0)
mean_tpr[-1] = 1.0
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f)' % (mean_auc),
        lw=2, alpha=.8)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
    title="ROC figure")
ax.legend(loc="lower right")
plt.show()

    