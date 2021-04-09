from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import f1_score,plot_roc_curve
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

def load_datasets(dataset):
    newX = np.load(file=dataset + "_Combine_Matrix.npy")
    label = np.load(file=dataset + "_Label.npy")

    print('Combine (sizes * features * time points): ', newX.shape)
    print('Label size: ', len(label))

    return newX, label

MODEL = "SVM" # BernoulliNB, MultinomialNB, LogisticRegression, RandomForest, XGBoost
MODE = "Flatten" # Single
# preprocessing
data_x, data_y = load_datasets('knat')
if MODE == "Flatten":
    data_x=np.reshape(data_x,(data_x.shape[0],-1)) #flatten the time-series
if MODE == "Single":
    size=data_x.shape[0]   #number of examples
    n_t=data_x.shape[2]
    data_x=np.delete(data_x,range(n_t-1),2)
    data_x=np.reshape(data_x,(size,-1))


# store the cross validation result
best_params=[]
train_f1s = []
test_f1s = []
aucs=[]

# serve for plotting averge roc curve purpose
mean_fpr = np.linspace(0, 1, 100)
tprs = [] 
fig, ax = plt.subplots()

# set trainning params
# there is no Closed solution for RandomForest and XGBoost
# so we run the algorithm 10 times to calculate the averge
times = 1
if MODEL == "SVM":
    from sklearn.svm import SVC
    params_dict = {"C": [0.1,1, 10, 100], "gamma": [1,0.1,0.01],'kernel': ['rbf','linear']}
elif MODEL == "BernoulliNB": #Bernoulli Naive Bayes
    from sklearn.naive_bayes import BernoulliNB
    params_dict={"alpha": [0.1,1,3]}
elif MODEL == "MultinomialNB": #Multinomial Naive Bayes
    from sklearn.naive_bayes import MultinomialNB
    params_dict={"alpha": [0.1,1,3]}
elif MODEL == "LogisticRegression":
    from sklearn.linear_model import LogisticRegression
    params_dict = {"C": [0.1,1, 10, 100],"penalty":['l2']}
elif MODEL == "RandomForest":
    times = 10
    from sklearn.ensemble import RandomForestClassifier
    params_dict = {'n_estimators': [30,100],
              'max_depth': [4,8],
                 'min_samples_leaf': [2,4]}
elif MODEL == "XGBoost":
    from xgboost.sklearn import XGBClassifier
    times = 10
    params_dict = { "gamma":[0,0.2],
                "lambda":[0,1],
                "max_depth":[3,6],
                "min_child_weight":[1],
                "learning_rate": [0.1],
                "n_estimators":[10]
                }

# nested cross-validation
n_fold=1
kf = StratifiedKFold(n_splits=5)

for i in range(times): 
    # outer cv
    for train_idx, test_idx in kf.split(data_x, data_y):
        print('------------------------------------------------------------------------')
        print(f'Training for fold {n_fold} ...')
        
        # inner 5-fold cv
        if MODEL == "SVM":
            clf = GridSearchCV(SVC(), param_grid=params_dict,scoring='f1',cv=5) # use f1 as scoring
        elif MODEL == "BernoulliNB":
            clf = GridSearchCV(BernoulliNB(), param_grid=params_dict,scoring='f1',cv=5)
        elif MODEL == "MultinomialNB":   
            clf = GridSearchCV(MultinomialNB(), param_grid=params_dict,scoring='f1',cv=5)
        elif MODEL == "LogisticRegression":
            clf = GridSearchCV(LogisticRegression(max_iter=300), param_grid=params_dict,scoring='f1',cv=5)
        elif MODEL =="RandomForest":
            clf = GridSearchCV(RandomForestClassifier(), param_grid=params_dict,scoring='f1',cv=5)
        elif MODEL == "XGBoost":
            clf = GridSearchCV(XGBClassifier(), param_grid=params_dict,scoring='f1',cv=5,verbose=1)

        clf.fit(data_x[train_idx], data_y[train_idx])
        train_f1s.append(clf.best_score_)
        best_params.append(clf.best_params_)

        # use best estimator attained to predict
        y_pred = clf.predict(data_x[test_idx]) 
        if MODEL == "SVM":
            y_pred_label = y_pred
        else:
            y_pred_label = (y_pred>[0.5]).astype(int)
        
        test_f1=f1_score(y_true=data_y[test_idx], y_pred=y_pred)
        test_f1s.append(test_f1)

        # plot roc curve for each fold
        viz = plot_roc_curve(clf, data_x[test_idx], data_y[test_idx],
                            name='ROC fold {}'.format(n_fold),
                            alpha=0.3, lw=1, ax=ax)
        aucs.append(viz.roc_auc)
        
        # save variables for plotting averge roc curve
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr) 
        n_fold+=1

# output the averge result
mean_auc=np.mean(aucs)
mean_f1_train=np.mean(train_f1s)
mean_f1_test=np.mean(test_f1s)
print("mean_auc: ",mean_auc)
print("mean_f1_train: ",mean_f1_train)
print("mean_f1_test: ",mean_f1_test)

# plot averge roc curve
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f)' % (mean_auc),
        lw=2, alpha=.8)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
    title="ROC figure")
ax.legend(loc="lower right")
plt.show()




