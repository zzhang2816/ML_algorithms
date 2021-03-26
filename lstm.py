#version for tuning the parameter
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import plot_roc_curve
import tensorflow 
np.random.seed(10)
tensorflow.random.set_seed(5)

def load_datasets(dataset):
    newX = np.load(file=dataset + "_Combine_Matrix.npy")
    label = np.load(file=dataset + "_Label.npy")

    print('Combine (sizes * features * time points): ', newX.shape)
    print('Label size: ', len(label))

    return newX, label


def lstm_model(num_time_step, dim, num_features):
    # Define the input layer and specify the shape
    X = Input(shape=(num_time_step, num_features))
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    Y = LSTM(units=dim, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    Y = Dropout(rate=0.5)(Y)
    # Propagate X through a Dense layer with 5 units
    Y = Dense(units=1)(Y)
    # Add a sigmoid activation
    Y = Activation('sigmoid')(Y)
    model = Model(inputs=X, outputs=Y)
    return model


data_x, data_y = load_datasets('knat')
data_x = data_x.transpose(0, 2, 1)  # X (sizes , time points , features)
n_a = 6  # number of dimensions for the hidden state of each LSTM cell.
n_features = data_x.shape[2]  # number of features  
size=data_x.shape[0]   #number of examples
n_t= data_x.shape[1]     #time step

model = lstm_model(n_t, n_a, n_features)
opt = SGD(learning_rate=0.01, momentum=0.1)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
kf = KFold(n_splits=5,shuffle=True,random_state=10) # 5 fold cross-validation
callback = EarlyStopping(monitor='loss', patience=3) #early stop
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)  #reduce learning rate
f1s=[] #store the F1 scores
#tune only f1s_train=[]  
aucs=[] #store the auc scores

tprs = [] #for plotting the roc curve
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
n_fold = 1

for train, test in kf.split(data_x, data_y):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {n_fold} ...')
    model.fit(data_x[train],data_y[train], epochs=100, 
    callbacks=[callback,reduce_lr])

    y_pred_prob = model.predict(data_x[test]) #make prediction
    #tune only y_train_prob = model.predict(data_x[train]) 
    y_pred_label = y_pred_prob.argmax(axis=-1)
    #tune only y_train_label = y_train_prob.argmax(axis=-1) 
    f1=f1_score(y_true=data_y[test], y_pred=y_pred_label,average='micro')
    #tune only f1_train=f1_score(y_true=data_y[train], y_pred=y_train_label,average='micro')  
    fpr, tpr, thresholds = roc_curve(data_y[test], y_pred_prob)
    auc = roc_auc_score(data_y[test], y_pred_prob,average='micro')

    interp_tpr = np.interp(mean_fpr, fpr, tpr) 
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    f1s.append(f1)
    #tune only f1s_train.append(f1_train) 
    aucs.append(auc)
    K.clear_session() #clear the graph
    n_fold = n_fold + 1  # Increase fold number

mean_f1=np.mean(f1s)
mean_auc=np.mean(aucs)
print("aucs: ",aucs)
print("mean_f1: ",mean_f1)

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

