#!/usr/bin/env python
# coding: utf-8

# ### Identifying patients with a risk of cardio disease 

# 

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score


# In[5]:


import shap


# In[6]:


from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB


# In[7]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier


# In[10]:


from sklearn.metrics import precision_score, recall_score, f1_score


# In[11]:


from sklearn.metrics import confusion_matrix


# In[345]:


data=pd.read_csv('data1.csv')


# In[346]:


data.info()


# In[347]:


data.describe()


# In[348]:


data.head(10)


# In[349]:


data.rename(columns={'age_year':'age','ap_hi':'systolic_blood_pressure','ap_lo':'diastolic_blood_pressure',
                     'gluc':'glucose_level','smoke':'smoker', 'alco':'alcohol_consump',
                     'active':'physical_activity', 'cardio':'heart_disease'}, inplace=True)


# In[350]:


data.head()


# In[351]:


data = data.drop('age_days',1)


# In[352]:


data = data.drop('id',1)


# In[353]:


data.head()


# In[272]:


#Transform variable age to integer without the decimal part

data['age']=data['age'].apply(lambda x: round(x,1))


# In[273]:


#Transform variable height from sentimeter to foot

#data['height']=data['height'].apply(lambda x: round(x/30.48,1))


# In[274]:


#Transform variable weight from kg to lbs

#data['weight']= data['weight'].apply(lambda x: round(x*2.205,1))


# In[354]:


#Creating new column for Body Index
data['BMI']=data['weight']/(data['height']/100)**2


# In[356]:


#For BMI below 25 we will arrange 0 - healthy range , for BMI higher 25 - we will arrange 1 - overweight
data['BMI']=data.BMI.apply(lambda x: 1 if x>=25 else 0)


# In[357]:


data.head(3)


# In[363]:


data= data.drop('weight',1)


# In[364]:


data= data.drop('height',1)


# In[367]:


data=data[['age','gender','BMI','systolic_blood_pressure','diastolic_blood_pressure','cholesterol','glucose_level','smoker','alcohol_consump','physical_activity','heart_disease']]


# In[368]:


data.head()


# In[369]:


target_balance = data.heart_disease.value_counts() 
print(target_balance)

target_balance.plot(kind='bar')


# In[370]:


X = data.loc[:,'age':'physical_activity']
y = data['heart_disease']


# In[371]:


X.head()


# In[372]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=45)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# In[373]:


logit_1= LogisticRegression(C=1000)
logit_1.fit(X_train,y_train)


# In[374]:


logit_1.score(X_test,y_test)


# In[375]:


print("The score for logistic regression is")
print("Training: {:6.2f}%".format(100*logit_1.score(X_train, y_train)))
print("Test set: {:6.2f}%".format(100*logit_1.score(X_test, y_test)))


# In[376]:


print('Logistic regression validation metrics:  \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (
         precision_score(y_val, logit_1.predict(X_test)), 
         recall_score(y_val, logit_1.predict(X_test)),
         f1_score(y_val, logit_1.predict(X_test))
        )
     )


# Sklearn logistic regression we should make sure to scale our features prior to fitting, since regularzation is used by default 

# In[377]:


std_scale = StandardScaler()


X_train_scaled = std_scale.fit_transform(X_train)
X_val_scaled= std_scale.fit_transform(X_val)
X_test_scaled= std_scale.fit_transform(X_test)

logit_2 = LogisticRegression()
logit_2.fit(X_train_scaled, y_train)

y_predict = logit_2.predict(X_train_scaled) 
logit_2.score(X_train_scaled, y_train)


# In[378]:


print("The score for logistic regression 2 is")
print("Training: {:6.2f}%".format(100*logit_2.score(X_train_scaled, y_train)))
print("Val set: {:6.2f}%".format(100*logit_2.score(X_test_scaled, y_test)))


# In[379]:


print('Logistic regression 2 validation metrics:  \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (
         precision_score(y_val, logit_2.predict(X_val_scaled)), 
         recall_score(y_val, logit_2.predict(X_val_scaled)),
         f1_score(y_val, logit_2.predict(X_val_scaled))
        )
     )


# In[380]:


from sklearn.metrics import roc_auc_score, roc_curve


# In[381]:


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, logit_2.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr, tpr)

x = np.linspace(0,1, 100000)
plt.plot(x, x, linestyle='--')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Logistic Regression'])


# In[382]:


logit_1_confusion = confusion_matrix(y_test, logit_2.predict(X_test_scaled))
logit_1_confusion


# In[383]:


logit_1_confusion = confusion_matrix(y_test, logit_2.predict(X_test_scaled))
plt.figure(dpi=150)
sns.heatmap(logit_1_confusion, cmap=plt.cm.Blues, annot=True, square=True,
           xticklabels=['False','True'],
           yticklabels=['False','True'])

plt.xlabel('Predicted illness')
plt.ylabel('Actual illness')
plt.title('Logistic regression confusion matrix');


# #### For a start, it is not bad. However, let's try other models and ensembles of different models to check the best-performing path. 

# ### kNN Model

# In[384]:



knn_model = KNeighborsClassifier(15)


# In[385]:


knn_model.fit(X_train_scaled,y_train)


# In[386]:


print("The score for KNN model is")
print("Training: {:6.2f}%".format(100*knn_model.score(X_train_scaled, y_train)))
print("Val set: {:6.2f}%".format(100*knn_model.score(X_val_scaled, y_val)))


# In[387]:


print('kNN validation metrics:  \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (
         precision_score(y_val, knn_model.predict(X_val_scaled)), 
         recall_score(y_val, knn_model.predict(X_val_scaled)),
         f1_score(y_val, knn_model.predict(X_val_scaled))
        )
     )


# ### Random Forest Model

# In[388]:


rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy')


# In[389]:


rf_model.fit(X_train,y_train)


# In[390]:


print("The score for Random Forest Classifier is")
print("Training: {:6.2f}%".format(100*rf_model.score(X_train, y_train)))
print("Val set: {:6.2f}%".format(100*rf_model.score(X_val, y_val)))


# In[391]:


print('Random Forest validation metrics:  \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (
         precision_score(y_val, rf_model.predict(X_val)), 
         recall_score(y_val, rf_model.predict(X_val)),
         f1_score(y_val, rf_model.predict(X_val))
        )
     )


# ### Extra Trees

# In[392]:


et_model = ExtraTreesClassifier(n_estimators=100)


# In[393]:


et_model.fit(X_train,y_train)


# In[394]:


print("The score for Extra Trees Classifier is")
print("Training: {:6.2f}%".format(100*et_model.score(X_train, y_train)))
print("Val set: {:6.2f}%".format(100*et_model.score(X_val, y_val)))


# In[395]:


print('Extra Trrs validation metrics:  \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (
         precision_score(y_val, rf_model.predict(X_val)), 
         recall_score(y_val, rf_model.predict(X_val)),
         f1_score(y_val, rf_model.predict(X_val))
        )
     )


# ### XG Boost 

# In[396]:


gbm = xgb.XGBClassifier( 
                        n_estimators=30000,
                        max_depth=4,
                        objective='binary:logistic', 
                        learning_rate=.05, 
                        subsample=.8,
                        min_child_weight=3,
                        colsample_bytree=.8
                       )

eval_set=[(X_train,y_train),(X_val,y_val)]
fit_model = gbm.fit( 
                    X_train, y_train, 
                    eval_set=eval_set,
                    eval_metric='error',
                    early_stopping_rounds=50,
                    verbose=False
                   )

accuracy_score(y_test, gbm.predict(X_test, ntree_limit=gbm.best_ntree_limit)) 


# In[397]:


print('Gradient Boosting validation metrics:  \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (
         precision_score(y_val, gbm.predict(X_val)), 
         recall_score(y_val, gbm.predict(X_val)),
         f1_score(y_val, gbm.predict(X_val))
        )
     )


# In[398]:


xgb.plot_importance(gbm)
xgb.plot_importance(gbm, importance_type='gain')


# ### Gaussian Naive Bayes

# In[399]:


nb_model = GaussianNB()


# In[400]:


nb_model.fit(X_train, y_train)


# In[401]:


print("The score for Naive Bayes is")
print("Training: {:6.2f}%".format(100*nb_model.score(X_train, y_train)))
print("Val set: {:6.2f}%".format(100*nb_model.score(X_val, y_val)))


# In[402]:


print('Naive Bayes validation metrics:  \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (
         precision_score(y_val, nb_model.predict(X_val)), 
         recall_score(y_val, nb_model.predict(X_val)),
         f1_score(y_val, nb_model.predict(X_val))
        )
     )


# In[403]:


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, logit_2.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr, tpr)

fpr, tpr, _ = roc_curve(y_test, knn_model.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr, tpr)

fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)

fpr, tpr, _ = roc_curve(y_test, et_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)

fpr, tpr, _ = roc_curve(y_test, gbm.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)

fpr, tpr, _ = roc_curve(y_test, nb_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)



x = np.linspace(0,1, 100000)
plt.plot(x, x, linestyle='--')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Logistic Regression','kNN', 'Random Forest', 'Extra Trees', 'Gradient Boosting', 'Gaussian Naive Bayes'])


# In[405]:


explainer = shap.Explainer(gbm)
shap_values = explainer(X_test)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])


# In[406]:


shap.plots.bar(shap_values)


# ####  For this project, identifying the positive class is more important than the total accuracy of the model. The best model amply the highest recall. The best result has Random Forest from the previous models so we will continue with it. 

# In[407]:


print("The recall we have recived from different models:")
print("Logistic Regression: {:6.2f}%".format(100*recall_score(y_val, logit_2.predict(X_val_scaled))))
print("kNN: {:6.2f}%".format(100*recall_score(y_val, knn_model.predict(X_val_scaled))))
print("Random Forest: {:6.2f}%".format(100*recall_score(y_val, rf_model.predict(X_val))))
print("Extra Trees: {:6.2f}%".format(100*recall_score(y_val, rf_model.predict(X_val))))
print("Gradient Boosting: {:6.2f}%".format(100*recall_score(y_val, gbm.predict(X_val))))
print("Naive Bayes: {:6.2f}%".format(100*recall_score(y_val, nb_model.predict(X_val))))


# #### However, the base Random Forest model shows a big overfitting (the difference in accuracy between train and validation data is more than 30%)

# In[408]:


print("The score for Random Forest Classifier is")
print("Training: {:6.2f}%".format(100*rf_model.score(X_train, y_train)))
print("Test set: {:6.2f}%".format(100*rf_model.score(X_val, y_val)))


# #### So we need to tune hyperparameters in order to reduce overfitting
# #### 1. Increase n_estimators, the more trees, the less likely the RF is to overfit
# #### 2. reduce max_features parameter
# #### 3. max_depth tuned to 12 
# #### 4. tuned max_samples_leaf greater than 2.

# In[410]:


rf_model_new = RandomForestClassifier(n_estimators=170, criterion='entropy', max_features=3, max_depth=12, min_samples_leaf=3)


# In[411]:


rf_model_new.fit(X_train,y_train)


# In[412]:


print("The score for Random Forest New Classifier is")
print("Training: {:6.2f}%".format(100*rf_model_new.score(X_train, y_train)))
print("Test set: {:6.2f}%".format(100*rf_model_new.score(X_val, y_val)))


# In[413]:


print('Random Forest test metrics:  \n Precision: %.4f \n Recall: %.4f \n F1: %.4f' %
        (
         precision_score(y_test, rf_model_new.predict(X_test)), 
         recall_score(y_test, rf_model_new.predict(X_test)),
         f1_score(y_test, rf_model_new.predict(X_test))
        )
     )


# In[414]:


rf_confusion = confusion_matrix(y_test, rf_model_new.predict(X_test))
rf_confusion


# #### As our goal is to receive the highest recall, I will try to use soft prediction with the thresholds 0.3 and 0.4.

# In[415]:


rf_confusion_04 = confusion_matrix(y_test, (rf_model_new.predict_proba(X_test)[:,1]>=0.4).astype(int))
rf_confusion_04


# In[416]:


rf_confusion_03 = confusion_matrix(y_test, (rf_model_new.predict_proba(X_test)[:,1]>=0.3).astype(int))
rf_confusion_03


# In[417]:


print ('Recall score with threshold 0.4: {:6.2f}%'. 
       format(100* recall_score(y_test, (rf_model_new.predict_proba(X_test)[:,1]>=0.4).astype(int))))
print ('Recall score with threshold 0.3: {:6.2f}%'. 
       format(100* recall_score(y_test, (rf_model_new.predict_proba(X_test)[:,1]>=0.3).astype(int))))


# In[418]:


from sklearn.metrics import roc_curve


fpr, tpr, _ = roc_curve(y_test, (rf_model_new.predict_proba(X_test)[:,1]>0.3))
plt.plot(fpr, tpr)

fpr, tpr, _ = roc_curve(y_test, (rf_model_new.predict_proba(X_test)[:,1]>0.4))
plt.plot(fpr, tpr)

x = np.linspace(0,1, 100000)
plt.plot(x, x, linestyle='--')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['Random Forest Thr 0.3','Random Forest Thr 0.4'])


# In[419]:


print("ROC AUC score 0.4 = ", roc_auc_score(y_test, (rf_model_new.predict_proba(X_test)[:,1]>=0.4)))
print("ROC AUC score 0.3 = ", roc_auc_score(y_test, (rf_model_new.predict_proba(X_test)[:,1]>=0.3)))


# #### As we see the ROC AUC decreasing with the threshold of 0.3, in order to have better model predicting capability, we will keep the model with the threshold of 0.4.

# In[420]:


explainer_rf = shap.Explainer(rf_model_new)
shap_values_rf = explainer(X_test)


shap.plots.waterfall(shap_values_rf[0])


# In[421]:


shap.plots.bar(shap_values_rf)


# In[422]:


from sklearn.inspection import permutation_importance


# In[423]:


rf_model_new.feature_importances_


# In[ ]:





# In[439]:


sorted_f = rf_model_new.feature_importances_.argsort()
plt.barh(data.columns[sorted_f], rf_model_new.feature_importances_[sorted_f])
plt.xlabel("Random Forest Feature Importance")


# In[331]:


perm_importance = permutation_importance(rf_model_new, X_test, y_test)


# In[438]:


sorted_fc = perm_importance.importances_mean.argsort()
plt.barh(data.columns[sorted_fc], perm_importance.importances_mean[sorted_fc])
plt.xlabel("Permutation Importance")


# In[428]:


data['probability']=rf_model_new.predict_proba(X)[:,1]


# In[430]:


data.head()


# In[431]:


data = data[data.systolic_blood_pressure.between(0,500)]


# In[432]:



data = data[data.diastolic_blood_pressure.between(0,500)]


# In[433]:


#let's see how data looks for high-risk patients
data_HR = data[data.probability>=0.85]
                 


# In[434]:


data_HR.describe()


# In[435]:


data_HR.mode()


# In[436]:


#data for patients with medium risk
data_MR=data[(data.probability<0.85)&(data.probability>0.60)]


# In[437]:


data_MR.mode()


# In[ ]:





# In[ ]:





# In[ ]:




