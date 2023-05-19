#!/usr/bin/env python
# coding: utf-8

# **This notebook contains followings:**
# * Data Merging
# * Feature Extraction
# * Model Selection
# * Hyperparameter Tuning
# * Results on All Datasets
# * Conclusion
# 

# In[1]:


get_ipython().run_cell_magic('capture', '', '#importing necessary libraries\nimport os\nimport numpy as np \nimport pandas as pd \nimport matplotlib.pyplot as plt \nimport scipy\nfrom scipy.stats import entropy\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score,f1_score\nfrom sklearn import preprocessing\nimport xgboost as xgb\nimport catboost as cb\nimport lightgbm as lgbm\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.neural_network import MLPClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.svm import SVC\nfrom sklearn.gaussian_process import GaussianProcessClassifier\nfrom sklearn.gaussian_process.kernels import RBF\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis\nimport plotly.express as px\nimport plotly.graph_objects as go\nimport optuna')


# # Merging Data

# In[2]:


get_ipython().run_cell_magic('capture', '', "#Data paths\ndataset_path_1st = 'dataset/vibration/1st_test'\ndataset_path_2nd = 'dataset/vibration/2nd_test'\ndataset_path_3rd = 'dataset/vibration/3rd_test'")


# In[3]:


# Test for the first file
dataset = pd.read_csv('dataset/vibration/1st_test/2003.10.22.12.06.24', sep='\t')
ax = dataset.plot(figsize = (24,6), title= "Bearing Vibration" , legend = True)
ax.set(xlabel="cycle(n)", ylabel="vibration/acceleration(g)")
plt.show()


# # Feature Extraction #

# In this section several signal-based statistical features has been extracted from the data                                 
# **Definition and formula of the features:**
# * ***Absolute Mean*** $$\overline{x} = \frac{1}{N}\sum_{i=1}^{N}|x_i| $$
# 
# * ***Standart Deviation:*** $$\sigma         = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \overline{x})^2}$$
# * ***Skewness:*** 
# Asymmetry of a signal distribution. Faults can impact distribution symmetry and therefore increase the level of skewness.
# $$\mathrm{Sk} = \frac{1}{N}\sum_{i=1}^{N}\frac{(x_i-\overline{x})^3}{\sigma^3}$$
# * ***Kurtosis:***
# Length of the tails of a signal distribution, or equivalently, how outlier prone the signal is. Developing faults can increase the number of outliers, and therefore increase the value of the kurtosis metric.
# $$\mathrm{K} = \frac{1}{N}\sum_{i=1}^{N}\frac{(x_i-\overline{x})^4}{\sigma^4}$$
# * ***Entropy:***$$ H(X) = -\sum_{i=1}^{N} P(x_i)\log{P(x_i)} $$
#                             
# * ***RMS:*** $$x_{rms} =\sqrt{(\frac{1}{N})\sum_{i=1}^{N}(x)^{2}}$$
# 
# * ***Peak to Peak:*** $$ x_p = \max \mathrm{value} - \min \mathrm{value}$$
# 
# * ***Crest Factor:*** 
# Peak value divided by the RMS. Faults often first manifest themselves in changes in the peakiness of a signal before they manifest in the energy represented by the signal root mean squared. The crest factor can provide an early warning for faults when they first develop. 
# $$x_{crest} =\frac{\max \mathrm{value}}{\mathrm{x_{rms}}}$$
# 
# * ***Clearence Factor:*** 
# Peak value divided by the squared mean value of the square roots of the absolute amplitudes. For rotating machinery, this feature is maximum for healthy bearings and goes on decreasing for defective ball, defective outer race, and defective inner race respectively. The clearance factor has the highest separation ability for defective inner race faults.
# $$ x_{clear} = \frac{x_p}{(\frac{1}{N}\sum_{i=1}^{N}\sqrt{|x_i|})^2}  $$
# 
# * ***Shape Factor:*** 
# RMS divided by the mean of the absolute value. Shape factor is dependent on the signal shape while being independent of the signal dimensions.
# $$\frac{x_{rms}}{\overline{x}}$$
# 
# * ***Impulse:*** 
# Compare the height of a peak to the mean level of the signal.
# $$\frac{\max \mathrm{value}}{\overline{x}}  $$
# 

# In[4]:


# Root Mean Squared Sum
def calculate_rms(df):
    result = []
    for col in df:
        r = np.sqrt((df[col]**2).sum() / len(df[col]))
        result.append(r)
    return result

# extract peak-to-peak features
def calculate_p2p(df):
    return np.array(df.max().abs() + df.min().abs())

# extract shannon entropy (cut signals to 500 bins)
def calculate_entropy(df):
    ent = []
    for col in df:
        ent.append(entropy(pd.cut(df[col], 500).value_counts()))
    return np.array(ent)
# extract clearence factor
def calculate_clearence(df):
    result = []
    for col in df:
        r = ((np.sqrt(df[col].abs())).sum() / len(df[col]))**2
        result.append(r)
    return result

def time_features(dataset_path, id_set=None):
    time_features = ['mean','std','skew','kurtosis','entropy','rms','max','p2p', 'crest', 'clearence', 'shape', 'impulse']
    cols1 = ['B1_x','B1_y','B2_x','B2_y','B3_x','B3_y','B4_x','B4_y']
    cols2 = ['B1','B2','B3','B4']
    
    # initialize
    if id_set == 1:
        columns = [c+'_'+tf for c in cols1 for tf in time_features]
        data = pd.DataFrame(columns=columns)
    else:
        columns = [c+'_'+tf for c in cols2 for tf in time_features]
        data = pd.DataFrame(columns=columns)

        
        
    for filename in os.listdir(dataset_path):
        # read dataset
        raw_data = pd.read_csv(os.path.join(dataset_path, filename), sep='\t')
        
        # time features
        mean_abs = np.array(raw_data.abs().mean())
        std = np.array(raw_data.std())
        skew = np.array(raw_data.skew())
        kurtosis = np.array(raw_data.kurtosis())
        entropy = calculate_entropy(raw_data)
        rms = np.array(calculate_rms(raw_data))
        max_abs = np.array(raw_data.abs().max())
        p2p = calculate_p2p(raw_data)
        crest = max_abs/rms
        clearence = np.array(calculate_clearence(raw_data))
        shape = rms / mean_abs
        impulse = max_abs / mean_abs
        
        if id_set == 1:
            mean_abs = pd.DataFrame(mean_abs.reshape(1,8), columns=[c+'_mean' for c in cols1])
            std = pd.DataFrame(std.reshape(1,8), columns=[c+'_std' for c in cols1])
            skew = pd.DataFrame(skew.reshape(1,8), columns=[c+'_skew' for c in cols1])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,8), columns=[c+'_kurtosis' for c in cols1])
            entropy = pd.DataFrame(entropy.reshape(1,8), columns=[c+'_entropy' for c in cols1])
            rms = pd.DataFrame(rms.reshape(1,8), columns=[c+'_rms' for c in cols1])
            max_abs = pd.DataFrame(max_abs.reshape(1,8), columns=[c+'_max' for c in cols1])
            p2p = pd.DataFrame(p2p.reshape(1,8), columns=[c+'_p2p' for c in cols1])
            crest = pd.DataFrame(crest.reshape(1,8), columns=[c+'_crest' for c in cols1])
            clearence = pd.DataFrame(clearence.reshape(1,8), columns=[c+'_clearence' for c in cols1])
            shape = pd.DataFrame(shape.reshape(1,8), columns=[c+'_shape' for c in cols1])
            impulse = pd.DataFrame(impulse.reshape(1,8), columns=[c+'_impulse' for c in cols1])
            
        else:
            mean_abs = pd.DataFrame(mean_abs.reshape(1,4), columns=[c+'_mean' for c in cols2])
            std = pd.DataFrame(std.reshape(1,4), columns=[c+'_std' for c in cols2])
            skew = pd.DataFrame(skew.reshape(1,4), columns=[c+'_skew' for c in cols2])
            kurtosis = pd.DataFrame(kurtosis.reshape(1,4), columns=[c+'_kurtosis' for c in cols2])
            entropy = pd.DataFrame(entropy.reshape(1,4), columns=[c+'_entropy' for c in cols2])
            rms = pd.DataFrame(rms.reshape(1,4), columns=[c+'_rms' for c in cols2])
            max_abs = pd.DataFrame(max_abs.reshape(1,4), columns=[c+'_max' for c in cols2])
            p2p = pd.DataFrame(p2p.reshape(1,4), columns=[c+'_p2p' for c in cols2])
            crest = pd.DataFrame(crest.reshape(1,4), columns=[c+'_crest' for c in cols2])
            clearence = pd.DataFrame(clearence.reshape(1,4), columns=[c+'_clearence' for c in cols2])
            shape = pd.DataFrame(shape.reshape(1,4), columns=[c+'_shape' for c in cols2])
            impulse = pd.DataFrame(impulse.reshape(1,4), columns=[c+'_impulse' for c in cols2])
            
        mean_abs.index = [filename]
        std.index = [filename]
        skew.index = [filename]
        kurtosis.index = [filename]
        entropy.index = [filename]
        rms.index = [filename]
        max_abs.index = [filename]
        p2p.index = [filename]
        crest.index = [filename]
        clearence.index = [filename]
        shape.index = [filename]
        impulse.index = [filename] 
        
        # concat
        merge = pd.concat([mean_abs, std, skew, kurtosis, entropy, rms, max_abs, p2p,crest,clearence, shape, impulse], axis=1)
        data = data.append(merge)
        
    if id_set == 1:
        cols = [c+'_'+tf for c in cols1 for tf in time_features]
        data = data[cols]
    else:
        cols = [c+'_'+tf for c in cols2 for tf in time_features]
        data = data[cols]
        
    data.index = pd.to_datetime(data.index, format='%Y.%m.%d.%H.%M.%S')
    data = data.sort_index()
    return data                                  


# 1.  Calling feature extraction function defined above to merge extracted features      
# 2.  Saving as .csv file

# In[5]:


get_ipython().run_cell_magic('capture', '', "set1 = time_features(dataset_path_1st, id_set=1)\nset1.to_csv('set1_timefeatures.csv')")


# Reading Data again

# In[6]:


set1 = pd.read_csv("./set1_timefeatures.csv")


# Changing indexing column to time which is also name of the each file

# In[7]:


set1 = set1.rename(columns={'Unnamed: 0':'time'})
set1.set_index('time')
set1.describe()


# Plotting features of all bearings in both x and y axis

# In[8]:


time_features_list = ["mean","std","skew","kurtosis","entropy","rms","max","p2p", "crest", "clearence", "shape", "impulse"]
bearings_xy = [["B"+str(n)+"_"+str(o)+"_" for n in range(1,5)] for o in ['x','y'] ] 
#print(bearings_xy)
for tf in time_features_list:
    fig = plt.figure()
    # Divide the figure into a 1x4 grid, and give me the first section
    ax1 = fig.add_subplot(141)
    # Divide the figure into a 1x4 grid, and give me the second section
    ax2 = fig.add_subplot(142)
    #...so on
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)
    axes = [ax1,ax2,ax3, ax4]
    
    for i in range(4):
        col = bearings_xy[0][i]+tf
        set1[col].plot(figsize = (36,6), title="Bearing{} x-y_".format(i+1)+tf , legend = True, ax=axes[i])
        col = bearings_xy[1][i]+tf
        set1[col].plot(figsize = (36,6) , legend = True, ax=axes[i])
        axes[i].set(xlabel="cycle", ylabel="value")
        
    
    


# In[9]:


#Health Status labels are added according to following dictionary
B1 ={
    "early" : ["2003-10-22 12:06:24" , "2003-10-23 09:14:13"],
    "suspect" : ["2003-10-23 09:24:13" , "2003-11-08 12:11:44"],
    "normal" : ["2003-11-08 12:21:44" , "2003-11-19 21:06:07"],
    "suspect_1" : ["2003-11-19 21:16:07" , "2003-11-24 20:47:32"],
    "imminent_failure" : ["2003-11-24 20:57:32","2003-11-25 23:39:56"]
}
B2 = {
    "early" : ["2003-10-22 12:06:24" , "2003-11-01 21:41:44"],
    "normal" : ["2003-11-01 21:51:44" , "2003-11-24 01:01:24"],
    "suspect" : ["2003-11-24 01:11:24" , "2003-11-25 10:47:32"],
    "imminient_failure" : ["2003-11-25 10:57:32" , "2003-11-25 23:39:56"]
}

B3 = {
    "early" : ["2003-10-22 12:06:24" , "2003-11-01 21:41:44"],
    "normal" : ["2003-11-01 21:51:44" , "2003-11-22 09:16:56"],
    "suspect" : ["2003-11-22 09:26:56" , "2003-11-25 10:47:32"],
    "Inner_race_failure" : ["2003-11-25 10:57:32" , "2003-11-25 23:39:56"]
}

B4 = {
    "early" : ["2003-10-22 12:06:24" , "2003-10-29 21:39:46"],
    "normal" : ["2003-10-29 21:49:46" , "2003-11-15 05:08:46"],
    "suspect" : ["2003-11-15 05:18:46" , "2003-11-18 19:12:30"],
    "Rolling_element_failure" : ["2003-11-19 09:06:09" , "2003-11-22 17:36:56"],
    "Stage_two_failure" : ["2003-11-22 17:46:56" , "2003-11-25 23:39:56"]
}


# Adding labels to the dataframe

# In[10]:


B1_state = list()
B2_state = list()
B3_state = list()
B4_state = list()
cnt = 0

for row in set1["time"]:
    cnt += 1
    # B1
    if cnt<=151:
        B1_state.append("early")
    if 151 < cnt <=600:
        B1_state.append("suspect")
    if 600 < cnt <=1499:
        B1_state.append("normal")
    if 1499 < cnt <=2098:
        B1_state.append("suspect")
    if 2098 < cnt <= 2156:
        B1_state.append("imminent_failure")
    #B2
    if cnt<=500:
        B2_state.append("early")
    if 500 < cnt <=2000:
        B2_state.append("normal")
    if 2000 < cnt <=2120:
        B2_state.append("suspect")
    if 2120< cnt <=2156:
        B2_state.append("imminet_failure")

    #B3
    if cnt<=500:
        B3_state.append("early")
    if 500 < cnt <= 1790:
        B3_state.append("normal")
    if 1790 < cnt <=2120:
        B3_state.append("suspect")
    if 2120 < cnt <=2156:
        B3_state.append("Inner_race_failure")
    #B4
    if cnt<=200:
        B4_state.append("early")
    if 200 < cnt <=1000:
        B4_state.append("normal")
    if 1000 < cnt <= 1435:
        B4_state.append("suspect")
    if 1435 < cnt <=1840:
        B4_state.append("Inner_race_failure")
    if 1840 < cnt <=2156:
        B4_state.append("Stage_two_failure")
#controlling the counts
from collections import Counter
print(Counter(B1_state))
print(Counter(B2_state))
print(Counter(B3_state))
print(Counter(B4_state))

set1["B1_state"] = B1_state
set1["B2_state"] = B2_state
set1["B3_state"] = B3_state
set1["B4_state"] = B4_state

set1.head()


# Splitting each bearing data and adding them end to end vertically

# In[11]:


B1_cols = [col for col in set1.columns if "B1" in col]
B2_cols = [col for col in set1.columns if "B2" in col]
B3_cols = [col for col in set1.columns if "B3" in col]
B4_cols = [col for col in set1.columns if "B4" in col]

B1 = set1[B1_cols]
B2 = set1[B2_cols]
B3 = set1[B3_cols]
B4 = set1[B4_cols]
cols = ['Bx_mean','Bx_std','Bx_skew','Bx_kurtosis','Bx_entropy','Bx_rms','Bx_max','Bx_p2p','Bx_crest', 'Bx_clearence', 'Bx_shape', 'Bx_impulse',
        'By_mean','By_std','By_skew','By_kurtosis','By_entropy','By_rms','By_max','By_p2p','By_crest', 'By_clearence', 'By_shape', 'By_impulse',
        'class']
B1.columns = cols
B2.columns = cols
B3.columns = cols
B4.columns = cols
final_data = pd.concat([B1,B2,B3,B4], axis=0, ignore_index=True)
final_data.describe()


# Train-Test Split (70%-30%)

# In[12]:


X = final_data.copy()
y = X.pop("class")
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =1)


# Evaluating Performance of Base Models

# In[13]:


xgb_cl = xgb.XGBClassifier(use_label_encoder=False)
xgb_cl.fit(X_train, y_train)
preds = xgb_cl.predict(X_test)
accuracy_score(y_test, preds)


# **From now on, only X axis data will be used**
#                                                                                                                           
# So data acquired from y axis will be excluded  

# In[14]:


x_axis_cols = ["Bx_"+tf for tf in time_features_list]
print(x_axis_cols)
X_x = X.copy()
X_x = X[x_axis_cols]
cols = ['B_mean','B_std','B_skew','B_kurtosis','B_entropy',
        'B_rms','B_max','B_p2p','B_crest', 'B_clearence', 'B_shape', 'B_impulse']
X_x.columns = cols
X_x_train, X_x_test, y_train, y_test = train_test_split(X_x, y, test_size = 0.3, random_state =1)


# # Model Selection

# Now we can compare several learning algorithms

# In[15]:


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", 
         #"Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA","XGBoost","CatGBoost","LightGBoost"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    xgb.XGBClassifier(),
    cb.CatBoostClassifier(verbose = False),
    lgbm.LGBMClassifier()
    ]

for name, clf in zip(names,classifiers):
    print("training "+name+" ...")
    clf.fit(X_x_train,y_train)
    score = clf.score(X_x_test,y_test)
    print('Score of'+name+' is: '+str(score))
    #iterate over classifiers
    


# # Hyperparameter Tuning
# * Since accuracy score of the last 3 base model,namely XGBoost,CatBoost, LightBoost, are the highest ones, we will continue with one of them in further investigations  
# * Parameters of the selected model will be tuned with optuna library
# * Typical values were given as interval of parameters

# In[16]:


get_ipython().run_cell_magic('capture', '', 'def objective(trial):\n    xgb_params = dict(\n        max_depth=trial.suggest_int("max_depth", 2, 10),\n        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),\n        n_estimators=trial.suggest_int("n_estimators", 1000, 8000),\n        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),\n        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),\n        subsample=trial.suggest_float("subsample", 0.2, 1.0),\n        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),\n        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),\n        )\n    xgb_cl = xgb.XGBClassifier(\n            booster=\'gbtree\',\n            tree_method =\'gpu_hist\',\n            use_label_encoder=False,\n            **xgb_params)\n    xgb_cl.fit(X_x_train, y_train)\n    preds = xgb_cl.predict(X_x_test)\n    accuracy_score(y_test, preds)\n    return accuracy_score(y_test,preds)\n\nstudy = optuna.create_study(direction="maximize")\nstudy.optimize(objective, n_trials=20)\nxgb_params = study.best_params')


# Final XGBoost parameters were obtained by hyperparametertuning using GPU but final model should be trained by CPU to get possibly better result                                             
# 

# In[17]:


final_model = xgb.XGBClassifier(use_label_encoder=False,
                                **xgb_params)
final_model.fit(X_x_train, y_train)
preds = final_model.predict(X_x_test)
accuracy_score(y_test, preds)
#performance is increased


# Loading Dataset 2 and 3 to observe the result of the final model on them
# 

# In[18]:


get_ipython().run_cell_magic('capture', '', "set2 = time_features(dataset_path_2nd, id_set=2)\nset2.to_csv('set2_timefeatures.csv')\nset3 = time_features(dataset_path_3rd, id_set=3)\nset3.to_csv('set3_timefeatures.csv')")


# In[19]:


set2 = pd.read_csv("./set2_timefeatures.csv")
set3 = pd.read_csv("./set3_timefeatures.csv")
time_2 = set2['Unnamed: 0']
#Setting index columns to time
set2 = set2.rename(columns={'Unnamed: 0':'time'}).set_index('time')
set3 = set3.rename(columns={'Unnamed: 0':'time'}).set_index('time')


# Vertically splitting bearing datum to obtain each bearings' data seperately

# In[20]:


B1_cols = [col for col in set2.columns if "B1" in col]
B2_cols = [col for col in set2.columns if "B2" in col]
B3_cols = [col for col in set2.columns if "B3" in col]
B4_cols = [col for col in set2.columns if "B4" in col]

set2_B1 = set2[B1_cols]
set2_B2 = set2[B2_cols]
set2_B3 = set2[B3_cols]
set2_B4 = set2[B4_cols]

set3_B1 = set3[B1_cols]
set3_B2 = set3[B2_cols]
set3_B3 = set3[B3_cols]
set3_B4 = set3[B4_cols]

set2_B1.columns = cols
set2_B2.columns = cols
set2_B3.columns = cols
set2_B4.columns = cols
set3_B1.columns = cols
set3_B2.columns = cols
set3_B3.columns = cols
set3_B4.columns = cols


# # Results
# Here, we using plotly library which is more useful than matplotlib in terms of interactivity                   
# Extracted features are plotted with their "health state" label

# In[21]:


# HERE number and dataset of bearing can be changed !!!
bearing = set2_B1

#predicting state of bearing with final_model
preds = final_model.predict(bearing)
preds = le.inverse_transform(preds)
#inserting prediction and time to the dataframe
bearing.insert(12,'state',preds)
bearing.insert(13, 'time',bearing.index)

for tf in time_features_list:
    col = "B_{}".format(tf)
    print(col)
    fig=go.Figure((go.Scatter(x=bearing['time'], y=bearing[col],
                             mode='lines',
                             line=dict(color='rgba(0,0,220,0.8)'))))
    fig.add_traces(px.scatter(bearing, x='time', y=col, color='state').data)
    fig.update_layout(template='plotly_dark')
    fig.update_xaxes(showgrid=False)
    fig.show()
    


# In[22]:


bearing


# **NOTE**                                                                                                       
# This work is carried out in SensHero Predictive Maintenance Solutions
# 
