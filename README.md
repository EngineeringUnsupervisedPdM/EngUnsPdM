# Implementation of "Engineering an unsupervised predictive maintenance solution: a real world case-study".

Two main algorithms are provided:

1. **Profile Based**: An algorithm used to detect anomalies by measuring the distance of upcoming data to its intial state.

2. **PdM_TranAD**: An algorithm that utilize TranAD model from https://github.com/imperial-qore/TranAD


### Requirments for ProfileBased model
```
pip3 install -r requirementsPB.txt
```

### Requirments for PdM_TranAD model
```
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```

The documentation and examples for all methods can be found in **examples.ipynb** file.


To run the code, you will need to extract the files (methods.py, ProfileBased.py, and TranADscripts.py) from Methods.zip and place them in the root folder. To obtain the password required for extraction, please send a request to the following email address: engineeringunsupervised@gmail.com.


## Usage of Profile-Based for run-to-failure episodes:

```python
# apply profilebase algorithm
# We choose Selftuning technique to calculate threshold and window_ma=30 for smoothing the anomaly scores.
# Moreover by defaulr (LimitMaxdist = None) we choose the automatic proccess of profile calculation.
# the execution time is depending on choice of distance metric

# metric can be "euclidean","cc" for SBD,"dtw" for DTW and "rbf" for RBF kernel distance
anomalyscores,threshold=methods.profileBased(df,metric="cc",profileSize=60,factor=14,window_ma=30)

plt.plot(anomalyscores)
plt.axhline(threshold,color="red")
plt.show()
```
![alt text](./images/first.png?raw=true)

More examples are shown in **examples.ipynb** notebook.

## Usage of Run-to-Failure PdM Evaluation metric:

An appropriate evaluation metric for run-to-failure episodes.


Load data and run Profile Based and store results:
```python
df2=pd.read_csv("datasamples/Episode2.csv",index_col=0)
df2 = df2.sub(df2.min(axis=1), axis=0).div(df2.max(axis=1) - df2.min(axis=1), axis=0)

df=pd.read_csv("datasamples/Episode6.csv",index_col=0)
df = df.sub(df.min(axis=1), axis=0).div(df.max(axis=1) - df.min(axis=1), axis=0)


predictions=[]
thresholds=[]
for episodedf in [df,df2]:
    anomalyscores,threshold=methods.profileBased(episodedf,metric="cc",profileSize=60,factor=10,window_ma=30)
    predictions.append(anomalyscores)
    thresholds.append(threshold)

```

Use Range based evaluation method configured for PdM.
    
```python
recall,Precision,fbeta,axes=evaluation.myeval(predictions,thresholds,PH="300",lead="30")
plt.show()

print(f"F1: AD1 {fbeta[0]},AD2 {fbeta[1]},AD3 {fbeta[2]}")
print(f"Recall: AD1 {recall[0]},AD2 {recall[1]},AD3 {recall[2]}")
print(f"Precission: {Precision}")
```
![alt text](./images/second.png?raw=true)

Output:
```
F1: AD1 0.7069306930693069,AD2 0.5984911986588433,AD3 0.5419035427493688
Recall: AD1 1.0,AD2 0.6611111111111112,AD3 0.5371832718327184
Precission: 0.5467075038284839
```

## Case Study results:

**CaseStudy.ipynb** notebook has reference code for loading case study data, runing the proposed methods and finally evaluating the results.

Run method for all availible episodes.
```python
predictions=[]
thresholds=[]
indexes=[]
isfailure=[]
for episode in range(1,21):
    if episode<13:
        isfailure.append(1)
    else:
        isfailure.append(0)
    
    df=pd.read_csv(f"CaseStudyData/Episode{episode}.csv",index_col=0)
    
    dfFeats=methods.calculateFeatures(df)
    
    
    #calculate resets for that episode (position of index)
    ep_resets=episoderesets(dfFeats,allResets)
    
    allerrors,allindexes,allthresholds=methods.profileBasedWithResets(dfFeats,ep_resets,metric="euclidean",profileSize=60,factor=6.5,window_ma=30)

    predictions.append(allerrors)
    thresholds.extend(allthresholds)
    indexes.append(allindexes)
```

Run evaluation process.
```python
recall,Precision,fbeta,axes=evaluation.myeval(predictions,thresholds,datesofscores=indexes,PH="210",lead="26",beta=2,isfailure=isfailure,ignoredates=allingoreindexes)
plt.show()
print(f"F2: AD1 {fbeta[0]},AD2 {fbeta[1]},AD3 {fbeta[2]}")
print(f"Recall: AD1 {recall[0]},AD2 {recall[1]},AD3 {recall[2]}")
print(f"Precission: {Precision}")
```
![alt text](./images/third.png?raw=true)

Output:
```
F2: AD1 0.4852823703129917,AD2 0.42579345630746623,AD3 0.3491527272447874
Recall: AD1 0.9166666666666666,AD2 0.6892820953161437,AD3 0.4772849690355068
Precission: 0.16836027713625867
```
