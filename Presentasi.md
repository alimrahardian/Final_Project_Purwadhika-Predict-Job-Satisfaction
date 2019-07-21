
# Job Satisfaction Prediction Model
based on kaggle Stack Overflow 2018 Developer Survey



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
survey = pd.read_csv('stack-overflow-2018-developer-survey/survey_results_public.csv')
print(survey.shape)
```

# Selecting Feature To Be Used

feature selected for this model are rather general in nature, as most dev has experience or can easily determind his/her current condition base on feature below

- Age : respondent age
- Gender : respondent gender
- Hobby : do respondent code as a hobby
- Hours Computer: How many hours spent in front of the computer a day
- Hour Outside : how much time do you spend outside?
- Years coding: for how many years have you been coding?
- Time Fully Productive: time takes for new coder to be productive and contributing in new work    environment
- Exercise : how many times do you exercise in a week?
- Job Satisfaction : How satisfied are you with your current job
- Employment : current employment status


```python
column_selected = ['Age', 'Gender', 'Hobby', 'HoursComputer', 'HoursOutside', 
'YearsCoding', 'TimeFullyProductive', 'Exercise', 'Employment', 'JobSatisfaction']
rawData = survey[column_selected]
rawData.head()
```


```python
# check null raw data
sns.heatmap(rawData.isnull())
plt.tight_layout()
plt.show()
```

heatmap show how raw data has many missing value
this happen because on original survey some of the feature aren't necessary to be filled
by respondent and the survey it self change its content overtime


```python
# check value on a feature

rawData = survey[column_selected]
rawData.Employment.value_counts()
```

example above show how the survey adding conteng over time that makes
gender have so many classifications this create noise that need to be fix
before data can be explored

 #PRE-PROCESSING

treatment needed before we can make use of this data are
1. drop missing value
2. reduce classifaction on some feature
3. create dummy variable


```python
# drop missing value

rawData = rawData.replace('nan', np.nan)
rawData = rawData.dropna()

sns.heatmap(rawData.isnull())
plt.tight_layout()
plt.show()
```

figure abow show our data now have no missing value
now after data in clean we can do more further pre processing


```python
# set feature and target

categorical = ['Age', 'Gender', 'Hobby', 'HoursComputer', 'HoursOutside', 
'YearsCoding', 'TimeFullyProductive', 'Exercise', 'Employment']

x = rawData[categorical]
y = rawData['JobSatisfaction']
```


```python
## CHECK VALUES
# Feature 
for item in categorical:
    catCount = x[item].values
    print(f'{item}', len(catCount))
    print(x[item].value_counts())

# Target
targetCatCount = y.values
targetValueCount = y.value_counts()
print('JobSatisfaction', len(targetCatCount))
print(targetValueCount)
```

as shown above some of our variable have too many categories that need to be reduced
so can have more generalize insight rather than too specific


```python
### MINIMIZING CATEGORIES
## Feature
# gender
nonBinaryGender = ['Non-binary, genderqueer, or gender non-conforming' , 'Female;Transgender' , 'Male;Non-binary, genderqueer, or gender non-conforming' ,
'Female;Male' , 'Transgender' , 'Transgender;Non-binary, genderqueer, or gender non-conforming' , 'Female;Non-binary, genderqueer, or gender non-conforming' ,
'Male;Transgender' , 'Female;Male;Transgender;Non-binary, genderqueer, or gender non-conforming' , 'Female;Transgender;Non-binary, genderqueer, or gender non-conforming' ,
'Female;Male;Transgender' , 'Male;Transgender;Non-binary, genderqueer, or gender non-conforming' , 'Female;Male;Non-binary, genderqueer, or gender non-conforming' ]
x['Gender'].replace(nonBinaryGender, 'Non Binary', inplace=True)

# years coding
years2_5 = ['3-5 years']
years5_10 = ['6-8 years', '9-11 years']
years10more = ['12-14 years', '15-17 years', '18-20 years', '21-23 years', '24-26 years',
'27-29 years', '30 or more years']
x['YearsCoding'].replace(years2_5, '2-5years',inplace=True)
x['YearsCoding'].replace(years5_10, '5-10 years',inplace=True)
x['YearsCoding'].replace(years10more, '10 years or more',inplace=True)

# time fully productive
moreThan6 = ['Six to nine months', 'Nine months to a year', 'More than a year']
x['TimeFullyProductive'].replace(moreThan6, 'six to more than a year', inplace=True)


## CHECK VALUES
# Feature 
for item in categorical:
    catCount = x[item].values
    print(f'{item}', len(catCount))
    print(x[item].value_counts())

# target
targetCatCount = y.values
targetValueCount = y.value_counts()
print('JobSatisfaction', len(targetCatCount))
print(targetValueCount)
print()

## Target
satisfied = ['Slightly satisfied', 'Moderately satisfied', 'Extremely satisfied']
disatisfied = ['Slightly dissatisfied', 'Moderately dissatisfied', 'Extremely dissatisfied']
y.replace(satisfied, 'Satisfied', inplace=True)
y.replace(disatisfied, 'Disatisfied', inplace=True)
print(y.value_counts())
```


```python
### VISUALIZE after before processing

joinDf = x.join(y)

## Target - Job Satisfaction
target_name = ['Satisfied', 'Disatisfied', 'Neither satisfied nor dissatisfied']
plt.pie(x=joinDf['JobSatisfaction'].value_counts(), labels=target_name, autopct='%1.0f%%')
plt.title('Job Satisfaction')
plt.tight_layout()
plt.show()


## Feature
# feature relation to target variable
for item in categorical:
    sns.countplot(y=joinDf[item], hue=joinDf['JobSatisfaction'])
    plt.tight_layout()
    plt.show()
```

some insight

- most respondent feel satisfied as a coder
- most dev age are under 34yo
- most dev are male
- most dev start / code as a hobby
- most dev spend more than 9 hours in front of computer
- most dev spend outside under 2 hours
- most dev have code for more than 5 years
- most dev start contributing productive result on new work environment under 3 months
- some dev don't do excercise that much in a week
- most dev are fulltime employed


```python
print('feature before dummies', x.info())
```


```python
### labeling category
# dummies feature
# drop original
for feature in categorical:
    dummies = pd.get_dummies(x[feature], prefix=feature)
    x = pd.concat([x, dummies], axis=1)
    x.drop([feature], axis=1, inplace=True)
print(x.info())
print('feature rows/cols',x.shape)
print()


# encode target
y.replace(
    ['Satisfied', 'Disatisfied', 'Neither satisfied nor dissatisfied'],
    [0, 1, 2],
    inplace=True)
```


```python
print(y.head())
```

# Modelling Machine Learning


```python
### Train Test Split
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=0)
```


```python
## random forest
from sklearn import metrics
import timeit
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(n_estimators=100, 
                             n_jobs=-1, 
                             random_state=42,
                             max_features=0.2, 
                             min_samples_leaf=1)
rfc.fit(xtrain, ytrain)
ypred_rfc = rfc.predict(xtest)
rfc_as = metrics.accuracy_score(ytest, ypred_rfc)
print('Forest score', rfc_as)
```


```python
import joblib
joblib.dump(rfc, 'stackOverflowSurvey_rfc3_comp', compress=3)
```


```python
## FEATURE IMPORTANCE
# by feature
def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):
    if autoscale:
        x_scale = model.feature_importances_.max() + headroom
    else:
        x_scale = 1
        
    feature_dict=dict(zip(feature_names, model.feature_importances_))
    
    if summarized_columns:
        for col_name in summarized_columns:
            sum_value = sum(x for i, x in feature_dict.items() if col_name in i )
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i ]
            for i in keys_to_remove:
                feature_dict.pop(i)
            feature_dict[col_name] = sum_value
    results = pd.Series(feature_dict, index=feature_dict.keys())
    results.sort_values(inplace=True)
    print(results)
    results.plot(kind='barh', figsize=(4, 4.5))

graph_feature_importances(rfc, x.columns, summarized_columns=categorical)
plt.tight_layout()
plt.show()
```


```python
# by feature categories

feature_importance = pd.Series(
    rfc.feature_importances_, 
    index=x.columns).sort_values()
feature_importance.plot(kind='barh', figsize=(6, 8))
plt.tight_layout()
plt.show()
```


# Currect Summary
Base on our model some of the feature contribute better than other:
have impact on predicttion:
- Time fully productive
- Exercise
- Our outside
- Hour computer
        
Less impact:
- Emplyment status
- Hobby
- Gender
        
follow up example:
- Employer may optimize training for new employee so they can quickly adapt  to their new new evironment as quick as possible
- Limit worker to face computer around 5 - 12 hours a day
- for dev who feel un-satisfied with current status try exercise more rather than move to new job
    
# Optimizing
base on two plot above we can determind which feature are importand to our model
- keep what feature are more meaningfull to our model
- remove what doesn't contribute to better decission model
- try another feature then run again
- as current selected here mostly general question, different result migth come if more specific feature feed into model


```python
# confusion matrix
# create
cm = metrics.confusion_matrix(ytest, ypred_rfc)
print(cm)
print(cm.shape)

# vis
sns.heatmap(cm)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
```

- our model QUITE ACCURATE to predict given input as Satisfied
- but LESS ACCURATE to predict Dissatisfied and Neither Satisfied or Disatisfied
- and MORE LIKELY to predict Dissatisfied ans Neither Satisfied or Disatisfied as Satisfied


```python
#### Out-Out-Sample Prediction
## map input
# age
age18_24 = [1,0,0,0,0,0,0]
age25_34 = [0,1,0,0,0,0,0]
age35_44 = [0,0,1,0,0,0,0]
age45_54 = [0,0,0,1,0,0,0]
age55_64 = [0,0,0,0,1,0,0]
age65_older = [0,0,0,0,0,1,0]
age_under18 = [0,0,0,0,0,0,1]
# gender
gender_f = [1,0,0]
gender_m = [0,1,0]
gender_nb = [0,0,1]
# hobby
hobby_y = [1,0]
hobby_n = [0,1]
# hours on computer
hour1_4 = [1,0,0,0,0]
hour5_8 = [0,1,0,0,0]
hour9_12 = [0,0,1,0,0]
hour_lessThan1 = [0,0,0,1,0]
hour_over12 = [0,0,0,0,1]
# hour outside
out1_2 = [1,0,0,0,0]
out3_4 = [0,1,0,0,0]
out30_50mnt = [0,0,1,0,0]
out_30mntLess = [0,0,0,1,0]
out_over4 = [0,0,0,0,1]
# years coding
years0_2 = [1,0,0,0]
years10_orMore = [0,1,0,0]
years2_5 = [0,0,1,0]
years5_10 = [0,0,0,1]
# time fully productive
product_lessOne = [1,0,0,0]
product_1_3 = [0,1,0,0]
product_3_6 = [0,0,1,0]
product_6_more = [0,0,0,1]
# exercise
exe_1_2_week = [1,0,0,0]
exe_3_4_week = [0,1,0,0]
exe_daily = [0,0,1,0]
exe_no = [0,0,0,1]
# employment
fullTime = [1, 0]
partTime = [0, 1]
```


```python
# test & predict proba
# age, gender, hobby, hour computer, hour outside, years coding, time productive, exercise
sample1 = age25_34 + gender_m + hobby_y + hour1_4 + out1_2 + years0_2 + product_1_3 + exe_no + partTime

y_oos = rfc.predict([sample1])[0]
y_oos_proba = rfc.predict_proba([sample1])[0]
status = target_name[y_oos]
percent_status = round(y_oos_proba[y_oos], 3) * 100

print(y_oos)
print(y_oos_proba)
print(f'sample1 may {status} with {percent_status} %')
```
