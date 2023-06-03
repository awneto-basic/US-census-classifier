import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

train_path = r"us_census_full/census_income_learn.csv"
test_path = r"us_census_full/census_income_test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

features = [
'age',
'class of worker',
'detailed industry recode',
'detailed occupation recode',
'education',
'wage per hour',
'enroll in edu inst last wk',
'marital stat',
'major industry code',
'major occupation code',
'race',
'hispanic origin',
'sex',
'member of a labor union',
'reason for unemployment',
'full or part time employment stat',
'capital gains',
'capital losses',
'dividends from stocks',
'tax filer stat',
'region of previous residence',
'state of previous residence',
'detailed household and family stat',
'detailed household summary in household',
'weight',
'migration code-change in msa',
'migration code-change in reg',
'migration code-move within reg',
'live in this house 1 year ago',
'migration prev res in sunbelt',
'num persons worked for employer',
'family members under 18',
'country of birth father',
'country of birth mother',
'country of birth self',
'citizenship',
'own business or self employed',
'fill inc questionnaire for veterans admin',
'veterans benefits',
'weeks worked in year',
'year',
'>50k'
]

train_data.columns = features
test_data.columns = features

print(train_data.head())

# DATA EXPLORATION

# 0 - class balance
print(train_data[">50k"].value_counts())
'''
>50k
-50000      187140
 50000+.     12382
 
 The target class is heavily unbalanced in the training set
'''

# 0 Data types
print(train_data.dtypes)
'''
>50k
-50000      187140
 50000+.     12382
Name: count, dtype: int64
age                                            int64
class of worker                               object
detailed industry recode                       int64
detailed occupation recode                     int64
education                                     object
wage per hour                                  int64
enroll in edu inst last wk                    object
marital stat                                  object
major industry code                           object
major occupation code                         object
race                                          object
hispanic origin                               object
sex                                           object
member of a labor union                       object
reason for unemployment                       object
full or part time employment stat             object
capital gains                                  int64
capital losses                                 int64
dividends from stocks                          int64
tax filer stat                                object
region of previous residence                  object
state of previous residence                   object
detailed household and family stat            object
detailed household summary in household       object
weight                                       float64
migration code-change in msa                  object
migration code-change in reg                  object
migration code-move within reg                object
live in this house 1 year ago                 object
migration prev res in sunbelt                 object
num persons worked for employer                int64
family members under 18                       object
country of birth father                       object
country of birth mother                       object
country of birth self                         object
citizenship                                   object
own business or self employed                  int64
fill inc questionnaire for veterans admin     object
veterans benefits                              int64
weeks worked in year                           int64
year                                           int64
>50k                                          object
dtype: object
'''

# combining capital gains and capital losses into a net capital gain feature
train_data['net capital gains'] = train_data['capital gains'] - train_data['capital losses']
test_data['net capital gains'] = test_data['capital gains'] - test_data['capital losses']


# removing features deemed to be redundant / irrelevant
features_to_remove = ["detailed industry recode","detailed occupation recode","weight","year",
                      "region of previous residence","enroll in edu inst last wk", "tax filer stat",
                      "detailed household and family stat","migration code-change in msa","migration code-change in reg",
                      "migration code-move within reg","live in this house 1 year ago", "migration prev res in sunbelt",
                      "capital gains","capital losses","hispanic origin","member of a labor union",
                      "reason for unemployment", "state of previous residence", "detailed household summary in household",
                      "country of birth father","country of birth mother", "country of birth self",
                      "major industry code","fill inc questionnaire for veterans admin","veterans benefits"]
train_data = train_data.drop(columns=features_to_remove)
test_data = test_data.drop(columns=features_to_remove)

print(train_data.dtypes)


# selecting the target class

train_target = train_data[">50k"]
train_data = train_data.drop(columns=[">50k"])

test_target = test_data[">50k"]
test_data = test_data.drop(columns=[">50k"])


# DATA PRE-PROCESSING


from sklearn.compose import make_column_selector as selector

num_cols_selector = selector(dtype_exclude=object)
cat_cols_selector = selector(dtype_include=object)

num_cols = num_cols_selector(train_data)
cat_cols = cat_cols_selector(train_data)
cat_cols.remove("education")
ordinal_cols = ["education"]

print(f"Quantitative features: {num_cols}")
print(f"Nominal categorical features: {cat_cols}")
print(f"Ordinal categorical features: {ordinal_cols}")


from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

cat_preprocessor = OneHotEncoder(handle_unknown="ignore")
num_preprocessor = StandardScaler()

education_levels = [' Children', ' Less than 1st grade', ' 1st 2nd 3rd or 4th grade', ' 5th or 6th grade',
    ' 7th and 8th grade', ' 9th grade', ' 10th grade', ' 11th grade', ' 12th grade no diploma',
    ' High school graduate', ' Some college but no degree', ' Associates degree-occup /vocational',
    ' Associates degree-academic program', ' Prof school degree (MD DDS DVM LLB JD)',
    ' Bachelors degree(BA AB BS)', ' Masters degree(MA MS MEng MEd MSW MBA)', ' Doctorate degree(PhD EdD)']
# note: children get assigned a level of zero

ord_preprocessor = OrdinalEncoder(categories=[education_levels])

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", cat_preprocessor, cat_cols),
        ("standard-scaler", num_preprocessor, num_cols),
        ("ordinal-encoder", ord_preprocessor, ordinal_cols)
    ]
)

# Building and training models

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Model 1 - Baseline model (most frequent category)
model_1 = Pipeline(
    [
        ("pre-processor",preprocessor),
        ("classifier", DummyClassifier(strategy="most_frequent"))
    ]
)

model_1.fit(train_data,train_target)
score = model_1.score(test_data, test_target)
print(f"Accuracy of the dummy classifier: {score:.3f}")
# Accuracy of the dummy classifier: 0.000


# Model 2 - LogisticRegression
model_2 = Pipeline(
    [
        ("pre-processor",preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)
model_2.fit(train_data,train_target)
score = model_2.score(test_data, test_target)
print(f"Accuracy of the Logistic Regression classifier: {score:.3f}")


# Model 3 - DecisionTreeClassifier

model_3 = Pipeline(
    [
        ("pre-processor",preprocessor),
        ("classifier", DecisionTreeClassifier())
    ]
)
model_3.fit(train_data,train_target)
score = model_3.score(test_data, test_target)
print(f"Accuracy of the DecisionTreeClassifier: {score:.3f}")

# Model 4 - HistGradientBoostingClassifier
'''
model_4 = Pipeline(
    [
        ("pre-processor",preprocessor),
        ("classifier", HistGradientBoostingClassifier())
    ]
)
model_4.fit(train_data,train_target)
score = model_4.score(test_data, test_target)
print(f"Accuracy of the HistGradientBoostingClassifier: {score:.3f}")
'''

# Model 5 - RandomForestClassifier
model_5 = Pipeline(
    [
        ("pre-processor",preprocessor),
        ("classifier", RandomForestClassifier())
    ]
)
model_5.fit(train_data,train_target)
score = model_5.score(test_data, test_target)
print(f"Accuracy of the RandomForestClassifier: {score:.3f}")


# Model 6 - Support vector classifier
model_6 = Pipeline(
    [
        ("pre-processor",preprocessor),
        ("classifier", SVC())
    ]
)
model_6.fit(train_data,train_target)
score = model_6.score(test_data, test_target)
print(f"Accuracy of the Support vector classifier: {score:.3f}")