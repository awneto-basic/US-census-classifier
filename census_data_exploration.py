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
features_to_remove = ["detailed industry recode", "detailed occupation recode", "weight", "year",
                      "region of previous residence", "enroll in edu inst last wk", "tax filer stat",
                      "detailed household and family stat", "migration code-change in msa",
                      "migration code-change in reg",
                      "migration code-move within reg", "live in this house 1 year ago",
                      "migration prev res in sunbelt",
                      "capital gains", "capital losses"]
train_data.drop(columns=features_to_remove)
test_data.drop(columns=features_to_remove)

# 1 - Histogram

hist_features = ['age','wage per hour']

for f in hist_features:

    plt.figure(figsize=(8, 6))
    sns.histplot(data=train_data, x=f, hue='>50k', kde=True)
    plt.title(f'Distribution of {f} by Income')
    plt.xlabel(f)
    plt.ylabel('Count')
    plt.legend(title='Income', labels=['<=50k', '>50k'])
    plt.show()

# 2 - Bar plots

bar_features = ['class of worker', 'education', 'marital stat', 'race', 'sex', 'own business or self employed']
plt.figure(figsize=(15, 10))
for i, f in enumerate(bar_features):
    plt.subplot(3, 2, i + 1)  # 3x2 grid
    sns.countplot(data=train_data, y=f, hue='>50k')
    plt.title(f'Frequency of {f} by Income')
    plt.xlabel("Count")
    plt.ylabel(f)
    plt.legend(title='Income', labels=['<=50k', '>50k'])

plt.tight_layout()
plt.show()

f = "major occupation code"
plt.figure(figsize=(8, 6))
sns.countplot(data=train_data, y=f, hue='>50k')
plt.title(f'Frequency of {f} by Income')
plt.xlabel("Count")
plt.ylabel(f)
plt.legend(title='Income', labels=['<=50k', '>50k'])
plt.show()

# 3 - violin plots
v_features = ['age', 'wage per hour', 'net capital gains', 'weeks worked in year']
plt.figure(figsize=(15, 10))

for i, f in enumerate(v_features):
    plt.subplot(2, 2, i + 1)  # 2x2 grid
    sns.violinplot(data=train_data, x='>50k', y=f)
    plt.title(f'Distribution of {f} by Income')
    plt.xlabel('Income')
    plt.ylabel(f)

plt.tight_layout()
plt.show()

# 4 - pairplots (of a sample of the training set)

f = ["age", "wage per hour", "weeks worked in year", "net capital gains"]
sns.pairplot(data=train_data[:5000], vars=f,
             hue=">50k")  # only take data from the first 5000 samples to improve the readability of the plot
plt.show()

f = ["net capital gains", "dividends from stocks"]
sns.pairplot(data=train_data[:5000], vars=f,
             hue=">50k")  # only take data from the first 5000 samples to improve the readability of the plot
plt.show()

