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

# 1 Histograms

hist_features =  ['age']
for hist_feature in hist_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=train_data, x=hist_feature, hue='>50k', kde=True)
    plt.title(f'Distribution of {hist_feature} by Income')
    plt.xlabel(hist_feature)
    plt.ylabel('Count')
    plt.legend(title='Income', labels=['<=50k', '>50k'])
    plt.show()

# 2 Bar plot

bar_features = ['class of worker', 'education', 'marital stat', 'race', 'sex', 'citizenship', 'own business or self employed']


for f in bar_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_data, y=f, hue='>50k')
    plt.title(f'Frequency of {f} by Income')
    plt.xlabel("Count")
    plt.ylabel(f)
    plt.legend(title='Income', labels=['<=50k', '>50k'])
    plt.show()

# 3 violin plots
v_features = ['age', 'wage per hour', 'capital gains', 'weeks worked in year']

for f in v_features:
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=train_data, x='>50k', y=f)
    plt.title(f'Distribution of {f} by Income')
    plt.xlabel('Income')
    plt.ylabel(f)
    plt.show()


input("press any key to continue...")

train_target = train_data[">50k"]
train_data = train_data.drop(columns=[">50k"])

test_target = test_data[">50k"]
test_data = test_data.drop(columns=[">50k"])

from sklearn.compose import make_column_selector as selector

num_cols_selector = selector(dtype_exclude=object)
cat_cols_selector = selector(dtype_include=object)

num_cols = num_cols_selector(train_data)
cat_cols = cat_cols_selector(train_data)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

cat_preprocessor = OneHotEncoder(handle_unknown="ignore")
num_preprocessor = StandardScaler()

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    [
        ("one-hot-encoder", cat_preprocessor, cat_cols),
        ("standard-scaler", num_preprocessor, num_cols)
    ]
)