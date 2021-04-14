## 데이터 로드
import os
import pandas as pd
import numpy as np

data_dir = "./datasets/titanic"
train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))

## 데이터 준비
train_x = train_data.drop("Survived", axis=1)
train_y = train_data["Survived"].copy()

## pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DropAttribs(BaseEstimator, TransformerMixin):  # "PassengerId", "Name", "Ticket", "Cabin"
    def __init__(self, attribs_lst):
        self.attribs_lst = attribs_lst

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.drop(self.attribs_lst, axis=1)


other_attribs = ["PassengerId", "Name", "Ticket", "Cabin"]
imputer_attribs = ["Age", "Embarked", "Fare"]
cat_attribs = ["Sex"]
num_attribs = list(train_x.drop(cat_attribs + imputer_attribs + other_attribs, axis=1))

pipeline_age = Pipeline(steps=[("imputer", SimpleImputer()),
                               ("num", StandardScaler())])
pipeline_embarked = Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="S")),
                                    ("cat", OneHotEncoder())])
pipeline_fare = Pipeline(steps=[("imputer", SimpleImputer()),
                                    ("num", StandardScaler())])

pipeline_column = ColumnTransformer(transformers=[("age", pipeline_age, ['Age']),
                                           ("embarked", pipeline_embarked, ["Embarked"]),
                                           ("fare", pipeline_fare, ["Fare"]),
                                           ("cat", OneHotEncoder(), cat_attribs),
                                           ("num", StandardScaler(), num_attribs)])

pipeline = Pipeline(steps=[("drop", DropAttribs(other_attribs)),
                           ("column", pipeline_column)])


train_x_transformed = pipeline.fit_transform(train_x)

## val 테스트
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

scores = cross_val_score(SVC(), train_x_transformed, train_y,
                         cv=5, scoring="accuracy")

## 트레이닝
from sklearn.svm import SVC

svc_clf = SVC()
svc_clf.fit(train_x_transformed, train_y)

## 최종 예측
test_x_transformed = pipeline.transform(test_data)
y_pred = svc_clf.predict(test_x_transformed)

##
result_dir = os.path.join(data_dir, "result")
os.makedirs(result_dir, exist_ok=True)

gs = pd.read_csv(os.path.join(data_dir, "gender_submission.csv"))
submission = pd.DataFrame({'PassengerId': gs.PassengerId, 'Survived': y_pred})
submission.to_csv(os.path.join(result_dir, 'my_submission.csv'),
                  index=False)
