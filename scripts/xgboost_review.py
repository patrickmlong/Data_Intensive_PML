# Import packages
import os
import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn.model_selection import train_test_split

from yellowbrick.features import FeatureImportances
import shap
import xgboost as xgb

# Plotting packages
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['xtick.major.pad'] = 1
rcParams['ytick.major.pad'] = 1

import warnings
warnings.simplefilter(action='ignore')


def format_input(dir_path: str):
    
    # Import cleaned data 
    df = pd.read_csv(dir_path)

    # Omit hospital overall rating as it
    # is derived from target.
    df.drop(["hospital_overall_rating"], 
            axis = 1, inplace = True)

    # Drop provider_ids where target variable is NaN
    df = df[~df.patient_experience_national_comparison.isna()]

    # Assign target
    target = df.patient_experience_national_comparison
    
    provider_id = df.provider_id
    
        # Remove target from training
    df.drop(["patient_experience_national_comparison",
             "provider_id"],
            axis = 1, inplace = True)
    

    return df, target, provider_id


def get_dummies_wrapper(df_in):

    feature_list = df_in.select_dtypes(include = "object").columns
    
    dummy_df = pd.get_dummies(df_in[feature_list])
    
    df_in.drop(feature_list, axis = 1, inplace = True)
    
    df_out = pd.concat([df_in,dummy_df], axis = 1)
    
    return df_out

X, Y, provider_id = format_input("../data/processed_data/med_data_merged_geo.csv")

X = get_dummies_wrapper(X)

# Split to train:test (80:20)
X_train, X_test, y_train, y_test = \
train_test_split(X, Y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBClassifier(random_state = 42,
                              max_depth = 5)

print(xgb_model.fit(X_train, y_train, early_stopping_rounds = 10, eval_set = [(X_test, y_test)]))

def make_feature_impt_df(model, X, X_train, y_train, y_test):
    fi = FeatureImportances(model, labels = X.columns)
    fi.fit(X_train, y_train)
    imps = pd.Series(fi.feature_importances_)
    features = pd.Series(fi.features_)
    
    df = pd.DataFrame()
    df["importances"] = imps
    df["features"] = features
    
    df.sort_values(by = ["importances"],
                  ascending = False,
                  inplace = True)
    
    return df

df_xgb_features = \
make_feature_impt_df(xgb_model, X, X_train, y_train, y_test)

print(df_xgb_features[df_xgb_features.importances > 0])


booster = xgb_model.get_booster()
print(booster.get_dump()[0])
print(booster.get_dump()[1])
print(booster.get_dump()[2])


def save_shap_fig(model, png_name: str):
    
    s = shap.TreeExplainer(model)
    fig, ax = plt.subplots()
    shap_vals = s.shap_values(X_test)
    shap.summary_plot(shap_vals, X_test)
    fig.savefig("../results/{png_name}", 
                dpi = 300, bbox_inches = "tight")
    
save_shap_fig(xgb_model, "xgb_shap_plot.png")  
    
