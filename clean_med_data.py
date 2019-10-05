# Import packages
import os
import pandas as pd
from pandas import set_option
import numpy as np

import warnings
warnings.simplefilter(action='ignore')


def drop_exclude_cols(df, exclude_cols):

    exclude_cols = "|".join(exclude_cols)

    df.drop(df.filter(regex = exclude_cols, axis = 1).columns,
            axis = 1,
            inplace = True)

    return df


def tidy_columns(df):

    # Tidy column to keep
    df.columns = df.columns.str.replace(" ", "_").str.lower()

    return df


def clean_na_values(df, na_to_clean = list):

    # Assign NA values
    for na in na_to_clean:

        df = df.replace(na, np.nan)

    return df

def save_cleaned_csv(df, csv_path: str):

    df.to_csv(f"{csv_path.split('.')[0]}_cleaned.csv",
              index = False)


def format_bool_to_int(df,convert_additional_cols):

    for c in df.select_dtypes(include='bool').columns:
        df[c] = df[c].fillna(3)
        df[c] = df[c].astype(int)

    if convert_additional_cols:
        for c in convert_additional_cols:
            df[c] = df[c].fillna(3)
            df[c] = df[c].astype(int)

    return df


def format_hospital_comparisons_cols(df):

    comparison_dict = {
    'Below the National average': 1,
    'Same as the National average': 2,
    'Above the National average':3 }

    df = df.replace(comparison_dict)

    return df


def pivot_readmission_types(df_in):

    agg_cols = ["number_of_readmissions",
                "number_of_discharges",
                "excess_readmission_ratio",
                "predicted_readmission_rate",
                "expected_readmission_rate"]

    df_agg = df_in.loc[:,["provider_number"]].drop_duplicates()

    for col in agg_cols:
        df_temp = pd.pivot_table(data = df_in,
             index=['provider_number'],
             columns = ["measure_name"],
             values= col,
            aggfunc = np.sum).reset_index()

        update_cols = \
        ["provider_number"] + \
        [f"{c.lower().replace('-', '_')}_{col}" \
        for c in df_temp.columns if "provider_number" not in c]

        df_temp.columns = update_cols

        df_agg = pd.merge(df_agg, df_temp, on = "provider_number", how = "inner")

    df_in.drop(agg_cols + ["measure_name"], axis = 1, inplace = True)
    df_out = pd.merge(df_in, df_agg, on = "provider_number", how = "inner"). \
         drop_duplicates(["provider_number"])

    df_out.replace({"Too Few to Report":0}, inplace = True)

    return df_out


def remove_all_cleaned_files(directory_path: str):

    cleaned_files = [f for f in os.listdir(directory_path) if "_cleaned" in f]

    for f in cleaned_files:
        os.remove(f)


def bin_states_to_region(directory_path: str):

    # Time zone dictionary 
    state_to_region = { 'AK': 'US/Alaska', 'AL': 'US/Central',
                       'AR': 'US/Central', 'AS': 'US/Samoa',
                       'AZ': 'US/Mountain', 'CA': 'US/Pacific',
                       'CO': 'US/Mountain', 'CT': 'US/Eastern',
                       'DC': 'US/Eastern', 'DE': 'US/Eastern',
                       'FL': 'US/Eastern', 'GA': 'US/Eastern',
                       'GU': 'Pacific/Guam', 'HI': 'US/Hawaii',
                       'IA': 'US/Central', 'ID': 'US/Mountain',
                       'IL': 'US/Central', 'IN': 'US/Eastern',
                       'KS': 'US/Central', 'KY': 'US/Eastern',
                       'LA': 'US/Central', 'MA': 'US/Eastern',
                       'MD': 'US/Eastern', 'ME': 'US/Eastern',
                       'MI': 'US/Eastern', 'MN': 'US/Central',
                       'MO': 'US/Central', 'MP': 'Pacific/Guam',
                       'MS': 'US/Central', 'MT': 'US/Mountain',
                       'NC': 'US/Eastern', 'ND': 'US/Central',
                       'NE': 'US/Central', 'NH': 'US/Eastern',
                       'NJ': 'US/Eastern', 'NM': 'US/Mountain',
                       'NV': 'US/Pacific', 'NY': 'US/Eastern',
                       'OH': 'US/Eastern', 'OK': 'US/Central',
                       'OR': 'US/Pacific', 'PA': 'US/Eastern',
                       'PR': 'America/Puerto_Rico', 'RI': 'US/Eastern',
                       'SC': 'US/Eastern','SD': 'US/Central',
                       'TN': 'US/Central', 'TX': 'US/Central',
                       'UT': 'US/Mountain', 'VA': 'US/Eastern',
                       'VI': 'America/Virgin','VT': 'US/Eastern',
                       'WA': 'US/Pacific', 'WI': 'US/Central',
                       'WV': 'US/Eastern',  'WY': 'US/Mountain',
                       '' : 'US/Pacific',  '--': 'US/Pacific' }
    
    state_to_region_df = pd.DataFrame(state_to_region.items())
    state_to_region_df.columns = ['state', 'region']

    # Merge time zones with dataframe and clean data types
    df = pd.read_csv(directory_path)
    
    df = pd.merge(left = df ,
                  right = state_to_region_df,
                  how = 'left',
                  on = 'state')

    # Save cleaned csv
    df.to_csv('med_data_merged_geo.csv', index = False)


def clean_general_info(csv_path: str,
                      exclude_cols: list,
                      na_to_clean: list,
                      convert_additional_cols: list):
    # Import csv
    df = pd.read_csv(csv_path)

    df = tidy_columns(df)
    
    df = drop_exclude_cols(df,exclude_cols)
    
    # Assign NA values
    df = clean_na_values(df, na_to_clean)
    
    df = format_bool_to_int(df, convert_additional_cols)
    
    df = format_hospital_comparisons_cols(df)
    
    # Save cleaned csv
    save_cleaned_csv(df, csv_path)


def clean_general_info(csv_path: str,
                      exclude_cols: list,
                      na_to_clean: list,
                      convert_additional_cols: list):
    # Import csv
    df = pd.read_csv(csv_path)

    df = tidy_columns(df)

    df = drop_exclude_cols(df,exclude_cols)

    # Assign NA values
    df = clean_na_values(df, na_to_clean)

    df = format_bool_to_int(df, convert_additional_cols)

    df = format_hospital_comparisons_cols(df)

    # Save cleaned csv
    save_cleaned_csv(df, csv_path)


def clean_mspb_info(csv_path: str,
                      exclude_cols: [],
                      na_to_clean: []):

    # Import csv
    df = pd.read_csv(csv_path)

    df = tidy_columns(df)

    df = drop_exclude_cols(df,exclude_cols)

    # Assign NA values
    df = clean_na_values(df, na_to_clean)

    # Save cleaned csv
    save_cleaned_csv(df, csv_path)


def clean_readmissions_info(csv_path: str,
                            exclude_cols: list,
                           na_to_clean = list):
    # Import csv
    df = pd.read_csv(csv_path)
    
    df = tidy_columns(df)
    
    df = drop_exclude_cols(df,exclude_cols)
    
    # pivot and aggregatate readmissions data by clinical area
    df = pivot_readmission_types(df)
    
    # Assign NA values
    df = clean_na_values(df, na_to_clean)
    
    # Save cleaned csv
    save_cleaned_csv(df,csv_path)


def merge_clean_tables(directory_path: str):

    cleaned_files = [f for f in os.listdir(directory_path) if "_cleaned" in f]
    df = pd.read_csv(cleaned_files[0])

    for table in cleaned_files[1:]:
        table = pd.read_csv(table)
        table.rename(columns ={"provider_number":"provider_id"}, inplace = True)
        df = pd.merge(df, table, on = "provider_id", how = "outer")

    df.to_csv("med_data_merged.csv", index = False)


remove_all_cleaned_files(os.getcwd())

clean_general_info("Hospital_General_Information.csv",
                  exclude_cols = ["footnote",
                                  "measure_id",
                                  "start_date",
                                  "end_date",
                                  "hospital_name",
                                  "zip_code",
                                  "location",
                                  "address",
                                  "phone_number",
                                  "city",
                                 "county_name"],
                  na_to_clean = ["Not Available"],
                  convert_additional_cols = \
                   ["meets_criteria_for_meaningful_use_of_ehrs"])


clean_mspb_info("Medicare_hospital_spending_per_patient__" \
                     "Medicare_Spending_per_Beneficiary____Additional_Decimal_Places.csv",
                    exclude_cols = ["footnote",
                                    "location",
                                    "measure_id",
                                    "start_date",
                                    "end_date",],
                na_to_clean = ["Not Available"])

clean_readmissions_info("Hospital_Readmissions_Reduction_Program.csv",
                       exclude_cols = ["footnote",
                                       "start_date",
                                       "end_date",
                                       "hospital_name", 
                                       "state",
                                       "region"],
                       na_to_clean = ["Not Available", "Too Few to Report"])

merge_clean_tables(os.getcwd())

bin_states_to_region("med_data_merged.csv")
