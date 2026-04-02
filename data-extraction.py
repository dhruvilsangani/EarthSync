# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (es_env)
#     language: python
#     name: es_env
# ---

# %%
import numpy as np
import pandas as pd

# %%
# 1. Load the data correctly
initial_data = pd.read_csv("../../Downloads/iex_dam_feb_mar_2026.csv")
new_data = pd.read_excel("../../Downloads/DAM_Market Snapshot (1).xlsx")

# %%
cleaned_new_data = new_data[4:].dropna().drop(columns = {"Unnamed: 1"}).copy().reset_index(drop=True)
cleaned_new_data.columns = ['period_start', 'period', 'purchase_bid', 'sell_bid', 'mcv', 'final_scheduled_volume', 'mcp']

# %%
cleaned_new_data['period_start'] = pd.to_datetime(cleaned_new_data['period_start'], dayfirst=True)
start_times = cleaned_new_data['period'].str.split(' - ').str[0]
cleaned_new_data['period_start'] = cleaned_new_data['period_start'] + pd.to_timedelta(start_times + ':00')
cleaned_new_data

# %%
initial_data['period_start'] = pd.to_datetime(initial_data['period_start'])
numeric_cols = ['purchase_bid', 'sell_bid', 'mcv', 'mcp', 'final_scheduled_volume']
for col in numeric_cols:
    cleaned_new_data[col] = pd.to_numeric(cleaned_new_data[col], errors='coerce')

combined_df = pd.concat([initial_data, cleaned_new_data], axis=0, ignore_index=True)

combined_df = combined_df.sort_values('period_start')
combined_df = combined_df.drop_duplicates(subset=['period_start'], keep='last')

combined_df = combined_df.reset_index(drop=True)

# %%
# 1. Get the start and end dates for the filename
first_date = combined_df['period_start'].min().strftime('%m%d')
last_date = combined_df['period_start'].max().strftime('%m%d')

# 2. Construct the dynamic filename
filename = f"iex-dam-{first_date}-{last_date}.csv"
save_path = f"../../Downloads/{filename}"

# 3. Save the file
combined_df.to_csv(save_path, index=False)

print(f"File saved successfully as: {filename}")

# %%
