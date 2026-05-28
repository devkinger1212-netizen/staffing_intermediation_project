# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:25:13 2026

@author: New
"""
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Load datasets
df = pd.read_stata(r"amr_GS_appended.dta")
df2 = pd.read_stata(r"amr_GS_appended_temp_payroll.dta")

# Merge primary and payroll datasets on employee code
df['ecode'] = df['ecode'].astype(str)
df2['ecode'] = df2['ecode'].astype(str)
df3 = pd.merge(df, df2, on='ecode', how='left')
print("", df3.columns.tolist()) #columns in the merged file

# Check for missing values in core date variables
df['amr_doj'].isna().value_counts() #no misssing values
df['amr_actualdol'].isna().value_counts()

# Convert variables to datetime and verify samples
df['amr_doj'] = pd.to_datetime(df['amr_doj'])
df['amr_actualdol'] = pd.to_datetime(df['amr_actualdol'])
df['amr_doj'].sample(10)
df['amr_actualdol'].sample(10)

# Compare contract end dates with actual dates of leaving
df3["amr_status"].sample(10)
df3["amr_contractend"].sample(10)

df3[["amr_contractend","amr_actualdol"]].sample(10)
completed_contracts = df[
    (df['amr_actualdol'].notna()) &
    (df['amr_actualdol'] == df['amr_contractend'])
]

# Output completed contracts count
completion_count = len(completed_contracts)
print(f"TOTAL COMPLETED CONTRACTS FOUND (Event = 0): {completion_count}")
(df3['amr_actualdol'] <= df3['amr_contractend']).all()
(df3['amr_actualdol'] == df3['amr_contractend']).sum()
df3["amr_contractend"].isna().value_counts()

# --- Task 1: Overall Survival Curve ---

# Conver required columns to datetime
df['amr_doj'] = pd.to_datetime(df['amr_doj'])
df['amr_actualdol'] = pd.to_datetime(df['amr_actualdol'])
df['amr_contractend'] = pd.to_datetime(df['amr_contractend'])

# Define exit event flag (0 = Active/Completed, 1 = Exited)
df['Event'] = 0

# Identify premature departures
left_early = (df['amr_actualdol'].notna()) & (df['amr_actualdol'] < df['amr_contractend'])

# Identify departures with missing contract end dates
left_no_contract = (df['amr_actualdol'].notna()) & (df['amr_contractend'].isna())

# Apply churn flag based on departure criteria
df.loc[left_early | left_no_contract, 'Event'] = 1

# Calculate tenure in days
study_end_date = df['amr_actualdol'].max()

# Impute end date for active/completed records using study end date
df['end_date'] = df['amr_actualdol'].fillna(study_end_date)
df['Tenure_Days'] = (df['end_date'] - df['amr_doj']).dt.days

# Check for data anomalies (negative tenure)
anomalies = df[df['Tenure_Days'] < 0]
error_count = len(anomalies)
print(f"No. of negative tenures: {error_count}")

# Fit overall Kaplan-Meier model
kmf = KaplanMeierFitter()
kmf.fit(durations=df['Tenure_Days'], event_observed=df['Event'], label='Overall Retention')

# Plot overall survival curve
kmf.plot_survival_function(color='#2E86AB', linewidth=2)
plt.title('Overall Survival Curve')
plt.xlabel('Tenure (Days)')
plt.ylabel('Proportion Still Active')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim([0, 1.05])
plt.xlim([0, 365])
plt.show()

# --- Task 2: Survival Curve - Age ---

# Clean and calculate age distribution
df['amr_dob'].isna().value_counts()
df['amr_dob'] = pd.to_datetime(df['amr_dob'])
df['Age'] = (df['amr_doj'] - df['amr_dob']).dt.days / 365
print("AGE DISTRIBUTION")
print(df['Age'].describe())

# Filter outliers and plot age histogram
df = df[(df['Age'] <= 80)]
plt.hist(df['Age'].dropna(), bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Joiners')
plt.xlabel('Age (Years)')
plt.ylabel('Number of Workers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Segment age into categorical bins
age_bins = [16, 25, 35, 45, 100]
age_labels = ['16-24', '25-34', '35-44', '45+']
df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

# Remove records missing age groupings
df_age = df.dropna(subset=['Age_Group'])

# Plot stratified survival curve by age group
fig, ax = plt.subplots(figsize=(10, 6))

for name, grouped_df in df_age.groupby('Age_Group'):
    kmf_age = KaplanMeierFitter()
    kmf_age.fit(
        durations=grouped_df['Tenure_Days'],
        event_observed=grouped_df['Event'],
        label=f'Age: {name}'
    )
    kmf_age.plot_survival_function(ax=ax, linewidth=2)

plt.title('Kaplan-Meier Survival Curve by Age Group')
plt.xlabel('Tenure (Days)')
plt.ylabel('Proportion Still Active')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim([0, 1.05])
plt.xlim([0, 365])
plt.show()

# --- Task 3: Survival Curve - Gender ---

# Analyze and standardize gender variable
df['amr_gender'].isna().value_counts()
df['amr_gender'].unique()
df['amr_gender'] = df['amr_gender'].replace('', pd.NA)
print(df['amr_gender'].value_counts(dropna=False))

# Prepare dataframe and remove null gender records
df_plot = df.copy()
df_plot['amr_gender'] = df_plot['amr_gender'].astype(str).str.strip()
df_plot = df_plot[~df_plot['amr_gender'].isin(['<NA>'])]

# Plot stratified survival curve by gender
fig, ax = plt.subplots(figsize=(10, 6))
for gender_name, grouped_df in df_plot.groupby('amr_gender'):
    kmf_gender = KaplanMeierFitter()
    kmf_gender.fit(
        durations=grouped_df['Tenure_Days'],
        event_observed=grouped_df['Event'],
        label=f'Gender: {gender_name}'
    )
    kmf_gender.plot_survival_function(ax=ax, linewidth=2)

plt.title('Kaplan-Meier Survival Curve by Gender')
plt.xlabel('Tenure (Days)')
plt.ylabel('Proportion Still Active')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim([0, 1.05])
plt.xlim([0, 365])
plt.show()

# --- Task 4: Survival Curve - Wage ---

# Standardize net pay variable to numeric
df['amr_netpay'].isna().value_counts()
df['amr_netpay'] = pd.to_numeric(df['amr_netpay'], errors='coerce')
print(df['amr_netpay'].describe())

# Remove top and bottom 1% to handle extreme outliers
lower_bound = df['amr_netpay'].quantile(0.01)
upper_bound = df['amr_netpay'].quantile(0.99)
df_wage = df[(df['amr_netpay'] >= lower_bound) & (df['amr_netpay'] <= upper_bound)].copy()

# Plot net pay distribution
plt.hist(df_wage['amr_netpay'].dropna(), bins=40, color='#2CA02C', edgecolor='black', alpha=0.7)
plt.title('Distribution of Net Pay (Take-Home)')
plt.xlabel('Net Pay Amount')
plt.ylabel('Number of Workers')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Segment wages into quartile
wage_labels = ['Bottom 25% Pay', 'Lower Mid 25%', 'Upper Mid 25%', 'Top 25% Pay']
df_wage['Wage_Tier'] = pd.qcut(df_wage['amr_netpay'], q=4, labels=wage_labels)

# Plot survival curve by wage
fig, ax = plt.subplots(figsize=(10, 6))

for tier_name, grouped_df in df_wage.groupby('Wage_Tier'):
    kmf_wage = KaplanMeierFitter()
    kmf_wage.fit(
        durations=grouped_df['Tenure_Days'],
        event_observed=grouped_df['Event'],
        label=f'Wage: {tier_name}'
    )
    kmf_wage.plot_survival_function(ax=ax, linewidth=2)

plt.title('Kaplan-Meier Survival Curve by Net Pay Tiers')
plt.xlabel('Tenure (Days)')
plt.ylabel('Proportion Still Active')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim([0, 1.05])
plt.xlim([0, 365])
plt.show()

# --- Task 5: Survival Curve - Client Size ---

# Clean client names and handle missing values
df['amr_clientname'].sample(10)
initial_missing = df['amr_clientname'].isna().sum()
print(f"Initial blanks (NaNs) in the database: {initial_missing}")

df['Client_Clean'] = df['amr_clientname'].astype(str).str.strip().str.upper()
bad_strings = ['NAN', 'NONE', '', '<NA>', 'NULL']
df.loc[df['Client_Clean'].isin(bad_strings), 'Client_Clean'] = None

total_missing = df['Client_Clean'].isna().sum()
print(f"Total missing values after text cleaning: {total_missing}")

# Analyze client distribution
unique_count = df['Client_Clean'].nunique()
print(f"\nTotal Unique Clients Found: {unique_count}")
client_headcounts = df['Client_Clean'].value_counts()

print("\n--- TOP 15 CLIENTS BY PLACEMENT VOLUME ---")
print(client_headcounts.head(15))

# Calculate client concentration risk metric
total_workers = len(df)
top_10_workers = client_headcounts.head(10).sum()
concentration = (top_10_workers / total_workers) * 100
print(f"\nBusiness Insight: Top 10 clients account for {concentration:.1f}% of entire workforce.")

# Visualize top 20 clients by volume
top_20_clients = client_headcounts.head(20)
top_20_clients = top_20_clients.sort_values(ascending=True)

plt.figure(figsize=(12, 8))
ax = top_20_clients.plot(kind='barh', color='#E24A33', edgecolor='black', alpha=0.85)
plt.title('Top 20 Clients by Placement Volume', fontsize=14, fontweight='bold')
plt.xlabel('Number of Workers (Headcount)')
plt.ylabel('Client Name')
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Annotate bar charts with exact values
for index, value in enumerate(top_20_clients):
    plt.text(value + (max(top_20_clients)*0.01), index, str(value), va='center', fontsize=10)

plt.xlim(0, max(top_20_clients) * 1.1)
plt.tight_layout()
plt.show()

# Prepare client data for survival analysis
df_client = df.copy()
df_client['amr_clientname'] = df_client['amr_clientname'].astype(str).str.strip().str.upper()
bad_clients = ['NAN', 'NONE', '', '<NA>']
df_client = df_client[~df_client['amr_clientname'].isin(bad_clients)]

# Map aggregate headcounts back to individual records
client_headcounts = df_client['amr_clientname'].value_counts()
df_client['Client_Size'] = df_client['amr_clientname'].map(client_headcounts)

print("Top 5 Largest Clients by Headcount:")
print(client_headcounts.head(5))
print("\nCrunching survival math for Client Sizes...")

# Segment clients into size tiers
size_bins = [0, 50, 250, 1000, float('inf')]
size_labels = ['Small (1-50)', 'Medium (51-250)', 'Large (251-1000)', 'Enterprise (1000+)']

df_client['Client_Tier'] = pd.cut(
    df_client['Client_Size'],
    bins=size_bins,
    labels=size_labels,
    right=True
)

# Plot stratified survival curve by client tier
fig, ax = plt.subplots(figsize=(10, 6))

for tier_name, grouped_df in df_client.groupby('Client_Tier'):
    kmf_client = KaplanMeierFitter()
    kmf_client.fit(
        durations=grouped_df['Tenure_Days'],
        event_observed=grouped_df['Event'],
        label=f'Client Size: {tier_name}'
    )
    kmf_client.plot_survival_function(ax=ax, linewidth=2)

plt.title('Kaplan-Meier Survival Curve by Client Size (Headcount)')
plt.xlabel('Tenure (Days)')
plt.ylabel('Proportion Still Active')
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim([0, 1.05])
plt.xlim([0, 365])
plt.show()

# --- DRAWING TENURE HISTOGRAM ---
print("\n--- DRAWING TENURE HISTOGRAM ---")

# Plot standard distribution of tenure
plt.figure(figsize=(12, 6))
plt.hist(
    df['Tenure_Days'].dropna(),
    bins=range(0, 505, 5),
    color='#4C72B0',
    edgecolor='black',
    alpha=0.75
)

# Overlay vertical lines for key retention milestones
milestones = [7, 30, 90, 365]
line_colors = ['#E24A33', '#F0A30A', '#348ABD', '#8EBA42']
line_labels = ['7 Days (Week 1)', '30 Days (Month 1)', '90 Days (Quarter 1)', '365 Days (Year 1)']

for i in range(len(milestones)):
    plt.axvline(
        x=milestones[i],
        color=line_colors[i],
        linestyle='--',
        linewidth=2.5,
        label=line_labels[i]
    )

# Chart Formatting
plt.title('Distribution of Worker Tenure with Key Milestones', fontsize=14, fontweight='bold')
plt.xlabel('Tenure (Days)', fontsize=12)
plt.ylabel('Number of Workers', fontsize=12)
plt.xlim(0, 500)
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
      
