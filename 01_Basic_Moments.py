# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:25:13 2026

@author: Dev
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

df = pd.read_stata(r"C:\Users\New\Desktop\Work\amr_GS_appended.dta")
df.to_parquet(r"C:\Users\New\Desktop\Work\amr_GS_appended.dta", engine='pyarrow')
df2 = pd.read_stata(r"C:\Users\New\Desktop\Work\amr_GS_appended_temp_payroll.dta")
df2.to_parquet(r"C:\Users\New\Desktop\Work\amr_GS_appended_temp_payroll1.dta", engine='pyarrow')

print("converted")

# Load datasets
cols = ['amr_emp_id', 'amr_gender', 'amr_dob', 'ecode', 'amr_contractend', 'amr_marital_status', 'amr_qualification',
    'amr_doj', 'amr_actualdol', 'amr_exitreason', 'amr_designation', 'amr_department', 
    'amr_jobcategory', 'amr_grade','amr_clientname', 'amr_joblocationpincode', 'amr_joblocationdistrict', 
    'amr_ptstate','amr_permanentaddress', 'amr_presentaddress', 'amr_mailingaddress',
    'amr_ctc', 'amr_gross', 'amr_status', 'amr_netpay', 'amr_basic', 'amr_stoppaystatus', 'amr_clientecode']
df = pd.read_parquet(r"C:\Users\New\Desktop\Staffing_project\amr_GS_appended.dta", columns = cols)
df2 = pd.read_parquet(r"C:\Users\New\Desktop\Staffing_project\amr_GS_appended_temp_payroll1.dta")
ecode_df = set(df['ecode'])
ecode_df2 = set(df2['ecode'])

df['amr_clientecode'].isna().value_counts()
df['amr_clientname'].isna().value_counts()
# Check for missing values in core date variables
df['amr_doj'].isna().value_counts() #no misssing values
df['amr_actualdol'].isna().value_counts()

# Convert variables to datetime and verify samples
df['amr_doj'] = pd.to_datetime(df['amr_doj'])
df['amr_actualdol'] = pd.to_datetime(df['amr_actualdol'])
df['amr_doj'].sample(10)
df['amr_actualdol'].sample(10)

df['amr_contractend'] = pd.to_datetime(df['amr_contractend'], errors='coerce')
df['amr_contractend'].dt.year.value_counts(dropna=False).sort_index()
df['amr_doj'].dt.year.value_counts().sort_index()

#-------------------------------------------------------------#
# --- Task 1a: Understanding Dates, status and exitreason ---
#-------------------------------------------------------------#

# ===========================================================================
# STEP 1: Standardize Dates & Filter for FY 2022+
# ===========================================================================
df['amr_doj'] = pd.to_datetime(df['amr_doj'], errors='coerce')
df['amr_actualdol'] = pd.to_datetime(df['amr_actualdol'], errors='coerce')
df['amr_contractend'] = pd.to_datetime(df['amr_contractend'], errors='coerce')

# Indian FY 2022 begins on April 1, 2022
FY22_START = pd.to_datetime('2022-04-01')
df_fy22 = df[df['amr_doj'] >= FY22_START].copy()

PROXY_PULL_DATE = df_fy22['amr_actualdol'].max()

print(f"Initial Rows in FY 2022+ Cohort: {len(df_fy22):,}")
print(f"Inferred Data Pull Date: {PROXY_PULL_DATE.date()}\n")

# ===========================================================================
# STEP 2: Calculate & Remove Chronological Anomalies
# ===========================================================================
print("ANOMALY REMOVAL: Chronological Integrity")
print("-" * 80)
mask_left_before_join = (df_fy22['amr_actualdol'].notna()) & (df_fy22['amr_actualdol'] < df_fy22['amr_doj'])
mask_contract_before_join = (df_fy22['amr_contractend'] < df_fy22['amr_doj'])

print(f"Dropped (Left Before Joining):      {mask_left_before_join.sum():,}")
print(f"Dropped (Contract Before Join):     {mask_contract_before_join.sum():,}")

is_anomaly = mask_left_before_join | mask_contract_before_join
df_clean = df_fy22[~is_anomaly].copy()

print("-" * 80)
print(f"Valid FY 2022+ Analytical Base:     {len(df_clean):,}\n")

# ===========================================================================
# CHECKPOINT 1: Range & Extremes Verification (On Clean Data)
# ===========================================================================
print("CHECKPOINT 1: Temporal Boundaries")
print("-" * 80)
print(f"DOJ Range:          {df_clean['amr_doj'].min().date()} to {df_clean['amr_doj'].max().date()}")
# Using dropna() just for the min/max calculation to avoid NaT errors
print(f"DOL Range:          {df_clean['amr_actualdol'].dropna().min().date()} to {df_clean['amr_actualdol'].max().date()}")
print(f"Contract End Range: {df_clean['amr_contractend'].min().date()} to {df_clean['amr_contractend'].max().date()}\n")

# ===========================================================================
# CHECKPOINT 2: Completeness Verification
# ===========================================================================
print("CHECKPOINT 2: Completeness (Confirming DOJ/Contract have no NAs)")
print("-" * 80)
print(f"Missing DOJ:           {df_clean['amr_doj'].isna().sum():,}")
print(f"Missing Contract End:  {df_clean['amr_contractend'].isna().sum():,}")
print(f"Missing DOL (Active):  {df_clean['amr_actualdol'].isna().sum():,}\n")

# ===========================================================================
# CHECKPOINT 3: Contract Alignment Matrix 
# ===========================================================================
print("CHECKPOINT 3: Behavioral Matrix (DOL vs. Contract End)")
print("-" * 80)

df_clean['contract_vs_actual'] = 'Active'
has_dol = df_clean['amr_actualdol'].notna()

# Group departed individuals based on contract timeline logic
df_clean.loc[has_dol & (df_clean['amr_actualdol'] < df_clean['amr_contractend']), 'contract_vs_actual'] = 'Left Early'
df_clean.loc[has_dol & (df_clean['amr_actualdol'] == df_clean['amr_contractend']), 'contract_vs_actual'] = 'Left exactly on contract end'
df_clean.loc[has_dol & (df_clean['amr_actualdol'] > df_clean['amr_contractend']), 'contract_vs_actual'] = 'overstayed contract'

# Identify ghost records (active status, but contract end is in the past)
is_ghost = df_clean['amr_actualdol'].isna() & (df_clean['amr_contractend'] < PROXY_PULL_DATE)
df_clean.loc[is_ghost, 'contract_vs_actual'] = 'Active but contract expired'

print(df_clean['contract_vs_actual'].value_counts().to_string())
print("-" * 80)
print(f"Total account check: {len(df_clean):,}\n")

# ===========================================================================
# CHECKPOINT 4: Tenure & Event Classification Audit
# ===========================================================================
print("CHECKPOINT 4: Tenure Distribution (Stayed vs. Left)")
print("-" * 80)

# Calculate exact tenure in days
df_clean['Duration_Days'] = np.nan

# For Leavers: DOL - DOJ
df_clean.loc[has_dol, 'Duration_Days'] = (df_clean.loc[has_dol, 'amr_actualdol'] - df_clean.loc[has_dol, 'amr_doj']).dt.days

# For Actives: Proxy Date - DOJ
df_clean.loc[~has_dol, 'Duration_Days'] = (PROXY_PULL_DATE - df_clean.loc[~has_dol, 'amr_doj']).dt.days

# Create a simple Employment Status column for aggregation
df_clean['Employment_Status'] = np.where(has_dol, 'Left Company', 'Active Worker')

# Display summary statistics of  tenure, split by Employment Status
tenure_summary = df_clean.groupby('Employment_Status')['Duration_Days'].describe()[['count', 'mean', '50%', 'max']]
tenure_summary = tenure_summary.round(1)  # Clean up decimals
print(tenure_summary.to_string())
print("-" * 80)

# Final Event Mapping Check (Integrating Contract Logic)
# If relying on contract end logic: Event = 1 if they left prematurely.
df_clean['Event'] = 0
df_clean.loc[df_clean['contract_vs_actual'] == 'Left Early (Potential Churn)', 'Event'] = 1

print("\nDerived Event Flag Breakdown:")
print(df_clean['Event'].value_counts().rename(index={0: '0 (Success/Censored)', 1: '1 (Premature Churn)'}))

# ===========================================================================
# --- Task 1b: Create event (attrition) mapping and plot the survival curve  ---
# ===========================================================================

# Default everyone to 0 
df_clean['Event'] = 0 

# Only flag people who broke the contract early as True Churns (1)
df_clean.loc[df_clean['contract_vs_actual'] == 'Left Early', 'Event'] = 1

print("Final Event Flag Breakdown:")
print("-" * 40)
print(df_clean['Event'].value_counts().rename(index={0: '0 (Success/Censored)', 1: '1 (Premature Churn)'}))
print("-" * 40)

# Fit the Kaplan-Meier Model
kmf = KaplanMeierFitter()
# We use the Duration_Days we already calculated, and our newly mapped Event flag
kmf.fit(durations=df_clean['Duration_Days'], 
        event_observed=df_clean['Event'])

# Plot the Curve
plt.figure(figsize=(12, 7))
kmf.plot_survival_function(color='#2E86AB', linewidth=2.5, legend = False)
plt.title('Kaplan-Meier Overall Survival Curve', fontsize=14, fontweight='bold')
plt.xlabel('Tenure (Days)', fontsize=12)
plt.ylabel('Proportion Still Active', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0, 1.05])
plt.xlim([0, df_clean['Duration_Days'].max()]) 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# ===========================================================================
# --- Task 1c: Survival curves with alternate definitions of attrition---
# ===========================================================================
# For a quick plot using alternative definition of attrition define following class
def apply_time_bound(df, time_limit_days):
    df_bound = df.copy()
    df_bound['Bounded_Duration'] = np.minimum(df_bound['Duration_Days'], time_limit_days)
    df_bound['Bounded_Event'] = np.where(
        (df_bound['Event'] == 1) & (df_bound['Duration_Days'] <= time_limit_days), 
        1, 0
    )
    
    return df_bound

# Define our specific time thresholds and colors for plotting
thresholds = {
    '7 Days': {'days': 7, 'color': '#E63946'},       # Red
    '30 Days': {'days': 30, 'color': '#F4A261'},     # Orange
    '90 Days': {'days': 90, 'color': '#2A9D8F'},     # Teal
    '1 Year': {'days': 365, 'color': '#264653'}      # Dark Navy
}

kmf = KaplanMeierFitter()

# ===========================================================================
# 2. Generate the 4 Individual Plots
# ===========================================================================
print("Generating individual survival curves...\n")

for label, config in thresholds.items():
    days = config['days']
    color = config['color']
    
    # 1. Apply our reusable function
    df_temp = apply_time_bound(df_clean, days)
    
    # 2. Fit the model
    kmf.fit(durations=df_temp['Bounded_Duration'], 
            event_observed=df_temp['Bounded_Event'], 
            label=f'{label} Retention')
    
    # 3. Plot Individual Curve
    plt.figure(figsize=(10, 5))
    kmf.plot_survival_function(color=color, linewidth=2.5)
    
    plt.title(f'Kaplan-Meier Survival Curve: {label} Attrition Definition', fontsize=14, fontweight='bold')
    plt.xlabel('Tenure (Days)', fontsize=12)
    plt.ylabel('Proportion Still Active', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([0.0, 1.05])
    plt.xlim([0, days]) # Strictly bound the X-axis to the definition
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
# ===========================================================================
# 3. Generate the Combined Comparison Plot
# ===========================================================================
# ===========================================================================
# 3. Generate the Combined 2x2 Matrix Comparison Plot
# ===========================================================================
print("Generating 2x2 matrix comparison curve...\n")

# Create a 2x2 figure layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Survival Curve Comparison by Attrition Definition', fontsize=16, fontweight='bold', y=1.02)

# Flatten the axes array to easily iterate through the 4 subplots
axes = axes.flatten()

# Loop through thresholds and plot on specific axes
for ax, (label, config) in zip(axes, thresholds.items()):
    days = config['days']
    color = config['color']
    
    df_temp = apply_time_bound(df_clean, days)
    
    kmf.fit(durations=df_temp['Bounded_Duration'], 
            event_observed=df_temp['Bounded_Event'], 
            label=f'{label} Window')
    
    # Plotting directly onto the specific axis (ax)
    kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5, ci_show=True)
    
    # Styling each subplot
    ax.set_title(f'{label} View', fontsize=12, fontweight='bold')
    ax.set_xlabel('Tenure (Days)', fontsize=10)
    ax.set_ylabel('Proportion Still Active', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0, days]) 
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_legend().remove() # Remove individual legends for a cleaner look

plt.tight_layout()
plt.show()

# ===========================================================================
# --- Task 1d: Understanding Dates, status and exitreason ---
# ===========================================================================

# Date Conversion & Proxy Date Setup ---
df['amr_doj'] = pd.to_datetime(df['amr_doj'], errors='coerce')
df['amr_actualdol'] = pd.to_datetime(df['amr_actualdol'], errors='coerce')

PROXY_PULL_DATE = df['amr_actualdol'].max()
print(f"Inferred Data Extraction Date: {PROXY_PULL_DATE.date()}")

# Drop impossible records where an employee left before they joined
err_left_before_join = (df['amr_actualdol'].notna()) & (df['amr_actualdol'] < df['amr_doj'])
print(f"Dropped {err_left_before_join.sum()} anomalous rows (Left before Joining).")
valid_df = df[~err_left_before_join].copy()

#Clean Text Variables
df['amr_status'] = df['amr_status'].astype(str).str.lower().str.strip()
df['amr_exitreason'] = df['amr_exitreason'].astype(str).str.lower().str.strip()

# Replace fake nulls with actual
df['amr_exitreason'].value_counts(dropna=False)
placeholders = ['nan', 'none', 'null', '', '-1']
df['amr_status'] = df['amr_status'].replace(placeholders, np.nan)
df['amr_exitreason'] = df['amr_exitreason'].replace(placeholders, np.nan)

# Create helper columns for matrices
df['dol_status'] = df['amr_actualdol'].notna().map({True: 'DOL Present', False: 'DOL Missing'})
df['reason_status'] = df['amr_exitreason'].notna().map({True: 'Reason Present', False: 'Reason Missing'})

#Generate Diagnostic Matrices
print("MATRIX 1: amr_status vs. Date of Leaving (amr_actualdol) Presence")
print("-" * 75)
matrix1 = pd.crosstab(
    index=df['amr_status'].fillna('<MISSING STATUS>'), 
    columns=df['dol_status'],
    margins=True,
    margins_name='Total'
)
print(matrix1.to_string())
print("\n" + "="*75 + "\n")

print("MATRIX 2: Exit Reason Presence vs. Date of Leaving Presence")
print("-" * 65)
binary_matrix = pd.crosstab(
    index=df['reason_status'], 
    columns=df['dol_status'],
    margins=True,
    margins_name='Total'
)
print(binary_matrix.to_string())
print("\n" + "="*75 + "\n")

print("MATRIX 3: The Contradiction Check (amr_status vs. amr_exitreason)")
print("-" * 75)
matrix3 = pd.crosstab(
    index=df['amr_exitreason'].fillna('<MISSING REASON>'),
    columns=df['amr_status'].fillna('<MISSING STATUS>'),
    margins=True,
    margins_name='Total'
)
matrix3 = matrix3.sort_values(by='Total', ascending=False)
print(matrix3.to_string())
print("\n" + "="*75 + "\n")

#Categorize Variables from Exit Reason
# Define the Category Lists
list_churn = [
    'resigned', 'terminated', 'absc', 'disc', 'terminated absconding', 
    'careerprg', 'hghredu', 'btrcpnstn', 'termination_absconding', 
    'familyreason', 'rltn', 'termination legal', 'termination dual employment', 
    'better compensation', 'terminated non performance employees', 
    'family commitment(marriage / family)', 'exit as per client'
]

list_success = [
    'work assignment expiry', 'project closure', 
    'wrkasgnexp', 'contract non-renewal', 'retirement', 'wrkasgnexpy'
]

list_neutral = [
    'others', 'death', 'mdclrn', 'mton', 
    'medical reasons(self/family)', 'medical reasons(self / family)'
]

list_invalid = ['not joined']
# Assign values based on lists
df['exit_category'] = 'Unknown / Missing' 
df.loc[df['amr_exitreason'].isin(list_churn), 'exit_category'] = '1. Definitely Churned'
df.loc[df['amr_exitreason'].isin(list_success), 'exit_category'] = '2. Completed Role'
df.loc[df['amr_exitreason'].isin(list_neutral), 'exit_category'] = '3. Neutral / Ambiguous'
df.loc[df['amr_exitreason'].isin(list_invalid), 'exit_category'] = '4. Not Joined'
# Print Final Categorization Check
print("Exit Reason Categorization Summary:")
print("-" * 50)
summary = df['exit_category'].value_counts()
print(summary.to_string())
print("-" * 50)
print(f"Total rows accounted for: {summary.sum():,}")

# ===========================================================================
# --- Task 1d: Overall Survival Curve ---
# ===========================================================================   

# Establish the Proxy Date for Right-Censoring
PROXY_PULL_DATE = df['amr_actualdol'].max()

# Initialize the Survival Variables
df['Event'] = 0
df['Duration_Days'] = 0

# 3. Calculate Base Durations
# For leavers (DOL Present): Duration = DOL - DOJ
has_dol = df['amr_actualdol'].notna()
df.loc[has_dol, 'Duration_Days'] = (df.loc[has_dol, 'amr_actualdol'] - df.loc[has_dol, 'amr_doj']).dt.days

# For active employees (DOL Missing): Duration = Proxy Date - DOJ
no_dol = df['amr_actualdol'].isna()
df.loc[no_dol, 'Duration_Days'] = (PROXY_PULL_DATE - df.loc[no_dol, 'amr_doj']).dt.days

# Map the Categories to the Event Flag
# Rule 1: Definitely Churned -> Event = 1
df.loc[df['exit_category'] == '1. Definitely Churned', 'Event'] = 1

# Rule 2: Completed Role -> Event = 0 (Already defaulted to 0)
# Rule 3: Neutral / Ambiguous -> Event = 0 (Already defaulted to 0)

# Rule 4: User Override -> "Not Joined" are Day 0 Churns
is_not_joined = df['exit_category'] == '4. Invalid (Not Joined)'
df.loc[is_not_joined, 'Event'] = 1
df.loc[is_not_joined, 'Duration_Days'] = 0  # Hardcode duration to 0

# 5. Safety Net Filter
# Ensure no negative durations slipped through (e.g., severe data entry errors)
valid_df = df[df['Duration_Days'] >= 0].copy()
len(valid_df)

# 6. Print Final Matrix & Stats Before Plotting
print("Final Event Flag Distribution:")
print("-" * 35)
print(valid_df['Event'].value_counts().rename(index={0: '0 (Survived/Censored)', 1: '1 (Churned)'}))
print("-" * 35)
print(f"Total rows passed to fitter: {len(valid_df):,}\n")

# understanding derived tenure
bins = [0, 365, 730, 1095, 1460, 1825, float('inf')]
labels = ['< 1 Year', '1 - 2 Years', '2 - 3 Years', '3 - 4 Years', '4 - 5 Years', '5+ Years']
valid_df['Tenure_Bucket'] = pd.cut(valid_df['Duration_Days'], bins=bins, labels=labels, right=False)

bucket_counts = valid_df['Tenure_Bucket'].value_counts().reindex(labels)

print("Tenure Distribution Summary:")
print("-" * 35)
print(bucket_counts.to_string())
print("-" * 35)

plt.figure(figsize=(10, 6))
bars = plt.bar(bucket_counts.index, bucket_counts.values, color='#2E86AB', edgecolor='black', alpha=0.8)

for bar in bars:
    yval = bar.get_height()
    # Format the number with commas (e.g., 500,000) for readability
    plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), 
             f'{int(yval):,}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Distribution of Contractor Tenure', fontsize=14, fontweight='bold')
plt.xlabel('Tenure Buckets', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)

plt.ylim(0, bucket_counts.max() * 1.15) 

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Fit and Plot the Kaplan-Meier Curve
kmf = KaplanMeierFitter()
kmf.fit(durations=valid_df['Duration_Days'], event_observed=valid_df['Event'], label='Overall Retention ')

plt.figure(figsize=(12, 7))
kmf.plot_survival_function(color='#2E86AB', linewidth=2.5)

plt.title('Kaplan-Meier Survival Curve (Strictly Exit Reason Based)', fontsize=14, fontweight='bold')
plt.xlabel('Tenure (Days)', fontsize=12)
plt.ylabel('Proportion Surviving', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim([0, 1.05])

# Add reference line
plt.axvline(x=365, color='#48CAE4', linestyle=':', alpha=0.8, label='1 Year Mark')
plt.legend()

plt.tight_layout()
plt.show()

#====================================================================#
# --- Task 2a: Understand age distribution for doj after 2021---
#====================================================================#
# Convert dob to datetime and check for NAs
df_clean['amr_dob'] = pd.to_datetime(df_clean['amr_dob'], format='%d-%b-%y', errors='coerce')
missing_dob = df_clean['amr_dob'].isna().sum()
print(f"Records with missing/unparseable DOBs: {missing_dob:,}\n")

# Calculate a temporary age to reveal the negative age problem
df_clean['Temp_Age'] = (df_clean['amr_doj'] - df_clean['amr_dob']).dt.days / 365
df_negative_check = df_clean[df_clean['Temp_Age'] < 0].copy()

print(f"DIAGNOSTIC: Total Negative Age Records detected pre-correction: {len(df_negative_check):,}")

if len(df_negative_check) > 0:
    # Extract just the year from the futuristic dates
    df_negative_check['Futuristic_Year'] = df_negative_check['amr_dob'].dt.year
    
    # Create dynamic 5-year bins for these futuristic years
    min_f_year = int(df_negative_check['Futuristic_Year'].min())
    max_f_year = int(df_negative_check['Futuristic_Year'].max())
    
    start_bin = (min_f_year // 5) * 5 
    bins_f = list(range(start_bin, max_f_year + 10, 5))
    labels_f = [f"{bins_f[i]} to {bins_f[i+1]-1}" for i in range(len(bins_f)-1)]
    
    df_negative_check['Futuristic_Year_Bucket'] = pd.cut(df_negative_check['Futuristic_Year'], bins=bins_f, labels=labels_f, right=False)
    
    print("Pre-Correction Futuristic DOB Year Distribution (5-Year Intervals):")
    print("-" * 65)
    print(df_negative_check['Futuristic_Year_Bucket'].value_counts().sort_index().to_string())
    print("-" * 65)
    print("-> This Pandas pivot behavior justifies the 100-year correction below.\n")

df_clean = df_clean.drop(columns=['Temp_Age'])

# Identify the Futuristic Years
is_futuristic = df_clean['amr_dob'] > df_clean['amr_doj']
records_to_shift = is_futuristic.sum()

print(f"Futuristic date of births found: {records_to_shift:,}")

# 3. Apply the -100 year shift safely using pd.DateOffset 
df_clean.loc[is_futuristic, 'amr_dob'] -= pd.DateOffset(years=100)

# Calculate Final Age
df_clean['Age'] = (df_clean['amr_doj'] - df_clean['amr_dob']).dt.days / 365

# Post-Correction Summary 
print("Age Distribution Summary:")
print("-" * 60)
print(df_clean['Age'].describe().round(1))
print("-" * 60)
print(f"Total rows in df_clean remains: {len(df_clean):,}")
print("-> NO ROWS HAVE BEEN DELETED.\n")

# Plot the Full, Unfiltered Histogram
plt.figure(figsize=(12, 6))

# We are plotting EVERY calculated age, including impossible extremes (like age 0 or 110).
plt.hist(df_clean['Age'], bins=60, color='#3A86FF', edgecolor='black', alpha=0.8)

plt.title('Age Distribution', fontsize=15, fontweight='bold')
plt.xlabel('Age at Joining (Years)', fontsize=12)
plt.ylabel('Number of Contractors', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Force the X-axis to span from the absolute minimum age to the absolute maximum age
plt.xlim(df_clean['Age'].min(), df_clean['Age'].max()) 

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Finally checking for the exact age distribution for clarify
max_age = int(df_clean['Age'].max()) + 5
bins = list(range(15, max_age, 5))
labels = [f"{bins[i]} to {bins[i+1]-1}" for i in range(len(bins)-1)]
df_clean['Age_5yr_Bracket'] = pd.cut(df_clean['Age'], bins=bins, labels=labels, right=False)
age_distribution = df_clean['Age_5yr_Bracket'].value_counts().sort_index()

#Count and sort the results 
print("\nFinal Age Distribution (5-Year Intervals):")
print("-" * 45)
print(age_distribution.to_string())
print("-" * 45)
print(f"Total valid records categorized: {age_distribution.sum():,}")

#====================================================================#
# --- Task 2b: Survival Curve Age---
#====================================================================#
# Define the 4 age boundaries
age_bins = [15, 25, 35, 45, 100]
age_labels = ['Under 25', '25 to 34', '35 to 44', '45 and Above']

# Apply the categories to the clean dataset
df_clean['Age_Category'] = pd.cut(df_clean['Age'], bins=age_bins, labels=age_labels, right=False)

# Check the final grouped distribution before plotting
print("Final Age Cohort Distribution:")
print("-" * 40)
print(df_clean['Age_Category'].value_counts().sort_index().to_string())
print("-" * 40)

# Setup the graph
plt.figure(figsize=(12, 7))
ax = plt.subplot(111)
colors = ['#4CC9F0', '#4361EE', '#7209B7', '#F72585']

# Iterate through the cohorts and fit the Kaplan-Meier model
for (name, grouped_df), color in zip(df_clean.groupby('Age_Category', observed=True), colors):
    
    if len(grouped_df) == 0:
        continue
        
    kmf_age = KaplanMeierFitter()
    kmf_age.fit(
        durations=grouped_df['Duration_Days'], 
        event_observed=grouped_df['Event'],
        label=f'{name}  (n={len(grouped_df):,})' 
    )
    kmf_age.plot_survival_function(ax=ax, linewidth=2.5, color=color, ci_show=False)

# Styling the visual output
plt.title('Kaplan-Meier Survival Curve by Age Cohort', fontsize=14, fontweight='bold')
plt.xlabel('Tenure (Days)', fontsize=12)
plt.ylabel('Proportion Still Active', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Lock the Y-axis and dynamically bound the X-axis
plt.ylim([0.0, 1.05])
plt.xlim([0, df_clean['Duration_Days'].max()]) 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Format the legend to stand out clearly
plt.legend(title='Age Cohort', title_fontproperties={'weight':'bold'}, loc='best')

plt.tight_layout()
plt.show()
#==============================================#
# --- Task 3: Survival Curve - Gender ---
#==============================================#
# Standardize and Clean the Gender Column
df_clean['amr_gender'] = df_clean['amr_gender'].astype(str).str.strip().str.title()
df_clean['amr_gender'].isna().value_counts()
df_clean['amr_gender'].value_counts()

# Handle text variations of missing data that appear after string conversion
missing_flags = ['Nan', 'None', '', '<Na>', 'Na', 'Unknown']
df_clean['amr_gender'] = df_clean['amr_gender'].replace(missing_flags, pd.NA)

print("Gender Distribution (Before filtering):")
print("-" * 40)
print(df_clean['amr_gender'].value_counts(dropna=False))
print("-" * 40)

# Filter out 29 missing gender records
df_gender_clean = df_clean.dropna(subset=['amr_gender']).copy()
dropped_count = len(df_clean) - len(df_gender_clean)
print(f"Dropped {dropped_count:,} records with missing gender assignments.\n")

# Setup the Graph
plt.figure(figsize=(12, 7))
ax = plt.subplot(111)
colors = ['#3A86FF', '#FF006E', '#FFBE0B', '#8338EC']
genders = sorted(df_gender_clean['amr_gender'].unique())

# 4. Iterate through the genders and fit the Kaplan-Meier model
for gender, color in zip(genders, colors):
    grouped_df = df_gender_clean[df_gender_clean['amr_gender'] == gender]
    
    if len(grouped_df) == 0:
        continue
        
    kmf_gender = KaplanMeierFitter()
    kmf_gender.fit(
        durations=grouped_df['Duration_Days'], 
        event_observed=grouped_df['Event'],
        label=f'{gender} (n={len(grouped_df):,})' 
    )
    
    kmf_gender.plot_survival_function(ax=ax, linewidth=2.5, color=color, ci_show=False)

# Styling the visual output
plt.title('Kaplan-Meier Survival Curve by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Tenure (Days)', fontsize=12)
plt.ylabel('Proportion Surviving', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Lock the Y-axis and dynamically bound the X-axis to the actual data limit
plt.ylim([0.0, 1.05])
plt.xlim([0, df_gender_clean['Duration_Days'].max()]) 

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Format the legend to stand out clearly
plt.legend(title='Gender', title_fontproperties={'weight':'bold'}, loc='best')

plt.tight_layout()
plt.show()

#============================================#
# --- Task 4a: Understanding Wage ---
#============================================#
# Standardize and check for missing values
df_clean['amr_netpay'] = pd.to_numeric(df_clean['amr_netpay'], errors='coerce')
missing_wage = df_clean['amr_netpay'].isna().sum()
print(f"Records with missing/invalid Net Pay: {missing_wage:,}")

# Create a wage-specific dataframe to avoid dropping people from df_clean and drop 148 missing values
df_wage_clean = df_clean.dropna(subset=['amr_netpay']).copy()

# --- Check for and remove zero-wage entries ---
zero_wage_mask = df_wage_clean['amr_netpay'] == 0
zero_wage_count = zero_wage_mask.sum()
print(f"Records with exactly 0 Net Pay: {zero_wage_count:,}")

# Keep only those with a wage strictly greater than 0
df_wage_clean = df_wage_clean[~zero_wage_mask].copy()

print("\nComplete Wage Summary Statistics (Excluding NAs and 0s)")
print("-" * 60)
# The lambda function ensures the summary stats print with clean commas
print(df_wage_clean['amr_netpay'].describe().round(2).apply(lambda x: f"{x:,.2f}"))
print("-" * 60 + "\n")

# Calculate a realistic upper limit to zoom in on the core workforce (99th percentile)
upper_limit = df_wage_clean['amr_netpay'].quantile(0.99)

# Plot the Histogram
plt.figure(figsize=(12, 6))

# Add the 'range' parameter so the 50 bins are distributed evenly across the zoomed area, not the outliers
plt.hist(df_wage_clean['amr_netpay'], bins=50, range=(0, upper_limit), color='#2E86AB', edgecolor='white', alpha=0.85)

plt.title('Distribution of Net Pay', fontsize=14, fontweight='bold')
plt.xlabel('Net Pay Amount (INR)', fontsize=12)
plt.ylabel('Number of Contractors', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Strictly bound the X-axis to our calculated limit
plt.xlim(0, upper_limit)

# Clean borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Format X-axis with commas for readability
ax = plt.gca()
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.tight_layout()
plt.show()

# Generate a clean Distribution Table (Handling the long tail)
# Base the bin sizes on the 99th percentile to focus on the core workforce
upper_limit = df_wage_clean['amr_netpay'].quantile(0.99)
min_wage = int(df_wage_clean['amr_netpay'].min())

# Calculate a step size for the core 99% of the data (roughly 10 bins)
step = int((upper_limit - min_wage) // 10)

# Round step to the nearest 5,000 to create highly readable HR salary brackets
step = max(5000, round(step / 5000) * 5000)

# Create bins up to just past the 99th percentile
max_core_bin = int(round(upper_limit / step) * step) + step
bins = list(range(0, max_core_bin, step))

# Add a final catch-all bin (infinity) for the extreme high-earning outliers
bins.append(float('inf'))

# Generate matching labels dynamically
labels = [f"{bins[i]:,} to {bins[i+1]-1:,}" for i in range(len(bins)-2)]
labels.append(f"{bins[-2]:,} and above")

# Categorize and count
df_wage_clean['Wage_Bracket'] = pd.cut(df_wage_clean['amr_netpay'], bins=bins, labels=labels, right=False)
wage_distribution = df_wage_clean['Wage_Bracket'].value_counts().sort_index()

print("Final Wage Distribution:")
print("-" * 45)
print(wage_distribution.to_string())
print("-" * 45)
print(f"Total valid records categorized: {wage_distribution.sum():,}")

#============================================#
# --- Task 4b: Survival Curve - Wage ---
#============================================#

# Categorize Wages into Quartiles
# Divide the data into 4 statistically equal-sized bins based on percentiles
wage_labels = ['Bottom 25% Pay', 'Lower Middle 25% Pay', 'Upper Middle 25% Pay', 'Top 25% Pay']

# duplicates='drop' ensures the function doesn't crash if a massive group shares the exact same rupee value
df_wage_clean['Wage_Quartile'], bins = pd.qcut(
    df_wage_clean['amr_netpay'], 
    q=4, 
    labels=wage_labels, 
    retbins=True, 
    duplicates='drop'
)

# Print the exact wage boundaries for each quartile for stakeholder transparency
print("Calculated Wage Quartile Boundaries (INR):")
print("-" * 60)
for i in range(len(wage_labels)):
    print(f"{wage_labels[i]:<25}: {bins[i]:,.0f} to {bins[i+1]:,.0f}")
print("-" * 60)

# Check the grouped distribution
print("\nFinal Wage Cohort Distribution:")
print("-" * 40)
print(df_wage_clean['Wage_Quartile'].value_counts().sort_index().to_string())
print("-" * 40)

# Setup the graph
plt.figure(figsize=(12, 7))
ax = plt.subplot(111)
colors = ['#EF476F', '#FFD166', '#06D6A0', '#118AB2']
for (name, grouped_df), color in zip(df_wage_clean.groupby('Wage_Quartile', observed=True), colors):
    
    if len(grouped_df) == 0:
        continue
        
    kmf_wage = KaplanMeierFitter()
    kmf_wage.fit(
        durations=grouped_df['Duration_Days'], 
        event_observed=grouped_df['Event'],
        label=f'{name} (n={len(grouped_df):,})' 
    )
    
    kmf_wage.plot_survival_function(ax=ax, linewidth=2.5, color=color, ci_show=False)

plt.title('Kaplan-Meier Survival Curve by Net Pay', fontsize=14, fontweight='bold')
plt.xlabel('Tenure (Days)', fontsize=12)
plt.ylabel('Proportion Still Active', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Lock the Y-axis and dynamically bound the X-axis
plt.ylim([0.0, 1.05])
plt.xlim([0, df_wage_clean['Duration_Days'].max()]) 

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Format the legend
plt.legend(title='Wage Tier', title_fontproperties={'weight':'bold'}, loc='best')

plt.tight_layout()
plt.show()
#============================================#
# --- Task 5a: understanding clientsize and clientecode---
#============================================#
# Standardize text to catch hidden empties (whitespace only)
# Convert to string, strip spaces, and uppercase everything for clean grouping
df_clean['amr_clientname_clean'] = df_clean['amr_clientname'].astype(str).str.strip().str.upper()
df_clean['amr_clientecode_clean'] = df_clean['amr_clientecode'].astype(str).str.strip().str.upper()

# Explicitly define what counts as "missing" after string conversion
missing_flags = ['NAN', 'NONE', '', '<NA>', 'NULL']
df_clean.loc[df_clean['amr_clientname_clean'].isin(missing_flags), 'amr_clientname_clean'] = np.nan
df_clean.loc[df_clean['amr_clientecode_clean'].isin(missing_flags), 'amr_clientecode_clean'] = np.nan

# Missing Value Analysis
missing_names = df_clean['amr_clientname_clean'].isna().sum()
missing_codes = df_clean['amr_clientecode_clean'].isna().sum()
total_rows = len(df_clean)

print("1. MISSING VALUE ANALYSIS:")
print("-" * 50)
print(f"Missing Client Names: {missing_names:,} ({(missing_names/total_rows)*100:.2f}%)")
print(f"Missing Client Codes: {missing_codes:,} ({(missing_codes/total_rows)*100:.2f}%)\n")

# Unique Value Analysis
unique_names = df_clean['amr_clientname_clean'].nunique()
unique_codes = df_clean['amr_clientecode_clean'].nunique()

print("2. UNIQUE VALUE ANALYSIS:")
print("-" * 50)
print(f"Total Unique Client Names: {unique_names:,}")
print(f"Total Unique Client Codes: {unique_codes:,}\n")

# Top 5 Distribution (To check for skew and alignment)
print("3. TOP 5 CLIENT NAMES BY VOLUME:")
print("-" * 50)
print(df_clean['amr_clientname_clean'].value_counts().head(5).to_string())
print("\n")

print("4. TOP 5 CLIENT CODES BY VOLUME:")
print("-" * 50)
print(df_clean['amr_clientecode_clean'].value_counts().head(5).to_string())
print("-" * 50)

# Clean the Client Names
# Convert to string, strip whitespace, and uppercase to fix standard typos (e.g., 'Amazon' vs 'AMAZON')
df_clean['amr_clientname_clean'] = df_clean['amr_clientname'].astype(str).str.strip().str.upper()

# Explicitly handle text variations of missing data
missing_flags = ['NAN', 'NONE', '', '<NA>', 'NULL']
df_clean.loc[df_clean['amr_clientname_clean'].isin(missing_flags), 'amr_clientname_clean'] = pd.NA

# Isolate clean data for client analysis
df_client_clean = df_clean.dropna(subset=['amr_clientname_clean']).copy()

# 2. Get Top 20 Clients
top_20_clients = df_client_clean['amr_clientname_clean'].value_counts().head(20)

print(f"Total Unique Clients Found: {df_client_clean['amr_clientname_clean'].nunique():,}\n")

# 3. Plot Top 20 Horizontal Bar Chart
plt.figure(figsize=(12, 8))

# We reverse the order ([::-1]) so the largest bar appears at the very top of the chart
bars = plt.barh(top_20_clients.index[::-1], top_20_clients.values[::-1], color='#3A86FF', edgecolor='white')

plt.title('Top 20 Clients by Contractor Volume (FY22+)', fontsize=14, fontweight='bold')
plt.xlabel('Number of Contractors Placed', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Add exact numeric labels to the end of each bar for immediate stakeholder readability
for bar in bars:
    plt.text(bar.get_width() + (top_20_clients.max() * 0.01), 
             bar.get_y() + bar.get_height()/2, 
             f"{int(bar.get_width()):,}", 
             va='center', ha='left', fontsize=10)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

#============================================#
# --- Task 5c: Set up the firms categoriztion---
#============================================#
# 1. Get the raw volume per client
client_volumes = df_client_clean['amr_clientname_clean'].value_counts()

print("1. NATURAL DISTRIBUTION OF CLIENT SIZES:")
print("-" * 50)
print("This tells you how big your average and top-tier clients actually are.")
print(client_volumes.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(1))
print("-" * 50 + "\n")

# 2. Test your boundaries (Adjust these numbers based on the table above)
test_bins = [0, 250, 1000, 5000, float('inf')]
test_labels = ['Small (1-250)', 'Medium (251-1000)', 'Large (1001-5000)', 'Enterprise (5000+)']

# 3. Apply the test boundaries to the clients
client_tiers = pd.cut(client_volumes, bins=test_bins, labels=test_labels)

# 4. Map the test tiers back to the main worker dataframe to check statistical power
df_client_clean['Test_Tier'] = df_client_clean['amr_clientname_clean'].map(client_tiers)

print("2. WORKER DISTRIBUTION CHECK (Statistical Power):")
print("-" * 50)
print("This tells you how many individual WORKERS will power each survival curve.")
print("Goal: Ensure no category has fewer than ~5,000 workers.")
print(df_client_clean['Test_Tier'].value_counts().sort_index().to_string())
print("-" * 50)
#============================================#
# --- Task 5b: Plot the survival curve---
#============================================#
# Define the client size obtained from iteration of previous step
bins = [0, 250, 1000, 5000, float('inf')]
labels = ['Small (1-250)', 'Medium (251-1000)', 'Large (1001-5000)', 'Enterprise (5000+)']

# Apply the tiers
df_client_clean['Client_Size_Tier'] = pd.cut(df_client_clean['Client_Total_Headcount'], bins=bins, labels=labels)

# Setup the Graph for Survival Curve
plt.figure(figsize=(12, 7))
ax = plt.subplot(111)
colors = ['#FF9F1C', '#2EC4B6', '#E71D36', '#011627']

# Iterate through the size tiers and fit the Kaplan-Meier model
for (name, grouped_df), color in zip(df_client_clean.groupby('Client_Size_Tier', observed=True), colors):
    if len(grouped_df) == 0:
        continue
        
    kmf_client = KaplanMeierFitter()
    kmf_client.fit(
        durations=grouped_df['Duration_Days'], 
        event_observed=grouped_df['Event'],
        label=f'{name} (n={len(grouped_df):,})' 
    )
    
    kmf_client.plot_survival_function(ax=ax, linewidth=2.5, color=color, ci_show=False)

# Styling the visual output
plt.title('Kaplan-Meier Survival Curve by Client Size Tier', fontsize=14, fontweight='bold')
plt.xlabel('Tenure (Days)', fontsize=12)
plt.ylabel('Proportion Surviving', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.ylim([0.0, 1.05])
plt.xlim([0, df_client_clean['Duration_Days'].max()]) 
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.legend(title='Client Size Tier', title_fontproperties={'weight':'bold'}, loc='best')

plt.tight_layout()
plt.show()
#============================================#
# --- Task 6: Drawing Hist with tenure ---
#============================================#

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

#============================================#
# --- Task 7: Survival Curve - Payroll vs Temping ---
#============================================#

#Clean ecode
df['ecode'] = df['ecode'].astype(str)
df2['ecode'] = df2['ecode'].astype(str)

# remove invisible spaces
df['ecode'] = df['ecode'].str.strip()
df2['ecode'] = df2['ecode'].str.strip()

# force uppercase for match
df['ecode'] = df['ecode'].str.upper()
df2['ecode'] = df2['ecode'].str.upper()

# Stats about payrolling and tempting
set_df = set(df['ecode'].dropna())
set_df2 = set(df2['ecode'].dropna())

matched_ecode = set_df.intersection(set_df2)
only_df = set_df-set_df2
only_df2 = set_df2-set_df

#print the stats obtained
print("--- POST-CLEANING MATCH SUMMARY ---")
print(f"Total Unique in df:  {len(set_df)}")
print(f"Total Unique in df2: {len(set_df2)}")
print("-" * 30)
print(f"MATCHING ECODES:     {len(matched_ecode)}")
print(f"Only in df:          {len(only_df)}")
print(f"Only in df2:         {len(only_df2)}")

# Survival Curve: Bring 'associate_type' into 'df' where the survival curve exists 
# create a dictionary from df2 and map it to df based on the 'ecode'
mapping_dict = df2.set_index('ecode')['associate_type']
df['associate_type'] = df['ecode'].map(mapping_dict)

# drop people who didn't have a match
df_plot_type = df.dropna(subset=['associate_type']).copy()

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Loop through associate_type and draw separate line for each
for name, grouped_df in df_plot_type.groupby('associate_type'):
    kmf_type = KaplanMeierFitter()
    
    kmf_type.fit(
        durations=grouped_df['Tenure_Days'],
        event_observed=grouped_df['Event'],
        label=f'{name}'
    )
    kmf_type.plot_survival_function(ax=ax, linewidth=2.5)

# Format the chart
plt.title('Survival Curve: Payrolling vs. Temping Candidates', fontsize=14, fontweight='bold')
plt.xlabel('Tenure (Days)', fontsize=12)
plt.ylabel('Proportion Still Active', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Lock the axes to standard percentages and 1 year
plt.ylim([0, 1.05])
plt.xlim([0, 365])
plt.tight_layout()
plt.show()
#============================================#
# --- Task 8: Bar Chart Attrition---
#============================================#

total_workers = len(df_clean)

# Define required variables using our clean pipeline metrics
# Renege: Workers who left on Day 0 (or anomalies with negative duration mapped to 0)
renege_count = len(df_clean[df_clean['Duration_Days'] == 0])

# Retention Milestones: Workers who survived up to these specific days
retained_7 = len(df_clean[df_clean['Duration_Days'] >= 7])
retained_14 = len(df_clean[df_clean['Duration_Days'] >= 14])
retained_30 = len(df_clean[df_clean['Duration_Days'] >= 30])
retained_90 = len(df_clean[df_clean['Duration_Days'] >= 90])

# Completion: Anyone mapped as a Success/Censored record (Event == 0)
completion_count = len(df_clean[df_clean['Event'] == 0])

# 2. Calculate percentages
percentages = {
    'Renege (Day 0)': (renege_count / total_workers) * 100,
    '7 Days': (retained_7 / total_workers) * 100,
    '14 Days': (retained_14 / total_workers) * 100,
    '1 Month': (retained_30 / total_workers) * 100,
    '3 Months': (retained_90 / total_workers) * 100,
    'Completed / Active': (completion_count / total_workers) * 100
}

df_milestones = pd.DataFrame(list(percentages.items()), columns=['Milestone', 'Proportion (%)'])

print("Retention Milestone Breakdown:")
print("-" * 40)
print(df_milestones.to_string(index=False))
print("-" * 40 + "\n")

# Draw the bar chart 
plt.figure(figsize=(10, 6))

# Highlight Renege (Red) and Completion (Green), keep intermediate milestones uniform (Blue)
colors = ['#E71D36', '#2EC4B6', '#2EC4B6', '#2EC4B6', '#2EC4B6', '#2CA02C']

bars = plt.bar(
    df_milestones['Milestone'], 
    df_milestones['Proportion (%)'], 
    color=colors, 
    edgecolor='white',
    alpha=0.9
)

# Formatting the chart
plt.title('Workforce Retention at Key Milestones (FY22+)', fontsize=14, fontweight='bold')
plt.ylabel('Proportion of Total Workforce (%)', fontsize=12)
plt.xlabel('Retention Milestone', fontsize=12)

# Lock Y-axis to 100%
plt.ylim(0, 100) 
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add exact percentage text on top of each bar for clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, 
             f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()




# --- 1. Data Cleaning ---
# Force to string, lowercase, and strip accidental whitespaces
df['amr_status'] = df['amr_status'].astype(str).str.lower().str.strip()
df['amr_exitreason'] = df['amr_exitreason'].astype(str).str.lower().str.strip()
df['amr_status'].isna().value_counts() #No na

# Replace the fake nulls and pandas string 'nan' with actual np.nan
placeholders = ['nan', 'none', 'null', '', '-1']
df['amr_status'] = df['amr_status'].replace(placeholders, np.nan)
df['amr_exitreason'] = df['amr_exitreason'].replace(placeholders, np.nan)

# Create a clear helper column for DOL presence
df['dol_status'] = df['amr_actualdol'].notna().map({True: 'DOL Present', False: 'DOL Missing'})

# --- 2. Generate the Matrices ---

print("MATRIX 1: amr_status vs. Date of Leaving (amr_actualdol) Presence")
print("-" * 75)
matrix1 = pd.crosstab(
    index=df['amr_status'].fillna('<MISSING STATUS>'), 
    columns=df['dol_status'],
    margins=True,
    margins_name='Total'
)
print(matrix1.to_string())
print("\n" + "="*75 + "\n")

df['dol_status'] = df['amr_actualdol'].notna().map({True: 'DOL Present', False: 'DOL Missing'})
df['reason_status'] = df['amr_exitreason'].notna().map({True: 'Reason Present', False: 'Reason Missing'})

# 3. Generate the Clean 2x2 Matrix
print("MATRIX: Exit Reason Presence vs. Date of Leaving Presence")
print("-" * 65)
binary_matrix = pd.crosstab(
    index=df['reason_status'], 
    columns=df['dol_status'],
    margins=True,
    margins_name='Total'
)

print(binary_matrix.to_string())


print("MATRIX 3: The Contradiction Check (amr_status vs. amr_exitreason)")
print("-" * 75)
matrix3 = pd.crosstab(
    index=df['amr_exitreason'].fillna('<MISSING REASON>'),
    columns=df['amr_status'].fillna('<MISSING STATUS>'),
    margins=True,
    margins_name='Total'
)
# Sort by Total for readability
matrix3 = matrix3.sort_values(by='Total', ascending=False)
print(matrix3.to_string())


df['amr_exitreason'].unique().tolist()



after_2024= df[df['amr_doj'].dt.year>2021].copy()
after_2024['amr_doj'].dt.year.value_counts(dropna=False).sort_index()
after_2024['amr_actualdol'].dt.year.value_counts(dropna=False).sort_index()




list_success = [
    'work assignment expiry', 'project closure', 
    'wrkasgnexp', 'contract non-renewal', 'retirement', 'wrkasgnexpy'
]

df_filter = df[df['amr_exitreason'].isin(list_success)].copy()
len(df_filter)


# Establish the Proxy Date for Right-Censoring
PROXY_PULL_DATE = df_filter['amr_actualdol'].max()

# Initialize the Survival Variables
df_filter['Event'] = 0
df_filter['Duration_Days'] = 0

# 3. Calculate Base Durations
# For leavers (DOL Present): Duration = DOL - DOJ
has_dol = df_filter['amr_actualdol'].notna()
df_filter.loc[has_dol, 'Duration_Days'] = (df_filter.loc[has_dol, 'amr_actualdol'] - df_filter.loc[has_dol, 'amr_doj']).dt.days

# For active employees (DOL Missing): Duration = Proxy Date - DOJ
no_dol = df_filter['amr_actualdol'].isna()
df_filter.loc[no_dol, 'Duration_Days'] = (PROXY_PULL_DATE - df_filter.loc[no_dol, 'amr_doj']).dt.days

# 5. Safety Net Filter
# Ensure no negative durations slipped through (e.g., severe data entry errors)
valid_df = df_filter[df_filter['Duration_Days'] >= 0].copy()
len(valid_df)

# 6. Print Final Matrix & Stats Before Plotting
print("Final Event Flag Distribution:")
print("-" * 35)
print(valid_df['Event'].value_counts().rename(index={0: '0 (Survived/Censored)', 1: '1 (Churned)'}))
print("-" * 35)
print(f"Total rows passed to fitter: {len(valid_df):,}\n")

# understanding derived tenure
bins = [0, 365, 730, 1095, 1460, 1825, float('inf')]
labels = ['< 1 Year', '1 - 2 Years', '2 - 3 Years', '3 - 4 Years', '4 - 5 Years', '5+ Years']
valid_df['Tenure_Bucket'] = pd.cut(valid_df['Duration_Days'], bins=bins, labels=labels, right=False)

bucket_counts = valid_df['Tenure_Bucket'].value_counts().reindex(labels)

print("Tenure Distribution Summary:")
print("-" * 35)
print(bucket_counts.to_string())
print("-" * 35)

plt.figure(figsize=(10, 6))
bars = plt.bar(bucket_counts.index, bucket_counts.values, color='#2E86AB', edgecolor='black', alpha=0.8)

for bar in bars:
    yval = bar.get_height()
    # Format the number with commas (e.g., 500,000) for readability
    plt.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.02), 
             f'{int(yval):,}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Distribution of Contractor Tenure', fontsize=14, fontweight='bold')
plt.xlabel('Tenure Buckets', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)

plt.ylim(0, bucket_counts.max() * 1.15) 

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

df['amr_actualdol'].isna().value_counts()

# 1. Force the raw column to string so we can inspect the text
raw_dob_strings = df_clean['amr_dob'].astype(str).str.strip()

# 2. Use a regular expression to see if a 4-digit sequence exists in the string
# If it has 4 digits (e.g., "1995-12-01"), it's YYYY. If not (e.g., "95-12-01"), it's YY.
has_4_digit_year = raw_dob_strings.str.contains(r'\d{4}', na=False)

# 3. Filter out the fake nulls so they don't get counted as 2-digit years
valid_strings = ~raw_dob_strings.isin(['nan', 'none', 'null', '', 'NaT'])

# Calculate the populations
count_4_digit = (has_4_digit_year & valid_strings).sum()
count_2_digit = (~has_4_digit_year & valid_strings).sum()

print("RAW DATA YEAR FORMAT CHECK:")
print("-" * 45)
print(f"Total dates with 4-digit years (YYYY): {count_4_digit:,}")
print(f"Total dates with 2-digit years (YY):   {count_2_digit:,}")
print("-" * 45)

# 4. View the actual text of the culprits
if count_2_digit > 0:
    print("\nHere is exactly what the raw 2-digit entries look like:")
    # We use ~has_4_digit_year to pull the ones missing the 4-digit year
    print(df_raw[~has_4_digit_year & valid_strings]['amr_dob'].head(15))



