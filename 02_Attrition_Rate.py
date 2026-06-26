# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:07:17 2026

@author: Dev
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.express as px
# Load datasets
# only importing a few columns for better speed
cols = ['amr_emp_id', 'amr_gender', 'amr_dob', 'ecode', 'amr_contractend', 'amr_marital_status', 'amr_qualification',
    'amr_doj', 'amr_actualdol', 'amr_exitreason', 'amr_status', 'amr_designation', 'amr_department', 
    'amr_jobcategory', 'amr_grade','amr_clientname', 'amr_joblocationpincode', 'amr_joblocationdistrict', 
    'amr_ptstate','amr_permanentaddress', 'amr_presentaddress', 'amr_mailingaddress',
    'amr_ctc', 'amr_gross', 'amr_netpay', 'amr_basic']
df = pd.read_parquet(r"C:\Users\New\Desktop\Staffing_project\amr_GS_appended.dta", columns=cols)
df2 = pd.read_parquet(r"C:\Users\New\Desktop\Staffing_project\amr_GS_appended_temp_payroll1.dta")

#=====================================================#
# --- Task 0: Overall Attrition rate variables---
#=====================================================#

# Convert required columns to datetime
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

#=====================================================#
# --- Task 8a: Headcount Map of India full data---
#=====================================================#

#clean states
df['amr_ptstate'].isna().value_counts()
df['amr_ptstate'] = df['amr_ptstate'].astype(str).str.strip().str.title()
state_counts = df['amr_ptstate'].value_counts(dropna=False)

print(f"Total Unique States/UTs found: {len(state_counts)}")
print("\nFull List of States and Headcounts:")
print(state_counts.to_string()) #forces to show all 35 states; Lakshwadeep is missing

# 8a) PLOT FOR NUMBER OF WORKERS IN EACH STATE
import plotly.express as px

# Clean the text
df['amr_ptstate'] = df['amr_ptstate'].astype(str).str.strip().str.title()

# Apply the mapping dictionary from the online Github file GeoJSON used to map states of India
geojson_mapping = {
    'Nct Of Delhi': 'Delhi',
    'Andaman And Nicobar': 'Andaman & Nicobar',
    'Andaman & Nicobar Island': 'Andaman & Nicobar',
    
    # 2020 merger requires these UT to be named as follows
    'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadara & Nagar Havelli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',

    # normal typo
    'Orrisa': 'Odisha',
    'Chhattishgarh': 'Chhattisgarh',
    'Tamilnadu': 'Tamil Nadu',
    'Jammu And Kashmir':'Jammu & Kashmir'
}
df['amr_ptstate'] = df['amr_ptstate'].replace(geojson_mapping)

# Calculate the Aggregate Headcount
state_headcounts = df.groupby('amr_ptstate').agg(Total_Headcount=('Event', 'size')).reset_index()

# Standard GeoJSON URL obtained from Github
india_geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"

# construce the Headcount Map
fig_headcount = px.choropleth(
    state_headcounts,
    geojson=india_geojson_url,
    locations='amr_ptstate',         
    featureidkey='properties.ST_NM', 
    color='Total_Headcount',          
    color_continuous_scale='Blues',  # Blue scale for Volume/Headcount 
    title='Total Workforce Headcount by State',
    labels={'Total_Headcount': 'Number of Workers', 'amr_ptstate': 'State'}
)

# Format the map viewport
fig_headcount.update_geos(
    fitbounds="locations", 
    visible=False
)

fig_headcount.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0}, 
    title_font_size=16, 
    title_x=0.5
)

# Save the output in HTML format
save_path_1 = r"C:\Users\New\Desktop\Staffing_project\Headcount_Map.html"
fig_headcount.write_html(save_path_1)
print(f"Headcount Map saved successfully to: {save_path_1}")

# Confirm to see nothing is missed to be accounted for
total_map_workers = state_headcounts['Total_Headcount'].sum()
original_workers = len(df)
print(f"Total workers in map:      {total_map_workers}")
print(f"Total workers in database: {original_workers}") #Note sum is the same: 1016933
#=====================================================#
# --- Task 8b: Headcount Map of India cleaned data for which attrition is available---
#=====================================================#
# Clean the State text in df_clean
# Convert to string, strip whitespace, and title case
df_clean['amr_ptstate'] = df_clean['amr_ptstate'].astype(str).str.strip().str.title()

# Apply the Exhaustive GeoJSON mapping dictionary
geojson_mapping = {
    'Nct Of Delhi': 'Delhi',
    'New Delhi': 'Delhi',
    'Andaman And Nicobar': 'Andaman & Nicobar',
    'Andaman And Nicobar Islands': 'Andaman & Nicobar',
    'Andaman & Nicobar Island': 'Andaman & Nicobar',
    'Andaman & Nicobar Islands': 'Andaman & Nicobar',
    
    # EXHAUSTIVE DADRA & DAMAN MAPPING
    # Catching every variation of capitalization and ampersands produced by .title()
    'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadara & Nagar Havelli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra & Nagar Haveli And Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',

    # Standardizing standard typos to match the map file exactly
    'Orrisa': 'Odisha',
    'Orissa': 'Odisha',
    'Chhattishgarh': 'Chhattisgarh',
    'Tamilnadu': 'Tamil Nadu',
    'Jammu And Kashmir': 'Jammu & Kashmir',
    
    # Catching Pandas string conversion artifacts
    'Nan': 'Unknown',
    'None': 'Unknown'
}

df_clean['amr_ptstate'] = df_clean['amr_ptstate'].replace(geojson_mapping)

# Calculate the Aggregate Headcount
state_headcounts = df_clean.groupby('amr_ptstate').agg(Total_Headcount=('Event', 'size')).reset_index()

# Sort descending so the highest volume states appear at the top of the printout
state_headcounts = state_headcounts.sort_values(by='Total_Headcount', ascending=False).reset_index(drop=True)

# Pre-Plot Verification: Print the complete mapped list
print("Final Mapped States & Headcounts (Pre-Plotting):")
print("-" * 50)
print(state_headcounts.to_string())
print("-" * 50 + "\n")

# 5. Construct the Plotly Map
india_geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"

fig_headcount = px.choropleth(
    state_headcounts,
    geojson=india_geojson_url,
    locations='amr_ptstate',         
    featureidkey='properties.ST_NM', 
    color='Total_Headcount',          
    color_continuous_scale='Blues',  
    title='Total Workforce Headcount by State (FY22+)',
    labels={'Total_Headcount': 'Number of Workers', 'amr_ptstate': 'State'}
)

# Format the map viewport to zoom directly onto India
fig_headcount.update_geos(
    fitbounds="locations", 
    visible=False
)

fig_headcount.update_layout(
    margin={"r":0,"t":50,"l":0,"b":0}, 
    title_font_size=16, 
    title_x=0.5
)

# Save the output as an interactive HTML file exactly where you requested
save_path = r"C:/Users/New/Desktop/Staffing_project/Output/Headcount_Map.html"
fig_headcount.write_html(save_path)
print(f"-> Interactive map successfully saved to: {save_path}\n")

# Confirm Final integrity check whether everything was plotted or not
total_map_workers = state_headcounts['Total_Headcount'].sum()
original_workers = len(df_clean)

print("DATA INTEGRITY CHECK:")
print("-" * 50)
print(f"Total workers accounted for in map mapping: {total_map_workers:,}")
print(f"Total workers in df_clean dataset:          {original_workers:,}")

if total_map_workers == original_workers:
    print("Status: SUCCESS (No records lost during spatial grouping)")
else:
    print("Status: WARNING (Mismatch detected. Check for dropped rows.)")
print("-" * 50)
#=====================================================#
# --- Task 8c: Geographic Attrition Map of India using different definition of attrition---
#=====================================================## Clean the text again

# Standardize State Names (Ensuring mapping from previous step is intact)
df_clean['amr_ptstate'] = df_clean['amr_ptstate'].astype(str).str.strip().str.title()

geojson_mapping = {
    'Nct Of Delhi': 'Delhi', 'New Delhi': 'Delhi',
    'Andaman And Nicobar': 'Andaman & Nicobar', 'Andaman And Nicobar Islands': 'Andaman & Nicobar',
    'Andaman & Nicobar Island': 'Andaman & Nicobar', 'Andaman & Nicobar Islands': 'Andaman & Nicobar',
    'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadara & Nagar Havelli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra & Nagar Haveli And Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Orrisa': 'Odisha', 'Orissa': 'Odisha', 'Chhattishgarh': 'Chhattisgarh',
    'Tamilnadu': 'Tamil Nadu', 'Jammu And Kashmir': 'Jammu & Kashmir',
    'Nan': 'Unknown', 'None': 'Unknown'
}
df_clean['amr_ptstate'] = df_clean['amr_ptstate'].replace(geojson_mapping)

# 2. Define the Four Attrition Event Flags
# Overall Attrition is simply the base Event flag. 
# The others require the person to have churned (Event==1) AND within the specific day limit.
df_clean['Event_Overall'] = df_clean['Event']
df_clean['Event_30'] = ((df_clean['Event'] == 1) & (df_clean['Duration_Days'] <= 30)).astype(int)
df_clean['Event_90'] = ((df_clean['Event'] == 1) & (df_clean['Duration_Days'] <= 90)).astype(int)
df_clean['Event_365'] = ((df_clean['Event'] == 1) & (df_clean['Duration_Days'] <= 365)).astype(int)

# 3. Aggregate Data by State
# We calculate the mean of the binary flags (0 or 1) and multiply by 100 to get exact Attrition Percentages
state_attrition = df_clean.groupby('amr_ptstate').agg(
    Total_Headcount=('Event', 'size'),
    Attr_Overall_Pct=('Event_Overall', lambda x: (x.mean() * 100)),
    Attr_30Day_Pct=('Event_30', lambda x: (x.mean() * 100)),
    Attr_90Day_Pct=('Event_90', lambda x: (x.mean() * 100)),
    Attr_365Day_Pct=('Event_365', lambda x: (x.mean() * 100))
).reset_index()

# Round percentages for clean hover data on the maps
for col in ['Attr_Overall_Pct', 'Attr_30Day_Pct', 'Attr_90Day_Pct', 'Attr_365Day_Pct']:
    state_attrition[col] = state_attrition[col].round(1)

print("Pre-Plotting Verification: Attrition Rates per State (%)")
print("-" * 85)
print(state_attrition.sort_values(by='Total_Headcount', ascending=False).head(10).to_string())
print("-" * 85 + "\n")

# 4. Map Configuration Loop
india_geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"

# Dictionary to manage the 4 different plots: {Column_Name: (Map_Title, File_Name)}
map_configs = {
    'Attr_Overall_Pct': ('Overall Attrition Rate by State (%)', 'Attrition_Map_Overall.html'),
    'Attr_30Day_Pct':   ('30-Day Attrition Rate by State (%)', 'Attrition_Map_30Days.html'),
    'Attr_90Day_Pct':   ('90-Day Attrition by State (%)', 'Attrition_Map_90Days.html'),
    'Attr_365Day_Pct':  ('1-Year Attrition Rate by State (%)', 'Attrition_Map_365Days.html')
}

base_path = "C:/Users/New/Desktop/Staffing_project/Output/"

# Generate and save all 4 maps
for target_column, (map_title, file_name) in map_configs.items():
    
    fig = px.choropleth(
        state_attrition,
        geojson=india_geojson_url,
        locations='amr_ptstate',         
        featureidkey='properties.ST_NM', 
        color=target_column,          
        color_continuous_scale='YlOrRd',  # Yellow-Orange-Red scale highlights high attrition
        title=map_title,
        labels={target_column: 'Attrition Rate (%)', 'amr_ptstate': 'State', 'Total_Headcount': 'Base Volume'},
        hover_data=['Total_Headcount'] # Shows total volume on hover for context
    )

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, title_font_size=16, title_x=0.5)

    # Save output
    save_path = base_path + file_name
    fig.write_html(save_path)
    print(f"-> Successfully saved: {file_name}")

print("\nAll 4 attrition maps generated and exported successfully.")

#-----------------------------------------------------------------#
# --- Task 9: Mirgrant and non migrant workers attrition using pincode---
#-----------------------------------------------------------------#
import matplotlib.patches as mpatches
# Clean Job Pincode 
df_clean['job_pincode'] = df_clean['amr_joblocationpincode'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
job_valid_check = (df_clean['job_pincode'].str.len() == 6) & df_clean['job_pincode'].str.isnumeric()

# Extract Home Pincode from the text address
df_clean['home_pincode'] = df_clean['amr_permanentaddress'].astype(str).str.extract(r'(\b\d{6}\b)', expand=False)
home_valid_check = (df_clean['home_pincode'].str.len() == 6) & df_clean['home_pincode'].str.isnumeric()

print("1. PINCODE EXTRACTION DIAGNOSTICS:")
print("-" * 50)
print(f"Valid 6-Digit Job Pincodes Extracted:  {job_valid_check.sum():,}")
print(f"Valid 6-Digit Home Pincodes Extracted: {home_valid_check.sum():,}\n")

# Slice only the first 3 digits to determine broad geographic region
df_clean['job_pin_3'] = df_clean['job_pincode'].str[:3]
df_clean['home_pin_3'] = df_clean['home_pincode'].str[:3]

# Classify the Workers
conditions = [
    df_clean['home_pin_3'].isna() | df_clean['job_pin_3'].isna(),  # Either pin is missing
    df_clean['job_pin_3'] == df_clean['home_pin_3'],               # First 3 digits match perfectly
    df_clean['job_pin_3'] != df_clean['home_pin_3']                # First 3 digits do NOT match
]
choices = ['Unknown', 'Non-Migrant', 'Migrant']
df_clean['Migrant_Status'] = np.select(conditions, choices, default='Unknown')

print("2. MIGRANT CLASSIFICATION COUNTS:")
print("-" * 50)
print(df_clean['Migrant_Status'].value_counts().to_string())
print("-" * 50 + "\n")

# Define the Four Attrition Events (In case they weren't carried over)
df_clean['Event_Overall'] = df_clean['Event']
df_clean['Event_30'] = ((df_clean['Event'] == 1) & (df_clean['Duration_Days'] <= 30)).astype(int)
df_clean['Event_90'] = ((df_clean['Event'] == 1) & (df_clean['Duration_Days'] <= 90)).astype(int)
df_clean['Event_365'] = ((df_clean['Event'] == 1) & (df_clean['Duration_Days'] <= 365)).astype(int)

# Filter out the 'Unknown' categories for a clean plot base
plot_df = df_clean[df_clean['Migrant_Status'].isin(['Non-Migrant', 'Migrant'])].copy()

# Calculate the Absolute Volumes for all four definitions simultaneously
summary = plot_df.groupby('Migrant_Status').agg(
    Total_Workers=('Event', 'size'),          
    Attrited_Overall=('Event_Overall', 'sum'),
    Attrited_30=('Event_30', 'sum'),
    Attrited_90=('Event_90', 'sum'),
    Attrited_365=('Event_365', 'sum')
).reset_index()

# Plotting Loop Configuration
plot_configs = [
    {'column': 'Attrited_Overall', 'title': 'Overall Attrition: Migrants vs. Non-Migrants', 'color': '#8AB4F8'},
    {'column': 'Attrited_30',      'title': '30-Day Attrition: Migrants vs. Non-Migrants', 'color': '#8AB4F8'},
    {'column': 'Attrited_90',      'title': '90-Day (Q1) Churn: Migrants vs. Non-Migrants', 'color': '#8AB4F8'},
    {'column': 'Attrited_365',     'title': '1-Year (Annual) Churn: Migrants vs. Non-Migrants', 'color': '#8AB4F8'}
]

# X-axis positions
x_positions = np.arange(len(summary['Migrant_Status']))
color_base = '#F8F9FA'       
color_border = '#DEE2E6'

print("3. GENERATING 4 PLOTS...")

# Generate 4 distinct plots
for config in plot_configs:
    
    # Calculate the specific rate for this iteration
    summary['Attrition_Rate'] = (summary[config['column']] / summary['Total_Workers'] * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Draw the Background Bar (Total Base)
    ax.bar(
        x_positions, summary['Total_Workers'], 
        width=0.35, color=color_base, edgecolor=color_border, linewidth=1.2
    )

    # Draw the Foreground Bar (Specific Attrition Metric)
    ax.bar(
        x_positions, summary[config['column']], 
        width=0.15, color=config['color'], edgecolor='white', linewidth=1
    )

    # Add the Data Labels
    for i in range(len(x_positions)):
        total = summary['Total_Workers'].iloc[i]
        attrited = summary[config['column']].iloc[i]
        rate = summary['Attrition_Rate'].iloc[i]
        
        # Label for Total Headcount
        ax.annotate(
            f'Total Base\n{total:,.0f}',
            xy=(x_positions[i], total),
            xytext=(0, 15), textcoords="offset points",
            ha='center', va='bottom', fontsize=10.5, color='#6C757D'
        )
                    
        # Label for Attrition
        # If attrition is extremely low (e.g. 30 days might be small), offset label slightly so it doesn't get squished
        label_y_pos = attrited / 2 if attrited > (total * 0.1) else attrited + (total * 0.05)
        text_color = 'white' if attrited > (total * 0.1) else config['color']
        
        ax.annotate(
            f'{rate}%\n({attrited:,.0f})',
            xy=(x_positions[i], label_y_pos), 
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=text_color
        )

    # Format the Axes & Headroom
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary['Migrant_Status'], fontsize=11, fontweight='bold', color='#495057')
    ax.set_ylim(0, summary['Total_Workers'].max() * 1.35)

    # Format Y-axis with commas
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.tick_params(axis='y', colors='#ADB5BD', length=0) 

    # Custom Legends
    retained_patch = mpatches.Patch(facecolor=color_base, edgecolor=color_border, label='Total Base Headcount')
    attrited_patch = mpatches.Patch(facecolor=config['color'], edgecolor='white', label=f'Attrited ({config["title"].split(":")[0]})')

    ax.legend(
        handles=[retained_patch, attrited_patch], 
        loc='upper center', bbox_to_anchor=(0.5, 1.10), 
        ncol=2, fontsize=10.5, frameon=False, labelcolor='#495057'
    )

    # Styling
    ax.set_title(config['title'], fontsize=14, pad=40, color='#343A40')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#E9ECEF')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.yaxis.grid(True, linestyle='-', alpha=0.5, color='#E9ECEF')
    ax.set_axisbelow(True) 

    plt.tight_layout()
    plt.show()

print("-> Complete.")

#-----------------------------------------------------------------#
# --- Task 10: Getting commuting and migration distance---
#-----------------------------------------------------------------#

import pgeocode

# Extract 6 digits
df['present_pincode'] = df['amr_mailingaddress'].astype(str).str.extract(r'(\b\d{6}\b)', expand=False)
print(df['present_pincode'].isna().value_counts())

# To process this fast, we extract all unique pincodes across the whole dataset into a list
# This prevents the library from looking up the same city multiple times
all_unique_pins = pd.concat([
    df['job_pincode'], 
    df['home_pincode'], 
    df['present_pincode']
]).dropna().unique()

len(all_unique_pins)

# Initialize the offline spatial library
nomi = pgeocode.Nominatim('in')
geo_data = nomi.query_postal_code(all_unique_pins)

# Build a clean master lookup dictionary
geo_lookup = geo_data[['postal_code', 'latitude', 'longitude']].copy()
geo_lookup.columns = ['pincode', 'lat', 'lon']

# Merge coordinates back onto the dataset for JOB
df = pd.merge(df, geo_lookup.rename(columns={'pincode': 'job_pincode', 'lat': 'job_lat', 'lon': 'job_lon'}), on='job_pincode', how='left')

# Merge coordinates back onto the dataset for HOME
df = pd.merge(df, geo_lookup.rename(columns={'pincode': 'home_pincode', 'lat': 'home_lat', 'lon': 'home_lon'}), on='home_pincode', how='left')

# Merge coordinates back onto the dataset for PRESENT
df = pd.merge(df, geo_lookup.rename(columns={'pincode': 'present_pincode', 'lat': 'present_lat', 'lon': 'present_lon'}), on='present_pincode', how='left')

# The High-Speed Vectorized Distance Tool
def calculate_distance_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c # Radius of the Earth in km
    return km

# 1. Calculate Migration Distance (Home to Job)
df['migration_distance_km'] = calculate_distance_km(
    df['home_lat'], df['home_lon'], 
    df['job_lat'], df['job_lon']
).round(1)

print("\n--- 4. FINALIZING MISSING DATA LOGIC ---")

df['migration_distance_km'].isna().value_counts() # output: successfully plotted 926892 entries and 90041 are missing
df['migration_distance_km'].sample(10)

# Analysing Migration Distance
dist_data = df['migration_distance_km'].dropna()

# Print the exact percentiles
percentiles = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
stats_summary = dist_data.describe(percentiles=percentiles)

print(f"Total Valid Entries: {stats_summary['count']:,.0f}")
print(f"Minimum Distance:    {stats_summary['min']:.1f} km")
print(f"25th Percentile:     {stats_summary['25%']:.1f} km")
print(f"Median (50%):        {stats_summary['50%']:.1f} km")
print(f"75th Percentile:     {stats_summary['75%']:.1f} km")
print(f"90th Percentile:     {stats_summary['90%']:.1f} km")
print(f"95th Percentile:     {stats_summary['95%']:.1f} km")
print(f"99th Percentile:     {stats_summary['99%']:.1f} km")
print(f"Maximum Distance:    {stats_summary['max']:.1f} km")

print("\n--- 2. PLOTTING THE DISTRIBUTION (HISTOGRAM) ---")

#Plotting a Histogram with log of no of workers on Y axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the Histogram with 100% of the data
n, bins, patches = ax.hist(
    dist_data, 
    bins=75, 
    color='#8AB4F8',       
    edgecolor='white',     
    linewidth=1.2
)

ax.set_yscale('log')

# Format Axes & Labels
ax.set_title('Workforce Distribution by Migration Distance\n(100% of Data - Logarithmic Scale)', 
             fontsize=14, pad=20, color='#343A40', fontweight='bold')
ax.set_xlabel('Migration Distance from Hometown (Kilometers)', fontsize=12, color='#495057')
ax.set_ylabel('Number of Employees (Log Scale)', fontsize=12, color='#495057')

# Make the Log Scale labels highly readable (e.g., "100k" instead of "10^5")
def log_format(y, pos):
    if y >= 1000000:
        return f'{int(y/1000000)}M'
    elif y >= 1000:
        return f'{int(y/1000)}k'
    elif y >= 1:
        return f'{int(y)}'
    else:
        return ''

# Using the safely imported ticker module
ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_format))
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))

# Clean Borders & Gridlines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#E9ECEF')
ax.spines['bottom'].set_linewidth(1.5)

# Use a slightly more visible gridline so the log scale levels are easy to track
ax.yaxis.grid(True, linestyle='-', alpha=0.6, color='#DEE2E6') 
ax.set_axisbelow(True) 

# Render
plt.tight_layout()
plt.show()

#Plotting correlation of attrition and migration distance
# Filter out rows where distance or event is missing to keep the math clean
from scipy import stats
plot_df = df.dropna(subset=['migration_distance_km', 'Event']).copy()

# Define Geographic Buckets strictly based on the structural breaks in your Histogram
bins = [-1, 50, 250, 750, 1500, plot_df['migration_distance_km'].max() + 1]
labels = [
    '0-50 km', 
    '51-250 km', 
    '251-750 km', 
    '751-1500 km', 
    '1500+ km'
]

# Apply the bins
plot_df['Distance_Bucket'] = pd.cut(plot_df['migration_distance_km'], bins=bins, labels=labels)

print("--- 2. STATISTICAL CORRELATION ---")

# Point-Biserial Correlation (The correct statistical test for Continuous vs Binary data)
correlation, p_value = stats.pointbiserialr(plot_df['migration_distance_km'], plot_df['Event'])
print(f"Point-Biserial Correlation (r): {correlation:.4f}")
print(f"P-Value: {p_value:.4e}")
if p_value < 0.05:
    print("Result: There is a statistically SIGNIFICANT relationship between distance and attrition.")
else:
    print("Result: The relationship is NOT statistically significant.")


print("\n--- 3. CALCULATING AGGREGATES FOR CHART ---")

# Calculate the attrition rate and volume for each distance bucket
summary = plot_df.groupby('Distance_Bucket', observed=False).agg(
    Total_Workers=('Event', 'size'),
    Attrition_Rate=('Event', lambda x: x.mean() * 100)
).reset_index()

summary['Attrition_Rate'] = summary['Attrition_Rate'].round(1)
print(summary.to_string(index=False))

# THE SLEEK BINNED ATTRITION CHART
fig, ax = plt.subplots(figsize=(10, 6.5))
x_positions = np.arange(len(summary['Distance_Bucket']))

# Palette based on the sleek theme
color_bar = '#8AB4F8'     
color_text = '#495057'
color_muted = '#6C757D'

# Draw the Bars
bars = ax.bar(
    x_positions, 
    summary['Attrition_Rate'], 
    width=0.45,             # Slimmer bars for elegance
    color=color_bar,        
    edgecolor='white', 
    linewidth=1.5
)

# Add Data Labels
for i, bar in enumerate(bars):
    rate = summary['Attrition_Rate'].iloc[i]
    volume = summary['Total_Workers'].iloc[i]
    height = bar.get_height()
    
    # 1. Label inside the bar: Attrition Percentage
    ax.annotate(
        f'{rate}%',
        xy=(bar.get_x() + bar.get_width() / 2, height / 2), 
        ha='center', va='center',
        fontsize=12, fontweight='bold', color='white'
    )
    
    # 2. Label hovering above the bar: Headcount Volume
    ax.annotate(
        f"Base: {volume:,.0f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 10), 
        textcoords="offset points",
        ha='center', va='bottom',
        fontsize=10.5, color=color_muted
    )

# Format Axes
ax.set_xticks(x_positions)
ax.set_xticklabels(summary['Distance_Bucket'], fontsize=11, fontweight='bold', color=color_text)
ax.set_ylabel('Attrition Rate (%)', fontsize=12, color=color_text)

# Add 20% headroom so the text labels don't get crushed
ax.set_ylim(0, summary['Attrition_Rate'].max() * 1.25)

# Clean Borders & Titles
ax.set_title('Attrition by Migration Distance', 
             fontsize=15, pad=35, color='#343A40', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#E9ECEF')
ax.spines['bottom'].set_linewidth(1.5)

# Clean minimalist gridlines
ax.yaxis.grid(True, linestyle='-', alpha=0.5, color='#F8F9FA')
ax.tick_params(axis='y', colors='#ADB5BD', length=0)
ax.set_axisbelow(True) 

plt.tight_layout()
plt.show()

#-----------------------------------------------------------------#
# --- Task 11: Categorize jobs using Designation---
#-----------------------------------------------------------------#
df['designation'] = df['amr_designation'].astype(str).str.lower().str.replace('[^a-z0-9 ]', '', regex=True).str.strip()
# Optimized dictionary hitting typos, seasonal acronyms, and others after multiple rounds
conditions = [
    # Div 1: Managers (Added tl)
    df['designation'].str.contains(r'\b(?:manager|head|director|vp|president|lead|supervisor|teamleader|team leader|dsm|ssm|asm|mgr|incharge|csm|tl)\b', na=False),
    
    # Div 2: Professionals 
    df['designation'].str.contains(r'\b(?:engineer|developer|doctor|lawyer|accountant|analyst|educator|chemist|nurse|expert|trainer|ca|get)\b', na=False),
    
    # Div 5: Service and Sales Workers 
    df['designation'].str.contains(r'\b(?:sales|salesman|retail|relationship|promoter|pramoter|fashion|psr|isd|sec|sse|store|buyer|collection|advisor|adviser|isr|cro|xfe|ambassador|cre|visa|mdo|rso|fos|counsellor|oppo|cashier|telecaller|merchandiser|dsr|sd|consultant|fis|champ|isp|fa|demonstrator|fso|tso|dse|bdo|fse|ba|ro|pse|ase|dso|asr|cso|soc|re|specialist|delivery|biker|cook|stockist|po|usr|sc|vsr|vspr|abo|dbsr|se ayur|soho|sfa|las|csa|ssr|tfa|sr|fc|bde|lasm|rtf|bpt|cc|eca|tele caller|telecalling|salesperson|ambaassodar)\b', na=False),
    
    # Div 3: Technicians and Associate Professionals 
    df['designation'].str.contains(r'\b(?:technician|hdo|fiber|it support|technicain|techinican)\b', na=False),
    
    # Div 7: Craft and Related Trades Workers 
    df['designation'].str.contains(r'\b(?:wireman|electrician|mechanic|welder|plumber|carpenter|fitter|rigger|worker|trade|grinder|mech)\b', na=False),
    
    # Div 8: Plant and Machine Operators 
    df['designation'].str.contains(r'\b(?:operator|driver|machinist|forklift|batcher|processor)\b', na=False),
    
    # Div 9: Elementary Occupations 
    df['designation'].str.contains(r'\b(?:coolie|picker|sorter|helper|cleaner|sweeper|peon|loader|packer|handler|office boy|stock boy|house keeping|housekeeping|labour|runner)\b', na=False),
    
    # Div 4: Clerical Support Workers 
    df['designation'].str.contains(r'\b(?:clerk|admin|data entry|executive|exceutive|officer|associate|trainee|support|assistant|facilitator|intern|deo|coordinator|asi|executiveoperations|toll|collector|recruiter|team member|retainer|secretary|mis|back office|floater)\b', na=False)
]

choices = [
    'Managers & Senior Officials', 
    'Professionals', 
    'Service & Sales Workers',
    'Technicians & Assoc. Professionals', 
    'Craft & Related Trades', 
    'Plant & Machine Operators', 
    'Elementary Occupations', 
    'Clerical & Admin Support'
]

# Ensure the default fallback has no numbers
df['NCO_Division'] = np.select(conditions, choices, default='Unclassified / Other')

# Audit: Print the success rate
print("\nClassification Audit (Top Categories):")
print(df['NCO_Division'].value_counts())

# Filter the dataset to only look at the unclassified rows
unclassified_df = df[df['NCO_Division'] == 'Unclassified / Other']

# Count the exact designations inside this group
top_unclassified = unclassified_df['amr_designation'].value_counts().head(50)

print("\nTop 30 Unclassified Job Titles:")
print(top_unclassified)

#-----------------------------------------------------------------#
# --- Task 12: Bar Chart for attrition rate and categorization---
#-----------------------------------------------------------------#

plot_df = df.dropna(subset=['Event']).copy()

# Rename the 'Other' bucket so it reads properly (Removed the "0.")
plot_df['NCO_Division'] = plot_df['NCO_Division'].replace(
    'Unclassified / Other', 'Unclassified'
)

# 2. Aggregate Attrition Metrics
summary = plot_df.groupby('NCO_Division', observed=False).agg(
    Total_Workers=('Event', 'size'),
    Attrition_Rate=('Event', lambda x: x.mean() * 100)
).reset_index()

# 3. Sort highest attrition to the top
summary = summary.sort_values(by='Attrition_Rate', ascending=True)
summary['Attrition_Rate'] = summary['Attrition_Rate'].round(1)

# THE HORIZONTAL ATTRITION CHART

fig, ax = plt.subplots(figsize=(11, 7.5)) 
y_positions = np.arange(len(summary['NCO_Division']))

# Modern Corporate Palette
color_bar = '#8AB4F8'     
color_text = '#343A40'
color_muted = '#6C757D'

# Draw Horizontal Bars
bars = ax.barh(
    y_positions, 
    summary['Attrition_Rate'], 
    height=0.6,             
    color=color_bar,        
    edgecolor='white', 
    linewidth=1.5
)

# Add Data Labels
for i, bar in enumerate(bars):
    rate = summary['Attrition_Rate'].iloc[i]
    volume = summary['Total_Workers'].iloc[i]
    width = bar.get_width()
    
    # Inside Bar: Percentage 
    ax.annotate(
        f'{rate}%',
        xy=(width - (width * 0.03), bar.get_y() + bar.get_height() / 2), 
        ha='right', va='center',
        fontsize=11.5, fontweight='bold', color='white'
    )
    
    # Outside Bar: Base Volume Headcount
    ax.annotate(
        f" Base: {volume:,.0f}",
        xy=(width, bar.get_y() + bar.get_height() / 2),
        xytext=(8, 0), 
        textcoords="offset points",
        ha='left', va='center',
        fontsize=10.5, color=color_muted
    )

# Format Axes
ax.set_yticks(y_positions)
ax.set_yticklabels(summary['NCO_Division'], fontsize=11, fontweight='600', color=color_text)
ax.set_xlabel('Attrition Rate (%)', fontsize=12, color=color_muted, labelpad=10)

# Add 25% extra right-side headroom
ax.set_xlim(0, summary['Attrition_Rate'].max() * 1.25)

# Minimalist Styling & Borders
ax.set_title('Attrition Rate by Official Job Classification (NCO-2015)', 
             fontsize=16, pad=30, color=color_text, fontweight='bold', loc='left')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_color('#DEE2E6')
ax.spines['left'].set_linewidth(1.5)

# Faint vertical gridlines behind the bars
ax.xaxis.grid(True, linestyle='-', alpha=0.4, color='#CED4DA')
ax.tick_params(axis='x', colors='#ADB5BD', length=0)
ax.tick_params(axis='y', length=0) 
ax.set_axisbelow(True) 

# Render
plt.tight_layout()
plt.show()


    