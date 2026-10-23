import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset properly
df = pd.read_csv('medical_claims_mini_project.csv')
df = df.dropna(axis=1, how='all')
df = df.loc[:, ~df.columns.str.match('^Unnamed')]

# Check shape and info for data understanding
print(df.shape)
print(df.info())
print(df.head())

# Fill missing ClaimAmount with median value to avoid errors in analysis
median_claim = np.nanmedian(df['ClaimAmount'].values)
df['ClaimAmount'] = df['ClaimAmount'].fillna(median_claim)

# Standardize HospitalType naming
df['HospitalType'] = df['HospitalType'].replace({'Govt': 'Government'})

# Remove outliers in ClaimAmount using 1.5*IQR method
Q1 = np.percentile(df['ClaimAmount'], 25)
Q3 = np.percentile(df['ClaimAmount'], 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['ClaimAmount'] >= lower_bound) & (df['ClaimAmount'] <= upper_bound)]

# Encode DiseaseCategory as numeric codes
disease_categories = df['DiseaseCategory'].astype('category')
df['DiseaseCode'] = disease_categories.cat.codes

# Define age groups
age_bins = np.array([0, 18, 35, 50, 65, 100])
age_labels = ['<18', '18-35', '36-50', '51-65', '65+']
df['AgeGroup'] = pd.cut(df['PatientAge'], bins=age_bins, labels=age_labels, right=False)

# Average claim cost by state
state_avg_cost = df.groupby('State')['ClaimAmount'].mean()
df['Avg_Treatment_Cost'] = df['State'].map(state_avg_cost)

# Claim approval rate by state
state_approval = df.groupby('State')['ClaimStatus'].value_counts(normalize=True).unstack()
df['Claim_Approval_Rate'] = df['State'].map(state_approval['Approved'])

# Relative expense index per state
overall_mean_cost = np.mean(df['ClaimAmount'])
df['State_Medical_Expense_Index'] = df['Avg_Treatment_Cost'] / overall_mean_cost

# Treatment days and hospital stay cost stats for flags
treatment_days_mean = np.mean(df['TreatmentDays'])
treatment_days_std = np.std(df['TreatmentDays'])
hospital_stay_cost_median = np.median(df['HospitalStayCost'])

df['HighTreatmentDays'] = np.where(df['TreatmentDays'] > (treatment_days_mean + treatment_days_std), True, False)
df['HighHospitalStayCost'] = np.where(df['HospitalStayCost'] > hospital_stay_cost_median, True, False)

# Define categorical claim amount ranges with explicit labels
claim_bins = [0, 25000, 50000, 100000]
claim_labels = ['Low (0-25k)', 'Medium (25k-50k)', 'High (50k-100k)']
df['ClaimAmountCategory'] = pd.cut(df['ClaimAmount'], bins=claim_bins, labels=claim_labels)

# Define treatment day ranges with descriptive labels
treatment_days_bins = [0, 5, 10, 20, 30]
treatment_days_labels = ['Very Low (0-5 days)', 'Low (6-10 days)', 'Medium (11-20 days)', 'High (21-30 days)']
df['TreatmentDaysCategory'] = pd.cut(df['TreatmentDays'], bins=treatment_days_bins, labels=treatment_days_labels, right=True)

# Print average and median claims by state
states = df['State'].unique()
for state in states:
    state_data = df[df['State'] == state]
    avg_claim = np.mean(state_data['ClaimAmount'])
    median_claim = np.median(state_data['ClaimAmount'])
    print(f"{state} - Avg Claim: {avg_claim:.2f}, Median Claim: {median_claim:.2f}")

# Plot Claim Amount distribution
plt.figure(figsize=(5,3))
sns.histplot(df['ClaimAmount'], bins=20, kde=True)
plt.title('Claim Amount Distribution')
plt.xlabel('Claim Amount (INR)')
plt.ylabel('Number of Claims')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Heatmap: Avg Claim Amount by State and Age Group
plt.figure(figsize=(5,3))
pivot_table = df.pivot_table(index='State', columns='AgeGroup', values='ClaimAmount', aggfunc='mean', observed=False)
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='coolwarm')
plt.title('State-Wise Avg Claim Amount(INR) by Age Group')
plt.xlabel('Patient Age Group')
plt.ylabel('State')
plt.tight_layout()
plt.show()

# Count plot of disease categories
plt.figure(figsize=(5,3))
sns.countplot(x='DiseaseCategory', data=df, order=df['DiseaseCategory'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Claims per Disease Category')
plt.xlabel('Disease Category')
plt.ylabel('Number of Claims')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Pie chart of claim status distribution
claim_counts = df['ClaimStatus'].value_counts()
plt.figure(figsize=(5,3))
plt.pie(claim_counts, labels=claim_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops={'width':0.7})
plt.title('Claim Status Distribution')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Bar plot of approval rate by state
grouped_df = df.groupby(['State', 'ClaimStatus']).size().unstack().fillna(0)
grouped_df['Approval_Rate'] = grouped_df['Approved'] / (grouped_df['Approved'] + grouped_df['Rejected'] + grouped_df.get('Pending', 0))

plt.figure(figsize=(7,3))
sns.barplot(x=grouped_df.index, y=grouped_df['Approval_Rate']*100)
plt.xticks(rotation=45)
plt.ylabel('Approval Rate (%)')
plt.title('Claim Approval Rate per State')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Bar plot counts of treatment days categories
treatment_group_counts = df['TreatmentDaysCategory'].value_counts().sort_index()
plt.figure(figsize=(5,3))
sns.barplot(x=treatment_group_counts.index, y=treatment_group_counts.values)
plt.title('Treatment Days Category Counts')
plt.xlabel('Treatment Days Category')
plt.ylabel('Number of Claims')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Bar plot counts of claim amount categories
claim_cat_counts = df['ClaimAmountCategory'].value_counts().sort_index()
plt.figure(figsize=(5,3))
sns.barplot(x=claim_cat_counts.index, y=claim_cat_counts.values)
plt.title('Claim Amount Category Counts')
plt.xlabel('Claim Amount Category (INR)')
plt.ylabel('Number of Claims')
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
