# ============================================================
# Honors Thesis Analysis
# Temporal Patterns in Intimate Partner Violence (Los Angeles)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

print("Loading dataset...")
df = pd.read_csv("data/la_ipv.csv")
print("Dataset loaded.")

# ------------------------------------------------------------
# 2. DATA CLEANING & FEATURE ENGINEERING
# ------------------------------------------------------------

# Clean TIME OCC
df['TIME OCC'] = pd.to_numeric(df['TIME OCC'], errors='coerce')
df = df.dropna(subset=['TIME OCC'])
df['hour'] = (df['TIME OCC'] // 100).astype(int)

# Clean DATE OCC
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df = df.dropna(subset=['DATE OCC'])

df['weekday'] = df['DATE OCC'].dt.day_name()
df['weekday_num'] = df['DATE OCC'].dt.weekday
df['month'] = df['DATE OCC'].dt.month
df['year'] = df['DATE OCC'].dt.year

# Weekend indicator
df['weekend'] = df['weekday_num'].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

print("Data cleaning complete.")
print(f"Total IPV incidents analyzed: {len(df)}")

# ------------------------------------------------------------
# 3. DESCRIPTIVE TEMPORAL ANALYSIS
# ------------------------------------------------------------

# ---- 3.1 Hourly Distribution ----
hour_counts = df.groupby('hour').size().reset_index(name='count')
hour_counts = hour_counts.sort_values('hour')

plt.figure()
plt.bar(hour_counts['hour'], hour_counts['count'])
plt.xlabel("Hour of Day")
plt.ylabel("Number of IPV Incidents")
plt.title("IPV Incidents by Hour of Day")
plt.xticks(range(0,24))
plt.show()


# ---- 3.2 Hour × Weekday Heatmap ----
heatmap_data = df.groupby(['weekday', 'hour']).size().unstack()

weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heatmap_data = heatmap_data.reindex(weekday_order)

plt.figure()
sns.heatmap(heatmap_data, cmap="Reds")
plt.title("IPV Incidents by Hour and Weekday")
plt.xlabel("Hour")
plt.ylabel("Day of Week")
plt.show()


# ---- 3.3 Monthly Trend ----
monthly = df.groupby(['year','month']).size().reset_index(name='count')
monthly['date'] = pd.to_datetime(monthly[['year','month']].assign(day=1))

plt.figure()
plt.plot(monthly['date'], monthly['count'])
plt.xlabel("Date")
plt.ylabel("Number of IPV Incidents")
plt.title("Monthly IPV Incidents Over Time")
plt.show()


# ------------------------------------------------------------
# 4. COMPARATIVE DISTRIBUTION ANALYSIS
# ------------------------------------------------------------

# ---- 4.1 Weekday vs Weekend Boxplot ----
plt.figure()
sns.boxplot(x='weekend', y='hour', data=df)

plt.title("IPV Incident Times: Weekday vs Weekend")
plt.xlabel("")
plt.ylabel("Hour of Day")
plt.show()


# ---- 4.2 Hour of Incident by Premise Type (Box Plot) ----

print("\nTop Premise Types:")
print(df['Premis Desc'].value_counts().head(10))

# Automatically select top 5 most common locations
top_locations = df['Premis Desc'].value_counts().head(5).index
df_location = df[df['Premis Desc'].isin(top_locations)]

plt.figure()
sns.boxplot(
    x='Premis Desc',
    y='hour',
    data=df_location
)

plt.title("Distribution of IPV Incident Hours by Location Type")
plt.xlabel("Location Type")
plt.ylabel("Hour of Day")
plt.xticks(rotation=45)
plt.show()


# ------------------------------------------------------------
# 5. SPATIAL DISTRIBUTION: IPV BY LAPD AREA
# ------------------------------------------------------------

area_counts = df['AREA NAME'].value_counts().sort_values(ascending=False)

plt.figure()
area_counts.plot(kind='bar')
plt.title("IPV Incidents by LAPD Area")
plt.ylabel("Number of Incidents")
plt.xlabel("LAPD Area")
plt.xticks(rotation=75)
plt.show()

area_percent = (area_counts / area_counts.sum()) * 100

plt.figure()
area_percent.plot(kind='bar')
plt.title("Percentage of IPV Incidents by LAPD Area")
plt.ylabel("Percentage of Total IPV Incidents")
plt.xticks(rotation=75)
plt.show()


# ------------------------------------------------------------
# 6. SOCIOECONOMIC CORRELATION ANALYSIS
# ------------------------------------------------------------

income_data = {
    "77th Street": 45000,
    "Southeast": 48000,
    "Southwest": 52000,
    "Newton": 47000,
    "Central": 50000,
    "Pacific": 95000,
    "West LA": 110000,
    "Wilshire": 90000,
    "Hollywood": 75000,
    "Topanga": 120000
}

income_series = pd.Series(income_data, name="Median Income")

area_df = area_counts.reset_index()
area_df.columns = ["AREA NAME", "IPV Count"]

area_df = area_df.merge(income_series, left_on="AREA NAME", right_index=True, how="left")
area_df = area_df.dropna()

print("\nArea Data with Income:")
print(area_df)

correlation = area_df["IPV Count"].corr(area_df["Median Income"])

print("\nCorrelation between IPV Count and Median Income:")
print(correlation)

plt.figure()
sns.regplot(
    x="Median Income",
    y="IPV Count",
    data=area_df
)

plt.title("Relationship Between Median Income and IPV Incidents by LAPD Area")
plt.xlabel("Median Household Income")
plt.ylabel("IPV Incident Count")
plt.show()


# ------------------------------------------------------------
# 7. STATISTICAL MODELING (Poisson Regression)
# ------------------------------------------------------------

hour_counts_model = df.groupby('hour').size().reset_index(name='count')
hour_counts_model['intercept'] = 1

poisson_model = sm.GLM(
    hour_counts_model['count'],
    hour_counts_model[['intercept', 'hour']],
    family=sm.families.Poisson()
)

results = poisson_model.fit()

print("\nPoisson Regression Results:")
print(results.summary())


# ------------------------------------------------------------
# 8. DEMOGRAPHIC CHARACTERISTICS (Victim Descent)
# ------------------------------------------------------------

race_counts = df['Vict Descent'].value_counts()

descent_map = {
    'W': 'White',
    'B': 'Black',
    'H': 'Hispanic/Latino',
    'A': 'Other Asian',
    'C': 'Chinese',
    'D': 'Cambodian',
    'F': 'Filipino',
    'G': 'Guamanian',
    'I': 'American Indian/Alaskan Native',
    'J': 'Japanese',
    'K': 'Korean',
    'L': 'Laotian',
    'O': 'Other',
    'P': 'Pacific Islander',
    'S': 'Samoan',
    'U': 'Hawaiian',
    'V': 'Vietnamese',
    'X': 'Unknown',
    'Z': 'Asian Indian'
}

race_counts.index = race_counts.index.map(lambda x: descent_map.get(x, x))
race_counts = race_counts[race_counts.index != 'Unknown']

plt.figure()
plt.pie(race_counts, labels=race_counts.index, autopct='%1.1f%%')
plt.title("Victim Descent Distribution in IPV Incidents")
plt.show()


# ------------------------------------------------------------
# 9. IPV Victims vs Los Angeles Population Comparison
# ------------------------------------------------------------

race_counts = df['Vict Descent'].value_counts()

descent_map = {
    'W': 'White',
    'B': 'Black',
    'H': 'Hispanic',
    'A': 'Asian',
    'C': 'Asian',
    'D': 'Asian',
    'F': 'Asian',
    'J': 'Asian',
    'K': 'Asian',
    'L': 'Asian',
    'V': 'Asian',
    'Z': 'Asian'
}

race_counts.index = race_counts.index.map(lambda x: descent_map.get(x, 'Other'))
race_grouped = race_counts.groupby(race_counts.index).sum()

ipf_percent = (race_grouped / race_grouped.sum()) * 100

la_population = {
    'Hispanic': 47.2,
    'White': 28.3,
    'Black': 8.5,
    'Asian': 12.0,
    'Other': 4.0
}

la_population_series = pd.Series(la_population)

comparison_df = pd.DataFrame({
    'IPV Victims (%)': ipf_percent,
    'LA Population (%)': la_population_series
}).fillna(0)

comparison_df.plot(kind='bar')
plt.title("IPV Victim Demographics vs Los Angeles Population")
plt.ylabel("Percentage")
plt.xticks(rotation=45)
plt.show()

print("\nAnalysis complete.")

# ------------------------------------------------------------
# 10. Hour of Incident by Gender (Box Plot)
# ------------------------------------------------------------

df_gender = df[df['Vict Sex'].isin(['F', 'M'])]

plt.figure()
sns.boxplot(
    x='Vict Sex',
    y='hour',
    data=df_gender
)

plt.title("Distribution of IPV Incident Hours by Gender")
plt.xlabel("Gender")
plt.ylabel("Hour of Day")

# Replace tick labels
plt.xticks([0,1], ['Female', 'Male'])

plt.show()