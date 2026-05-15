"""
    This file practices graphs on auto-mpg database.
    Author: Shatroopa Saxena
"""
# Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Apply the default theme
sns.set_theme()
sns.set_style("white")          # White background

# Load data & describe it
df1 = os.path.join("datasets", "auto-mpg.csv")
data_auto = pd.read_csv(df1)
# Clean
# data_auto = data_auto.replace('?', np.nan)
data_auto.replace('?', pd.NA, inplace=True)
data_auto["horsepower"] = pd.to_numeric(data_auto["horsepower"], errors="coerce")
data_auto = data_auto.dropna()
print(data_auto.info())
print(data_auto.describe())
print(data_auto)
print("-----------------------------------------------------")
df2 = os.path.join("datasets", "patient_records.csv")
data_patients = pd.read_csv(df2)
data_patients.replace('?', pd.NA, inplace=True)
data_patients = data_patients.dropna()
print(data_patients.info())
print(data_patients.describe())
print(data_patients)
print("-----------------------------------------------------")
df3 = os.path.join("datasets", "diabetes.csv")
data_diabetes = pd.read_csv(df3)
data_diabetes.replace('?', pd.NA, inplace=True)
data_diabetes = data_diabetes.dropna()
print(data_diabetes.info())
print(data_diabetes.describe())
print(data_diabetes)
print("-----------------------------------------------------")
df4 = os.path.join("datasets", "abalone", "abalone.data")
data_abalone = pd.read_csv(df4)
data_abalone.columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight", "Sucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
data_abalone.replace('?', pd.NA, inplace=True)
data_abalone = data_abalone.dropna()
print(data_abalone.info())
print(data_abalone.describe())
print(data_abalone)
print("-----------------------------------------------------")
df5 = os.path.join("datasets", "adult", "adult.data")
data_adult = pd.read_csv(df5)
data_adult.columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
data_adult.replace('?', pd.NA, inplace=True)
data_adult = data_adult.dropna()
print(data_adult.info())
print(data_adult.describe())
print(data_adult)
print("-----------------------------------------------------")


# Plot frequency polygram
sns.histplot(data_auto["model year"], element="poly", fill=False, stat="count", discrete=True)
plt.show()

sns.histplot(data_auto["displacement"], element="poly", fill=False, stat="count", bins=[50,100,150,200,250,300,350,400,450,500])
plt.show()

# Plot histogram
sns.histplot(data_patients["Diabetes"], discrete=True)
plt.xticks(ticks=[0, 1], labels=["no", "yes"])
plt.show()

sns.histplot(data_auto["horsepower"], discrete=False, bins=10)
plt.locator_params(axis='x', nbins=10)
plt.show()

# plot scatterplot
sns.scatterplot(data=data_diabetes, x="Glucose", y="BMI")
plt.show()

sns.scatterplot(data=data_diabetes, x="Glucose", y="Insulin", color="purple")
plt.show()

sns.scatterplot(data=data_auto, x="horsepower", y="mpg")
plt.show()

# Plot boxplots
sns.boxplot(data=data_auto, x="mpg", showmeans=True, color="pink", fill=False)
plt.show()

# Plot boxplot on top of histogram
fig, (ax_box, ax_hist) = plt.subplots(nrows=2, sharex=True, height_ratios=[0.25, 1])
sns.boxplot(data=data_auto, x="mpg", ax=ax_box, showmeans=True, color="orange", fill=True)
sns.histplot(data=data_auto, x="mpg", ax=ax_hist, bins=[0,5,10,15,20,25,30,35,40,45,50,55], color="purple", fill=False)
plt.xticks(ticks=[0,5,10,15,20,25,30,35,40,45,50,55])
sns.despine(ax=ax_box, left=True, bottom=True)
sns.despine(ax=ax_hist)
plt.show()

# Plot multiple graphs
fig, axs = plt.subplots(nrows=3, ncols=3)
for i, col in enumerate(data_abalone.columns):
    sns.histplot(data_abalone, x=col, ax=axs[i//3, i%3], color="magenta", shrink=0.8)
plt.tight_layout()
plt.show()

data_abalone.hist(figsize=(15, 10), color="cyan")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
cols = ["displacement", "horsepower", "weight", "acceleration", "mpg"]
for i, col_i in enumerate(cols):
    for j, col_j in enumerate(cols):
        if col_i != col_j:
            sns.scatterplot(data=data_auto, x=col_j, y=col_i, ax=axs[i, j])
        else:
            axs[i, j].text(0.5, 0.5, col_i, ha='center', va='center', fontsize=9)
            # Remove ticks on diagonal plots
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
plt.subplots_adjust(wspace=0.1, hspace=0.1)
for ax in axs.flat:
    ax.tick_params(axis='both', labelsize=7)
plt.tight_layout()
plt.show()

sns.pairplot(data_auto[cols], diag_kind="hist")
plt.show()

sns.pairplot(data_auto[cols], diag_kind="kde", corner=True)
plt.show()

sns.pairplot(data_auto[cols], diag_kind="auto", corner=True)
plt.show()

sns.boxplot(data=data_auto, x="mpg", y="model year", orient="h",showmeans=True, color="orange", fill=False)
plt.show()

# Highlighting trends for a particular value 
cols = ["cylinders", "displacement", "weight", "acceleration", "model year", "mpg"]
# Highlight only cylinder=8
palette = {val: "lightgray" if val not in (8, 4) else "red" if val == 8 else "blue" for val in data_auto["cylinders"].unique()}
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for i, col in enumerate(cols):
    sns.histplot(data=data_auto, x=col, hue="cylinders", ax=axs[i//3, i%3], palette=palette, multiple="stack")
plt.tight_layout()
plt.show()