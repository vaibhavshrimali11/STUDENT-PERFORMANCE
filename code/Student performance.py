#!/usr/bin/env python
# coding: utf-8

# In[1]:


#LOADING IMPORTANT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# LOAD THE DATASET

df = pd.read_csv("C:\\Users\\cvyas\\Downloads\\archive (9)\\student-por.csv")


# In[3]:


#DISPLAY FIRST FEW ROWS
df.head()


# In[4]:


#GENERAL INFO ABOUT DATA
df.info()


# In[5]:


#CONVERT CATEGORICAL DATA TO BINARY DATA

df['schoolsup'] = df['schoolsup'].map({'yes':1,'no':0})


# In[6]:


df['schoolsup']


# In[7]:


df['internet'] = df['internet'].map({'yes':1,'no':0})


# In[8]:


df['internet']


# In[9]:


df['famsup'] = df['famsup'].map({'yes':1,'no':0})


# In[10]:


df['famsup']


# In[11]:


df['paid'] = df['paid'].map({'yes':1,'no':0})


# In[12]:


df['higher'] = df['higher'].map({'yes':1,'no':0})


# In[13]:


df['nursery'] = df['nursery'].map({'yes':1,'no':0})


# In[14]:


df['romantic'] = df['romantic'].map({'yes':1,'no':0})


# In[15]:


df['activities'] = df['activities'].map({'yes':1,'no':0})


# In[16]:


df.head()


# In[17]:


#MELT THE DATAFRAME  FOR COMPARISON OF G1 AND G2

df_melted = pd.melt(df, id_vars=['paid'], value_vars=['G1', 'G2'], 
                    var_name='Grade Type', value_name='Grade')

fig = px.bar(df_melted, x='paid', y='Grade', color='Grade Type',
             barmode='group', title='Comparison of G1 and G2 Grades by Paid Status',
             labels={'paid': 'Paid for Extra Classes', 'Grade': 'Grade'},
             color_discrete_map={'G1': '#004c6d', 'G3': '#d04c00'})

fig.update_layout(
    xaxis_title='Paid for Extra Classes',
    yaxis_title='Grade',
    legend_title='Grade Type',
    xaxis=dict(tickmode='linear'),  # Ensure x-axis shows all categories
    barmode='group'
)


fig.show()


# In[18]:


#no. of students with paid tutuions

unique_paid = df['paid'].value_counts()
print(unique_paid)

# only 39 students went for paid tutions


# In[19]:


plt.figure(figsize=(18, 6))

# Matplotlib box plot for G1 vs study time
plt.subplot(1, 3, 1)
plt.boxplot([df[df['studytime'] == i]['G1'] for i in sorted(df['studytime'].unique())], labels=sorted(df['studytime'].unique()))
plt.title('G1 vs Study Time (Matplotlib)')
plt.xlabel('Study Time')
plt.ylabel('G1')

# Seaborn box plot for G2 vs study time
plt.subplot(1, 3, 2)
sns.boxplot(x='studytime', y='G2', data=df)
plt.title('G2 vs Study Time (Seaborn)')
plt.xlabel('Study Time')
plt.ylabel('G2')


# In[20]:


plt.figure(figsize=(18, 6))

# Matplotlib scatter plot for absences vs G1
plt.subplot(1, 3, 1)
plt.scatter(df['absences'], df['G1'], alpha=0.5)
plt.title('Absences vs G1 (Matplotlib)')
plt.xlabel('Absences')
plt.ylabel('G1')

# Seaborn scatter plot for absences vs G2
plt.subplot(1, 3, 2)
sns.scatterplot(x='absences', y='G2', data=df, alpha=0.5)
plt.title('Absences vs G2 (Seaborn)')
plt.xlabel('Absences')
plt.ylabel('G2')

plt.tight_layout()
plt.show()


# In[21]:


# Plotly pie chart for Medu
fig_medu = px.pie(df, names='Medu', title="Mother's Education Level (Medu)")
fig_medu.show()

# Plotly pie chart for Fedu
fig_fedu = px.pie(df, names='Fedu', title="Father's Education Level (Fedu)")
fig_fedu.show()


# In[22]:


# Plotly pie chart for schoolsup
fig_schoolsup = px.pie(df, names='schoolsup', title='Students with School Support (schoolsup)')
fig_schoolsup.show()

# Plotly pie chart for famsup
fig_famsup = px.pie(df, names='famsup', title='Students with Family Support (famsup)')
fig_famsup.show()

# Plotly pie chart for paid
fig_paid = px.pie(df, names='paid', title='Students with Paid Classes (paid)')
fig_paid.show()

# Plotly pie chart for internet
fig_internet = px.pie(df, names='internet', title='Students with Internet Access (internet)')
fig_internet.show()


# In[28]:


# Create a bar chart ufor student failed in previous grades



df_counts = df['failures'].value_counts().reset_index()
df_counts.columns = ['failures', 'count']

fig_failures = px.bar(df_counts, x='failures', y='count',
                      title='Number of Students Who Failed in Previous Grades',
                      labels={'failures': 'Number of Failures', 'count': 'Number of Students'},
                      color='failures',
                      color_continuous_scale='Viridis',  # Choose a color scale that contrasts well
                      text='count')

fig_failures.update_layout(
    plot_bgcolor='white',  # Background color of the plotting area
    paper_bgcolor='lightgray',  # Background color of the entire figure
    font=dict(color='black'),  # Font color for axis labels and title
    xaxis_title_font=dict(size=14, color='black'),
    yaxis_title_font=dict(size=14, color='black'),
    title_font=dict(size=16, color='black'),
    xaxis=dict(showgrid=False, title_font=dict(size=14)),
    yaxis=dict(showgrid=False, title_font=dict(size=14))
)

# Update bar text visibility
fig_failures.update_traces(
    texttemplate='%{text}', 
    textposition='outside',  # Position text outside the bars
    marker=dict(color='darkblue')  # Bar color for better contrast
)

fig_failures.show()


# In[25]:


#Linear regression model for predicting grades using g1,and g2


# Step 1: Data Preparation
# Select the features and target variable
X = df[['G1', 'G2']]  # Features
y = df['G3']  # Target

# Step 2: Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Selection
model = LinearRegression()

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

comparison_df = pd.DataFrame({
    'Original Value': y_test.values,
    'Predicted Value': y_pred
})

print(comparison_df)

# Step 7: Making Predictions
# Predict future grades based on G1 and G2
new_data = pd.DataFrame({'G1': [15, 10], 'G2': [14, 12]})  # Example data
predicted_grades = model.predict(new_data)


# In[26]:


# visualizing comparison
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Predict G3 for the Test Set
y_pred = model.predict(X_test)

# Step 2: Visualize the Comparison
plt.figure(figsize=(10, 6))

# Scatter plot for actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Diagonal line
plt.title('Actual vs Predicted G3 Values')
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')

plt.show()

#  Display a table with actual and predicted values
comparison_df = pd.DataFrame({'Actual G3': y_test, 'Predicted G3': y_pred})
print(comparison_df.head())


# Results Summary
# The analysis of the student performance dataset revealed several insights:
# 
# Data Visualization:
# 
# Grades Comparison: Students who paid for extra classes tend to have higher grades in both G1 and G2.
# Support Systems: Pie charts showed distributions of students with family support, school support, and internet access.
# Failures: The bar chart highlighted the number of students with varying numbers of failures in previous grades.
# Exploratory Data Analysis:
# 
# Box Plots: Revealed variations in grades (G1 and G2) across different study times.
# Scatter Plots: Illustrated relationships between absences and grades, showing how absences can impact academic performance.
# Predictive Modeling:
# 
# Linear Regression: A model predicting final grades (G3) based on initial grades (G1 and G2) achieved a Mean Squared Error (MSE) of [MSE value] and an R-squared value of [R-squared value], indicating how well the model explains the variability in final grades.
# Overall, the analysis provided valuable insights into factors affecting student performance and demonstrated how initial grades can be used to predict final outcomes.
# 
# 
# 
# 
# 
# 
# 
