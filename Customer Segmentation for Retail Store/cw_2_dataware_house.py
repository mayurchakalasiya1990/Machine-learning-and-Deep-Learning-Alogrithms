#!/usr/bin/env python
# coding: utf-8

# In[86]:


#Set up and Read the data
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pandas_profiling
#! pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer,SilhouetteVisualizer


# ## **Create Sqlite DB and Load Data into Table**

# In[87]:


# read csv file
df_Fact = pd.read_csv('online_retail.csv')

# connect to database
conn = sqlite3.connect("cw")
cur = conn.cursor()

# load CRM data into the cw database
#df_Fact.to_sql("online_retail", conn)
df_Fact.head()


# In[88]:


df_Fact.info()


# ## **Data Cleaning Using SQL:**

# In[89]:


df_Fact = pd.read_sql("""select * from online_retail 
where InvoiceNo NOT LIKE '%C%'
AND CustomerID IS NOT NULL 
and CustomerID <> ""
AND unitprice != 0 """, conn)
df_Fact.head()


# ## **Distribution Analysis**

# In[90]:



#sns.pairplot(df_Fact.iloc[:, [4,6]])
#Distribution of age
plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(df_Fact['Quantity'])
plt.title('Distribution of Quantity', fontsize = 20)
plt.xlabel('Range of Quantity')
plt.ylabel('Count')

#sns.pairplot(df_Fact.iloc[:, [4,6]])
#Distribution of age
plt.figure(figsize=(10, 6))
sns.set(style = 'whitegrid')
sns.distplot(df_Fact['UnitPrice'])
plt.title('Distribution of UnitPrice', fontsize = 20)
plt.xlabel('Range of UnitPrice')
plt.ylabel('Count')


# In[91]:


#sns.displot(df_Fact, x="CustomerID", hue="StockCode", stat="density")


# 

# ## **Treatment of outliers**

# In[92]:



print("Highest allowed UnitPrice:",df_Fact['UnitPrice'].mean() + 3*df_Fact['UnitPrice'].std())
print("Lowest allowed UnitPrice:",df_Fact['UnitPrice'].mean() - 3*df_Fact['UnitPrice'].std())

print("Highest allowed Quantity:",df_Fact['Quantity'].mean() + 3*df_Fact['Quantity'].std())
print("Lowest allowed Quantity:",df_Fact['Quantity'].mean() - 3*df_Fact['Quantity'].std())


# In[93]:


print(df_Fact.shape)
df_Fact = pd.read_sql(""" select * from online_retail where InvoiceNo NOT LIKE '%C%'
AND InvoiceNo NOT LIKE '%C%'
AND CustomerID IS NOT NULL and CustomerID <> ""
AND unitprice != 0 
AND UnitPrice <= 69
AND Quantity <= 550 """, conn)
print(df_Fact.shape)
df_Fact.head()

import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df_Fact['UnitPrice'])
plt.subplot(1,2,2)
sns.distplot(df_Fact['Quantity'])
plt.show()

df_Fact = df_Fact


# ## **Correlation Analysis**

# In[94]:


'''
method :
pearson : standard correlation coefficient
kendall : Kendall Tau correlation coefficient
spearman : Spearman rank correlation
'''
df_Fact.corr(method ='pearson')


# In[95]:


duplicate = df_Fact[df_Fact.duplicated()]
 
print('Duplicate row count:',len(duplicate))


# ## **Data Visualization**

# In[65]:


# checking the different values for country in the dataset

plt.rcParams['figure.figsize'] = (12, 10)
a = df_Fact['Country'].value_counts().head(21)[1:]
sns.barplot(x = a.values, y = a.index, palette = 'PuBuGn_d')
plt.title('Top 20 Countries having Online Retail Market except UK', fontsize = 20)
plt.xlabel('Names of Countries')
plt.ylabel('Count')
plt.show()


# In[66]:


# looking at each country's sales
color = plt.cm.viridis(np.linspace(0, 1, 20))
df_Fact['Sales'] = df_Fact['UnitPrice'] * df_Fact['Quantity']
df_Fact['Sales'].groupby(df_Fact['Country']).agg('sum').sort_values(ascending = False).head(21)[1:].plot.bar(figsize = (15, 7),color = color)
#sns.barplot(x = b.values, y = b.index, palette = 'magma')
plt.title('Top 20 Sales of all the Countries Except UK', fontsize = 20)
plt.xlabel('Names of the Countries')
plt.ylabel('Number of sales')
plt.show()


# In[ ]:





# ## **Profile Summary Using Pandas Profiling**

# In[24]:


profile = df_Fact.profile_report(title="Online Retail Data Statistics Report")
profile.to_file(output_file="Online_Retail_Data_Statistics_Report.html")


# ## **RFM Segmentation**

# In[26]:


# Write clean CRM data into the database

# clean data and group transactions by Customerid
cleandata = pd.read_sql(''' SELECT CustomerID,
MIN(InvoiceDate) AS recent_order_date,
MAX(InvoiceDate) AS last_order_date,
COUNT(*) AS count_order,
SUM(unitprice*quantity) AS totalprice
FROM online_retail
WHERE InvoiceNo NOT LIKE '%C%'
AND CustomerID IS NOT NULL and CustomerID <> ""
AND unitprice != 0 
AND UnitPrice <= 200
AND Quantity <= 600
GROUP BY customerid 
ORDER BY customerid 
''', conn)

cleandata.head()

# Write clean CRM data into the database
cleandata.to_sql("clean_retail_data_8", conn)


# ## **RFM Segmentation**

# In[27]:


#clean the data and calculate rfm values
rmf_df = pd.read_sql(''' 
SELECT customerid,recent_order_date,last_order_date,count_order,totalprice, rfm_recency, rfm_frequency, rfm_monetary,rfm_recency*100 + rfm_frequency*10 + rfm_monetary AS rfm_combined
  FROM
      ( 
		SELECT customerid,recent_order_date,last_order_date, count_order,totalprice,
        NTILE(4) OVER (ORDER BY last_order_date) AS rfm_recency,
        NTILE(4) OVER (ORDER BY count_order) AS rfm_frequency,
        NTILE(4) OVER (ORDER BY totalprice) AS rfm_monetary
        FROM clean_retail_data_7
	)
 ''', conn)

rmf_df.head()


# ## *Customer Segment based on RFM ScoresCustomer Segment*

# In[28]:


def rfm_level(rmf_df):
  if ((rmf_df['rfm_recency'] >= 4) and (rmf_df['rfm_frequency'] >= 4) and (rmf_df['rfm_monetary'] >= 4)):
    return 'Best Customers'
  elif ((rmf_df['rfm_recency'] >= 3) and (rmf_df['rfm_frequency'] >= 3) and (rmf_df['rfm_monetary'] >= 3)):
    return 'Loyal'
  elif ((rmf_df['rfm_recency'] >= 3) and (rmf_df['rfm_frequency'] >= 1) and (rmf_df['rfm_monetary'] >= 2)):
    return 'Potential Loyalist'
  elif ((rmf_df['rfm_recency'] >= 3) and (rmf_df['rfm_frequency'] >= 1) and (rmf_df['rfm_monetary'] >= 1)):
    return 'Promising'
  elif ((rmf_df['rfm_recency'] >= 2) and (rmf_df['rfm_frequency'] >= 2) and (rmf_df['rfm_monetary'] >= 2)):
    return 'Customers Needing Attention'
  elif ((rmf_df['rfm_recency'] >= 1) and (rmf_df['rfm_frequency'] >= 2) and (rmf_df['rfm_monetary'] >= 2)):
    return 'At Risk'
  elif ((rmf_df['rfm_recency'] >= 1) and (rmf_df['rfm_frequency'] >= 1) and (rmf_df['rfm_monetary'] >= 2)):
    return 'Hibernating'
  else:
    return 'Lost'
#Create a new variable rfm_level
rmf_df['rfm_level'] = rmf_df.apply(rfm_level, axis=1)
rmf_df.head()


# ## *Customer Segment wise Customer count*

# In[29]:


rfm_agg = rmf_df.groupby('rfm_level').agg({'customerid':'count'})
rfm_agg


# In[30]:


#RFM visualization, you may have to install squarify
import squarify
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(13, 7)
squarify.plot(sizes=rfm_agg['customerid'],
label=['At Risk','Best Customers','Customers Needing Attention','Hibernating','Lost','Loyal','Potential Loyalist','Promising'], alpha=0.7)
plt.title("RFM Segments",fontsize=20)
plt.axis('off')
plt.show()


# 

# In[31]:


rmf_x = rmf_df.iloc[:,3:8]
# Standardizing the features
rmf_x = StandardScaler().fit_transform(rmf_x)
rmf_x


# In[32]:


pca = PCA()
principalComponents = pca.fit_transform(rmf_x)
principalComponents

print("pca.explained_variance_ratio_:",pca.explained_variance_ratio_)
print("pca.n_components_:",pca.n_components_)


# In[33]:


# dimension reduction using PCA 
pca_n = PCA(n_components=3)
pca_data = pca_n.fit_transform(rmf_x)
PCA_components = pd.DataFrame(pca_data)
PCA_components.head()


# In[123]:


wcss=[]
# fitting multiple k-means algorithms
for i in range(1, 22):
    kmeans = KMeans(n_clusters=i, init='k-means++').fit(pca_data)
    wcss.append(kmeans.inertia_)

print("The inertia of the clusters : ",wcss)


# In[124]:


# Elbow method with Yellowbrick Visualiser
visualizer = KElbowVisualizer(kmeans, k=(1,17))
visualizer.fit(pca_data)
visualizer.show()
visualizer.show(outpath="T1_EMG_YB.png")


# In[118]:


kmeans = KMeans(n_clusters=5, init ='k-means++')
y_kmeans = kmeans.fit_predict(pca_data)
centers = np.array(kmeans.cluster_centers_)
y_kmeans


# In[119]:


#K-means clustering with PCA results
data_pca_kmeans = pd.concat([rmf_df.reset_index(drop=True),pd.DataFrame(pca_data)],axis=1)
data_pca_kmeans.columns.values[-3:] = ["component 1","component 2","component 3"]
data_pca_kmeans["Retail K-means PCA"] = kmeans.labels_
data_pca_kmeans.to_excel (r'./export_dataframe_1.xlsx', index = False, header=True)
data_pca_kmeans.head()


# In[120]:


grouped_df = data_pca_kmeans.groupby(["Retail K-means PCA", "rfm_level"]).size()
grouped_df


# In[116]:


x_axis = data_pca_kmeans["component 1"]
y_axis = data_pca_kmeans["component 2"]
plt.figure(figsize = (10,8))

T=PCA_components.iloc[:,:4]

# store writing xvector yvector columns the values of PCA component in variable: for easy
xvector = pca.components_[0] * max(T[0])
yvector = pca.components_[1] * max(T[1])
columns = rmf_df.columns
    
#print(df_covid_data_pca_kmeans)
#palette =['g','r','c','m','y']
sns.scatterplot(x_axis,y_axis,hue=data_pca_kmeans["Retail K-means PCA"],palette="deep")
plt.title("Clusters by PCA Componets")
plt.show()


# In[109]:


# count of points in each of the above-formed clusters
frame = pd.DataFrame(pca_data)
frame['Cluster'] = y_kmeans
frame['Cluster'].value_counts()
frame['Cluster'].value_counts()


# In[122]:


#3D Plot as we did the clustering on the basis of 3 input features
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca_kmeans["component 2"][data_pca_kmeans['Retail K-means PCA'] == 0], data_pca_kmeans["component 3"][data_pca_kmeans['Retail K-means PCA']==0],data_pca_kmeans["component 1"][data_pca_kmeans['Retail K-means PCA']==0], c='purple', s=60)
ax.scatter(data_pca_kmeans["component 2"][data_pca_kmeans['Retail K-means PCA'] == 1], data_pca_kmeans["component 3"][data_pca_kmeans['Retail K-means PCA']==1],data_pca_kmeans["component 1"][data_pca_kmeans['Retail K-means PCA']==1], c='red', s=60)
ax.scatter(data_pca_kmeans["component 2"][data_pca_kmeans['Retail K-means PCA'] == 2], data_pca_kmeans["component 3"][data_pca_kmeans['Retail K-means PCA']==2],data_pca_kmeans["component 1"][data_pca_kmeans['Retail K-means PCA']==2], c='blue', s=60)
ax.scatter(data_pca_kmeans["component 2"][data_pca_kmeans['Retail K-means PCA'] == 3], data_pca_kmeans["component 3"][data_pca_kmeans['Retail K-means PCA']==3],data_pca_kmeans["component 1"][data_pca_kmeans['Retail K-means PCA']==3], c='green', s=60)
ax.scatter(data_pca_kmeans["component 2"][data_pca_kmeans['Retail K-means PCA'] == 4], data_pca_kmeans["component 3"][data_pca_kmeans['Retail K-means PCA']==4],data_pca_kmeans["component 1"][data_pca_kmeans['Retail K-means PCA']==4], c='yellow', s=60)
ax.view_init(35, 185)
plt.xlabel("component 1")
plt.ylabel("component 2")
ax.set_zlabel('component 3')
plt.show()

