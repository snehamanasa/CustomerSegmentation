#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd


# In[54]:


data=pd.read_csv('Mall_Customers.csv')


# In[55]:


data.head()


# In[56]:


data.tail()


# In[57]:


data.shape


# In[58]:


print("number of rows",data.shape[0])


# In[59]:


data.info()


# In[60]:


data.isnull().sum()


# In[61]:


data.describe()


# In[62]:


data.columns


# In[63]:


x = data[['Annual Income (k$)','Spending Score (1-100)']]
x


# In[64]:


from sklearn.cluster import KMeans


# In[65]:


k_means = KMeans()
k_means.fit(x)


# In[66]:


k_means = KMeans(n_clusters=5)
k_means.fit_predict(x)


# In[67]:


wcss=[]
for i in range(1,11):
    k_means = KMeans(n_clusters=i)
    k_means.fit(x)
    wcss.append(k_means.inertia_)


# In[68]:


wcss #within cluster sum of squares


# In[69]:


import matplotlib.pyplot as plt


# In[70]:


plt.plot(range(1,11),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# In[71]:


x = data[['Annual Income (k$)','Spending Score (1-100)']]


# In[72]:


k_means = KMeans(n_clusters=5,random_state =42)
y_means = k_means.fit_predict(x)


# In[73]:


y_means


# In[74]:


plt.scatter(x.iloc[y_means==0,0],x.iloc[y_means==0,1],s=100,c='red',label="Cluster 1")
plt.scatter(x.iloc[y_means==1,0],x.iloc[y_means==1,1],s=100,c='yellow',label="Cluster 2")
plt.scatter(x.iloc[y_means==2,0],x.iloc[y_means==2,1],s=100,c='green',label="Cluster 3")
plt.scatter(x.iloc[y_means==3,0],x.iloc[y_means==3,1],s=100,c='blue',label="Cluster 4")
plt.scatter(x.iloc[y_means==4,0],x.iloc[y_means==4,1],s=100,c='black',label="Cluster 5")
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100,c="magenta")
plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()


# In[75]:


k_means.predict([[80,80]])


# In[76]:


import joblib


# In[77]:


joblib.dump(k_means,"customer_segmentation")


# In[78]:


model = joblib.load("customer_segmentation")


# In[79]:


model.predict([[80,80]])


# In[80]:


#GUI 
from tkinter import *
import joblib


# In[179]:


def show_entry_fields():
    p1 = int(e1.get())
    p2 = int(e2.get())
    
    model = joblib.load('customer_segmentation')
    result = model.predict([[p1, p2]])
    
    cluster_labels = [
        "Customers with medium annual income and medium annual spend",
        "Customers with high annual income but low annual spend",
        "Customers with low annual income and low annual spend",
        "Customers low annual income but high annual spend",
        "Customers with high annual income and high annual spend"
    ]
    
    result_label.configure(text="This Customer belongs to cluster no: " + str(result[0]))
    cluster_desc_label.configure(text=cluster_labels[result[0]])

master = Tk()
master.title("Customer Segmentation")
master.geometry("500x200")

# Styling
master.configure(bg="#f2f2f2")
header_font = ("Helvetica", 14, "bold")
label_font = ("Helvetica", 12)
result_font = ("Helvetica", 12, "bold")

# Header label
header_label = Label(master, text="Customer Segmentation Using Machine Learning", font=header_font, bg="#f2f2f2")
header_label.grid(row=0, columnspan=2,padx=15, pady=10)

# Input labels and entry fields
Label(master, text="Annual Income", font=label_font, bg="#f2f2f2").grid(row=1, sticky=E)
Label(master, text="Spending Score", font=label_font, bg="#f2f2f2").grid(row=2, sticky=E)

e1 = Entry(master, font=label_font)
e2 = Entry(master, font=label_font)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)

# Predict button
predict_button = Button(master, text='Predict', command=show_entry_fields, font=label_font)
predict_button.grid(row=3, columnspan=2, pady=10)

# Result labels
result_label = Label(master, text="", font=result_font, bg="#f2f2f2")
result_label.grid(row=4, columnspan=2)

cluster_desc_label = Label(master, text="", font=label_font, bg="#f2f2f2")
cluster_desc_label.grid(row=5, columnspan=2)

mainloop()


# In[ ]:





# In[ ]:




