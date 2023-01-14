#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# In[2]:


courses = pd.read_csv("dataset.csv")
courses = courses.drop_duplicates()
courses.head(10)


# In[3]:


courses.columns


# In[4]:


courses.info()


# In[5]:


courses_group= courses.groupby('course')['studentID'].count().sort_values(ascending=False).head(30)
courses_group.plot.bar()


# In[6]:


marks_rate = courses[courses['marks']<99]
courses['marks'].plot.hist()


# In[7]:


courses.groupby('course')['studentID'].count().plot.pie(autopct = "%1.1f%%")


# In[8]:


courses.groupby('subject')['studentID'].count().plot.bar()


# In[9]:


selected_features = ['studentID','subject',]
for feature in selected_features:
    courses[feature] = courses[feature].fillna('')


# In[10]:


def combine_col(row):
    courses_rate =str(row['marks']) 
    student_id = str(row['studentID'])
    combine_features =student_id+' ' +row['subject']
    return combine_features
courses["studentID with subject"] = courses.apply(combine_col, axis=1) 


# In[11]:


courses.head(10)


# In[12]:


features_vector = CountVectorizer()
count_matrix = features_vector.fit_transform(courses['studentID with subject'])
print("Count Matrix : ",count_matrix.toarray())
cosine_simi = cosine_similarity(count_matrix)


# In[13]:


student_input = input("Enter Your completed  subject: ")
student_previous_subject = student_input

def get_index_from_medicine(course):
    return courses[courses.subject == course]["indexValue"].values[0]
course_index = get_index_from_medicine(student_previous_subject)

similar_medicine = list(enumerate(cosine_simi[course_index]))
sorted_similar_course = sorted(similar_medicine, key=lambda x:x[1], reverse=True)

def get_course_from_index(index):
    return courses[courses.index == index]["course"].values[0]
i=0
for course_list in sorted_similar_course:
    print("--------------")
    print("|--Recommendation course with previous student's subject :>>>>"+get_course_from_index(course_list[0])+" --||-- ",course_list[1])
    i=i+1
    if i>7:
        break
        
    
    


# In[ ]:





# In[ ]:





# In[ ]:




