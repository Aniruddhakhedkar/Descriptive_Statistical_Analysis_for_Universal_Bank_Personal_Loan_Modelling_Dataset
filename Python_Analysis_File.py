#!/usr/bin/env python
# coding: utf-8

# ## Project Title- Performing Descriptive Statistics with Python for an Universal Bank's Personal Loan Dataset

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt


# #### Question.1)-A statistics test was conducted for 10 learners in a class. The mean of their score is 85 and the variance of the score is zero. What can you interpret about the score obtained by all learners

# #### Answer-
# If the variance of the marks is zero, it implies that all 10 learners obtained the same marks, i.e. each learner achieved the identical score of 85. This indiates a same level of performance by 10 learners, indicating very high consistency in result.

# #### Question.2)- In a residential locality, the mean size of the house is 2224 square feet and the median value of the house is 1500 square feet. What can you interpret about the skewness in the distribution of house size? Are there bigger or smaller houses in the residential locality?

# #### Answer-
# 1) As the median of the variable is lower than the mean, the distribution is considered as a right/positively  skewed and asymmetric in nature.
# 2) Furthermore, as the difference between median and mean value is high (724 square feet), it indicates that there exists a significant spread of the variable and hence the residential locality will have bigger and smaller houses. Some of the bigger houses are shifting the mean of the distribution, thereby contributing to the variance.

# #### Question.3)-The following table shows the mean and variance of the expenditure for two groups of people.You want to compare the variability in expenditure for both groups with respect to their mean. Which statistical measure would you use to evaluate the variability in expenditure? Please provide an explanation for your answer.

# In[21]:


group_1_mean=500000
group_1_SD=125000
group_2_mean=40000
group_2_SD=10000

CV_group_1=(group_1_SD/group_1_mean)*100
print('Coefficient_of_Variation_for_Group_1=',CV_group_1)
CV_group_2=(group_2_SD/group_2_mean)*100
print('Coefficient_of_Variation_for_Group_2=',CV_group_2)


# #### Answer-
# 1) To evaluate the variability in expenditure for both groups with respect to their mean, I will prefer to use the statistical measure caled as the coefficient of variation (CV). The coefficient of variation measures relative variability, helpful when comparing data with different units or scales for consistency. 
# 
# 2) A lower CV indicates lower variability, while a higher CV indicates higher variabilityrelative to mean.
# 
# 3) In above case, as the CV for both the groups remains same, we can conclude that the variability in expenditures for these groups is same.

# #### Question.4)-During the survey, the ages of 80 patients infected by COVID and admitted to one of the city hospitals were recorded and the collected data is represented in the less than cumulative frequency distribution table.
# 
# a. Which class interval has the highest frequency?
# 
# b. Which age was affected the least?
# 
# c. How many patients aged 45 years and above were admitted?
# 
# d. Which is the modal class interval in the above dataset
# 
# e. What is the median class interval of age?

# #### Answer-
# 
# 1) Class interval (35-45) observes highest frequency.
# 
# 2) Age (55-65) was the less affected age, as it recorded only 5 observations.
# 
# 3) A total of 19 patients aged 45 years and above were admitted in the hospital.
# 
# 4) Class interval that observed highest no of observations is termed as model class. Hence in the given data set, the class (35-45) is termed as a model class.
# 
# 5) The median class for the given dataset is (35-45), as the median for the same is (35.87). It's calculation is given below.

# In[22]:


#Calculation for the median class-

CF=(21+11+6)
print('Cumulative frequency of the class preceding the median class (CF)=',CF)
f=(23)
print('Frequency of the median class (f)=',f)
L=(35)
print('Lower boundary of the median class (L)=',L)
h=(10)
print('Class width (h)=',h)
N=(6+11+21+23+14+5)
print('Total no of observations in a dataset (N)=',N)

Median=L+((N/2-CF)/f)*h
print('Median for the dataset=',Median)
print('The median class for the dataset is (35-45)')


# #### Question.5)-Assume you are the trader and you have invested over the years, and you are worried about the average return on investment. What average method would you use to compute the average return for the data given below?

# In[23]:


data={'Year':[2015,2016,2017,2018,2019,2020],
        'Return':[36,23,-48,-30,15,31],
     'Asset Price':[5000,6400,7980,9023,4567,3890]}
df=pd.DataFrame(data)
df

df['New Return']=1+df['Return']/100
x=df['New Return'].prod()
n=len(df)

g_mean=x**(1/n)

return_average = (g_mean-1)*100

return_average.round(2)


# #### Question.6)- Suppose you have been told to measure the average height of all the males on the earth. What would be your strategy for the same? Would the average height be a parameter or a statistic? Justify your answer.

# #### Answer-
# 
# 1) The global male population remained around 3.7 billion approximately till 2023. To calculate the average height of the all males, I will divide the globe into five distinct regions (North America, South America, Europe, Asia PAcific, and Middle east & Africa)
# 2) Then I will collect samples from these five distinct regions for the males having age above 24 years based on simple random sampling, as the sampling frame is not known to us.
# 3) Then I will calculate the average height of the males by using these samples, as they are representatiove of male population across the globe.
# 4) The average height of all males on Earth would be a parameter. Parameters describe characteristics of a population, in this case, all males globally. 
# 5) Statistics, on the other hand, refer to characteristics of a sample from that population, not the entire population itself.

# #### Question.7)- Calculate the z score of the following numbers:
# 
# X = [4.5,6.2,7.3,9.1,10.4,11]

# In[29]:


from scipy import stats
data=np.array([4.5,6.2,7.3,9.1,10.4,11.0])
print(data)

z_score=stats.zscore(data)
print('z-score of the given array=',z_score)


# In[32]:


df=pd.read_csv(r"C:\Users\aniru\Desktop\Descriptive Statistics_Graded_Project\Bank Personal Loan Modelling.csv")
df


# #### Dataframe Understanding- The above dataframe consists of a set of categorical and continuous variables and we have to segregate them to perform appropriate analysis.
# 
# Categorical variables Under the Study-
# 1) ID
# 2) ZIP Code
# 3) Education 
# 4) Personal Loan
# 5) Securities Account
# 6) CD Account
# 7) Online
# 8) Credit Card
# 
# Continuous Variables Under the Study-
# 1) Age
# 2) Experience
# 3) Income
# 4) Family
# 5) CCAvg
# 6) Mortgage 

# In[33]:


#Understanding the categorical variables in a detailed manner-
x=df[['ID','ZIP Code','Education','Personal Loan','Securities Account','CD Account','Online','CreditCard']]
print(x.isnull().sum())

x['Education']=x['Education'].replace({1:'Undergrad',2:'Graduate',3:'Advanced/Professional'})
x['Education'].value_counts()

x['Personal Loan']=x['Personal Loan'].replace({1:'Yes',0:'No'})
x['Personal Loan'].value_counts()

x['Securities Account']=x['Securities Account'].replace({1:'Yes',0:'No'})
x['Securities Account'].value_counts()

x['CD Account']=x['CD Account'].replace({1:'Yes',0:'No'})
x['CD Account'].value_counts()

x['Online']=x['Online'].replace({1:'Yes',0:'No'})
x['Online'].value_counts()

x['CreditCard']=x['CreditCard'].replace({1:'Yes',0:'No'})
x['CreditCard'].value_counts()

x['ZIP Code']=x['ZIP Code'].astype(object)
x['ID']=x['ID'].astype(object)
x.info()


# In[35]:


#Understanding the continuous variables in a detailed manner-
y=df[['Age','Experience','Income','Family','CCAvg','Mortgage']]
y.info()


# #### Question-8) Give us the statistical summary for all the variables in the dataset.

# In[36]:


print(y.describe().T) #For continuous variables
print(x.describe().T) #For categorical variables


# #### Question-9) Evaluate the measures of central tendency and measures of dispersion for all the quantitative variables in the dataset.

# In[37]:


#Measures of central tendency-
print('Mean for the variables--')
print(y.mean().T)
print('Median for the variables--')
print(y.median().T)
print('Mode for the variables--')
print(y.mode().T)

#Measures of dispersion- 
print('Variance for the variables--')
print(y.var().T)
print('Standard Deviation for the variables--')
print(y.std().T)
print('Skewness for the variables--')
print(y.skew().T)
print('Kurtosis for the variables--')
print(y.kurt().T)


# #### Question-10) What statistical method will you use to examine the presence of a linear relationship between age and experience variables? Also, create a plot to illustrate this relationship.

# 1) To determine the presence of linear relationship between age and experience, we will use correlation.
# 2) By using the pearson's correlation coefficient, we can determine the direction and strength of the relationship between these two variables.
# 3) In addition to that, we will use the scatter plot to visualize the relationship between these two variables. 

# In[38]:


a=y[['Age','Experience']].corr()
print(a)


# In[39]:


sns.scatterplot(x='Age',y='Income',data=y)


# #### Conclusion- 
# There exists a strong, non-linear, positive correlationship between Age and Income as,the value of pearson's correlation coefficient is higher than 0.85 and the scatter plot is not showcasing any kind of linearity pattern for the relationship between these two variables. 

# #### Question-11) What is the most frequent family size observed in this dataset?

# In[40]:


z=y['Family'].mode()
print(z)


# In[41]:


z=y['Family'].value_counts()
print(z)


# #### Conclusion- 
# The most frequest family size recorded in the data set is 1. 

# #### Question.12)- What is the percentage of variation you can observe in the ‘Income’ variable?

# In[45]:


print(y['Income'].skew())
# As the skewness value for the income variable lies inbetween (-1 to 1), i.e. aceptable range, 
# we can use mean and standard deviation to determine the percentage of variation. 

y['Percent_Variation_for_Income']=(y['Income'].std()/y['Income'].mean())*100
print(y['Percent_Variation_for_Income'].mean())
y


# #### Conclusion-
# The income variable showcases 62.4% of variation, which indicates that it has higher dispersion/spread.

# #### Question.13)-The ‘Mortgage’ variable has a lot of zeroes. Impute with some business logical value that you feel fit for the data.

# In[47]:


print(y['Mortgage'].mean()) #Before imputation
y['Mortgage'].replace({0:y['Mortgage'].mean()})
print(y['Mortgage'].isnull().sum())
sns.histplot(data=y, x='Mortgage')

#Here all the null values are filled by using the mean, as the mortgage variable is continuous in nature 
#and median for the variable is before replacing is coming as 0.


# #### Question.14)-Plot a density curve of the CCAvg variable for the customers who possess credit cards and write an interpretation about its distribution.

# In[49]:


new_df=pd.concat([x,y],axis=1)
new_df
new_df.info()

f1=new_df.loc[new_df['CreditCard'].str.contains('Yes')]
f1.info()
sns.kdeplot(data=f1,x='CCAvg')
plt.title('CCAvg_Density_Plot')

print('Skewnes for the CCAvg is-')
print(f1['CCAvg'].skew())
print('Kurtosis for the CCAvg is-')
print(f1['CCAvg'].kurt())
print('Variance for the CCAvg is-')
print(f1['CCAvg'].var())


# #### Interpretation of the density plot-
# The density plot indicates that the variable observes high spread and deviation from the mean.
# The plot also showcases that, the distribution of the variable is righly skewed (Asymmetric dstribution)
# Finally, as the value of kurtosis is 2.64, it can be considered as mesokurtic curve.

# #### Question.15)-Do you see any outliers in the dataset? If yes, what plot you would think will be suitable to showcase to the stakeholders?

# In[50]:


x2=y.drop(['Percent_Variation_for_Income'],axis=1)
x2
sns.boxplot(data=x2)
plt.title('Boxplot_for_Continuous_Variables_Under_Study')


# In[51]:


sns.boxplot(data=x2,x='CCAvg')


# ### Conclusion-
# There exists a distinct outliers in the dataser for variables namely, Income, CCAvg, and Mortgage.
# Box plot is used to visualize the presence of distinct outliers in the dataframe, hence we have used the boxplot to visualize the outliers for the above mentioned continuous variables. 

# #### Question.16)-Give us the decile values of the variable ‘Income’ in the dataset.

# In[52]:


a_1=x2['Income'].sort_values()
deciles = [a_1.quantile(i/10) for i in range(1, 10)]
print('Deciles_for_Income_Variable-',deciles)


# #### Question.17)-Give the IQR of all the variables which are quantitative and continuous.

# In[53]:


x2 #Dataframe containing all the continuous & quantitative variables.
A=x2.quantile(0.25)
B=x2.quantile(0.75)
IQR=(B-A)
print('IQR_for_Continuous_Variables_are_as_follows-')
print(IQR)


# #### Question.18)-Do the higher-income holders spend more on credit cards?

# To filterout the higher-income holders from the dataframe, 
# I have created a class for income holders present after the 60.0% (1/6th of decile) of income and called as high-income holders and accordingly created a new dataframe.

# In[54]:


x3=new_df.loc[(new_df['Income']>new_df['Income'].quantile(0.6)) & (new_df['CreditCard'].str.contains('Yes'))]
x3.info()

#x3-This dataframe contains information for those customers, 
#which has been using the credit card and coming in a newly created high-income category group.

print(new_df['Income'].quantile(0.6))
print(x3['Income'].min())
print(x3['Income'].max())


# In[209]:


#Mean of income for the newly created high income customer group-
print(x3['Income'].mean())

#Mean of Avg. Spending of credit cards for the newly created high income customer group-
print(x3['CCAvg'].mean())


# In[57]:


x4=new_df.loc[(new_df['Income']<new_df['Income'].quantile(0.6)) & (new_df['CreditCard'].str.contains('Yes'))]
x4.info()

#Mean of income for the remaining customers-
print(x4['Income'].mean())

#Mean of income for the remaining customers-
print(x4['CCAvg'].mean())


# #### Conclusion- 
# As the mean of annual income and avg. spending on credit cards per month for newly created higher income group is higher than the remaining customers(60%), we can to conclude that, higher income holders are spending more on credit cards

# #### Question.19)-How many customers use online banking? Do customers using bank internet facilities have higher incomes?

# In[62]:


x5=new_df.loc[(new_df['Online'].str.contains('Yes'))]
x5.info()
x6=new_df.loc[(new_df['Online'].str.contains('No'))]
x6.info()
print(x5['Online'].value_counts())
print('Mean for the customers that are using online banking-',x5['Income'].mean())
print('Mean for the customers that are not using online banking-',x6['Income'].mean())


# ### Conslusion-
# 1) A total of 2984 customers are using the online banking.
# 2) Customers who have opted for online banking have higher income, as their income mean (74.31) is higher than the mean of customers who have not opted for online services (72.97)

# #### Question.20)- Using the z-score of the income variable, find out the number of observations outside the +-3σ.

# In[221]:


from scipy.stats import zscore


# In[235]:


new_df['Income'].info()
mean=new_df['Income'].mean()
SD=new_df['Income'].std()
print(mean)
print(SD)
new_df['z_score'] = zscore(new_df['Income'])
outliers = new_df[abs(new_df['z_score']) > 3]
num_outliers = len(outliers)
outliers


# #### Conclusion- 
# There are a total of 2 outliers present outside the +-3σ range.

# In[ ]:




