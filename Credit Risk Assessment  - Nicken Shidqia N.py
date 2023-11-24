#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Assessment

# ## Import Libraries & Read Data

# In[1]:


#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#display all columns
pd.set_option('display.max_columns', None)

#display all row
pd.set_option('display.max_rows', None)


# In[2]:


#read data
df = pd.read_csv(r'D:\Dokumen\Portfolio Project\Credit Risk IDX Partners/loan_data_2007_2014.csv')
df.head()


# ## Data Cleaning

# In[3]:


#check rows & column
df.shape
print(f'row : {df.shape[0]}')
print(f'column : {df.shape[1]}')


# In[4]:


#drop columnn Unnamed: 0 because it is just index on dataset
df.drop(columns =['Unnamed: 0'], inplace = True)
df.head()


# In[5]:


#data understanding
df.info()


# In[6]:


#create subset based on column data type
categorical = df.select_dtypes(include = 'object')
numerical = df.select_dtypes(exclude = 'object')

#convert column to list
categorical_col = categorical.columns.to_list()
numerical_col = numerical.columns.to_list()


# In[7]:


df[numerical_col].describe()


# In[8]:


#check null values on numerical
df[numerical_col].isnull().sum()


# There are many features of numerical column that have null values :
# - annual_inc
# - delinq_2yrs
# - inq_last_6mths
# - mths_since_last_delinq
# - mths_since_last_record
# - open_acc
# - pub_rec
# - revol_util
# - total_acc
# - collections_12_mths_ex_med
# - mths_since_last_major_derog
# - annual_inc_joint              
# - dti_joint                     
# - verification_status_joint      
# - acc_now_delinq                   
# - tot_coll_amt                    
# - tot_cur_bal                     
# - open_acc_6m                   
# - open_il_6m                    
# - open_il_12m                    
# - open_il_24m                    
# - mths_since_rcnt_il            
# - total_bal_il                  
# - il_util                        
# - open_rv_12m                    
# - open_rv_24m                    
# - max_bal_bc                     
# - all_util                       
# - total_rev_hi_lim                
# - inq_fi                         
# - total_cu_tl                   
# - inq_last_12m                   

# In[9]:


#check null values on categorical
df[categorical_col].isnull().sum()


# There are many features of categorical column that have null values :
# - emp_title
# - emp_length
# - desc
# - last_pymnt_d              
# - next_pymnt_d           
# - last_credit_pull_d

# In[10]:


df[categorical_col].describe().T


# Insight :  
# Some features with date values still have object data type, need to convert it to datetime :
# - issue_d
# - last_pymnt_d
# - next_pymnt_d
# - last_credit_pull_d

# ### Check Duplicated Values

# In[11]:


#check duplicated values 
df.duplicated().sum()


# Insight :
# There is no duplicate data

# ### Check Missing Values

# In[12]:


#count total null in each column
sum_null = df.isnull().sum()

#count % of total null in each column
missing_percent = (sum_null * 100)/len(df)

#type of each column
df_type = [df[col].dtype for col in df.columns]

#create new dataframe for missing value 
df_isnull = pd.DataFrame({'total_null':sum_null,
                         'data_type':df_type,
                         'percentage_missing':missing_percent})

#sort percentage of missing value from largest to lowest
df_isnull.sort_values('percentage_missing', ascending = False, inplace = True)

#display all missing values
df_isnull_sort = df_isnull[df_isnull['percentage_missing']>0].reset_index()
df_isnull_sort


# ### Handling Missing Value

# In[13]:


#drop feature that have missing value > 50%
col_null = df_isnull.loc[df_isnull['percentage_missing']>50].index.tolist()
df.drop(columns = col_null, inplace = True)


# In[14]:


df.head(2)


# In[15]:


#replace missing values on feature tot_coll_amt, tot_cur_bal, total_rev_hi_lim with 0, 
#the assumption is the customer didnt' borrow again
for item in ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']:
    df[item] = df[item].fillna(0)


# **Notes :**  
# - 'tot_coll_amt' = Total collection amounts ever owed
# - 'tot_cur_bal' = Total current balance of all accounts
# - 'total_rev_hi_lim' = Total revolving high credit/credit limit

# In[16]:


df[['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']].head(2)


# In[17]:


#replace missing values on numerical category with median
numericals = df.select_dtypes(exclude = 'object')
for item in numericals:
    df[item] = df[item].fillna(df[item].median())


# In[18]:


numericals.head(2)


# In[19]:


#replace missing values on categorical with mode
categoricals = df.select_dtypes(include = 'object')
for item in categoricals:
    df[item] = df[item].fillna(df[item].mode().iloc[0])


# In[20]:


#check whether still have missing value
df.isnull().sum()


# ### Drop Unnecessary Column

# In[21]:


df.drop(columns = ['member_id','url','title','addr_state','zip_code','policy_code','application_type','emp_title'], inplace = True)
df_clean = df.copy()


# ## Feature Engineering

# ### Feature : 'earliest_cr_line','last_credit_pull_d','last_pymnt_d','issue_d','next_pymnt_d'

# In[22]:


#check date feature
df_clean[['earliest_cr_line','last_credit_pull_d','last_pymnt_d','issue_d','next_pymnt_d']].head(2)


# In[23]:


#change format to datetime
df_clean['earliest_cr_line']=pd.to_datetime(df_clean['earliest_cr_line'], format = '%b-%y')
df_clean['last_credit_pull_d']=pd.to_datetime(df_clean['last_credit_pull_d'], format = '%b-%y')
df_clean['last_pymnt_d']=pd.to_datetime(df_clean['last_pymnt_d'], format = '%b-%y')
df_clean['issue_d']=pd.to_datetime(df_clean['issue_d'], format = '%b-%y')
df_clean['next_pymnt_d']=pd.to_datetime(df_clean['next_pymnt_d'], format = '%b-%y')

#check date feature after conversion:
df_clean[['earliest_cr_line','last_credit_pull_d','last_pymnt_d','issue_d','next_pymnt_d']].head(2)


# **Notes :**  
# - 'earliest_cr_line' = The month the borrower's earliest reported credit line was opened
# - 'last_pymnt_d' = Last month payment was received
# - 'issue_d'= The month which the loan was funded
# - 'next_pymnt_d' = Next scheduled payment date

# In[24]:


#Adding new feature :
#pymnt_time = the number of month between 'next_pymnt_d' and 'last_pymnt_d'
#credit_pull_year = the number of year between 'last_credit_pull_d' and 'earliest_cr_line'

#adding pymnt_time
def diff_month(d1,d2):
    return (d1.year - d2.year)*12 + (d1.month - d2.month)
df_clean['pymnt_time'] = df_clean.apply(lambda x : diff_month(x.next_pymnt_d, x.last_pymnt_d), axis =1)

#adding credit_pull_year
def diff_year(d1,d2):
    return (d1.year - d2.year)
df_clean['credit_pull_year'] = df_clean.apply(lambda x : diff_year(x.last_credit_pull_d, x.earliest_cr_line), axis =1)

df_clean.head(2)


# In[25]:


#display pymnt_time < 0
df_clean[df_clean['pymnt_time']<0][['next_pymnt_d','last_pymnt_d','pymnt_time']]


# **Insight :**  
# There are negative values in pymnt_time, it means that the customer cannot afford to make a payment.   
# So the values gonna get replaced with 0

# In[26]:


#display credit_pull_year
df_clean[df_clean['credit_pull_year']<0][['last_credit_pull_d', 'earliest_cr_line', 'credit_pull_year']].head()


# **Insight :**  
# There are negative values in credit_pull_year, because there are false input in earliest_cr_line column.   
# So the values gonna get replaced with the maximum of the credit_pull_year feature.

# In[27]:


#replace the value of 'pymnt_time' and 'credit_pull_year'
df_clean.loc[df_clean['pymnt_time']<0] = 0
df_clean.loc[df_clean['credit_pull_year']<0] = df_clean['credit_pull_year'].max()


# In[28]:


#check if 'pymnt_time' and 'credit_pull_year' already getting replaced
df_clean[['pymnt_time','credit_pull_year']].sort_values(['pymnt_time','credit_pull_year'],ascending = [True, True]).head(5)


# ### Feature : Term

# In[29]:


#count term value
df_clean['term'].value_counts()


# In[30]:


#extract number from string with regex capture
df_clean['term'] = df_clean['term'].str.extract('(\d+)')


# In[31]:


#check whether the number already being extract
df_clean['term'].head()


# In[32]:


#count extract of term value
df_clean['term'].value_counts()


# In[33]:


#display feature data
df_clean.head(3)


# ## Exploratory Data Analysis (EDA)

# ### Removing Highly Correlated Features

# In[34]:


#copy dataset
df_eda = df_clean.copy()


# In[35]:


#drop id columns
df_eda.drop(columns = ['id'], inplace = True)


# In[36]:


#plot correlation matrix with 3 decimal places
plt.figure(figsize=(25,25))
sns.heatmap(df_eda.corr(), annot = True, fmt = '.3f')


# **Notes:**
# remove feature that have correlation > 0.9, because it could lead to biased result if left unchecked

# In[37]:


#correlation table with absolute number (no negative, only positive)
corr_matrix = df_eda.corr().abs()
corr_matrix


# **Notes :**  
# **Removing highly correlated features (Dimensionality Reduction in Python)**  
# Features that are perfectly correlated to each other, with a correlation coefficient of one or minus one, bring no new information to a dataset but do add to the complexity. So naturally, we would want to drop one of the two features that hold the same information. In addition to this we might want to drop features that have correlation coefficients close to one or minus one if they are measurements of the same or similar things. Not just for simplicity's sake but also to avoid models to overfit on the small, probably meaningless, differences between these values.

# **Notes :**
# Correlation coefficients whose magnitude are between 0.7 and 0.9 indicate variables which can be considered highly correlated. Because of that, for correlation > 0.9 are gonna removed

# In[38]:


#create and apply mask
mask = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype = bool), k=1))
mask


# In[39]:


#high correlated column
high_col = [col for col in mask.columns if any (mask[col] > 0.9)]
high_col


# In[40]:


#drop high correlated column
df_eda.drop(high_col, axis = 1, inplace = True)


# In[41]:


df_eda.head(2)


# In[42]:


#display categorical value
df_eda.select_dtypes(include = 'object').nunique()


# grade and sub_grade has similar interpretations, so drop the sub_grade, because already represented with grade

# In[43]:


#drop sub_grade column
df_eda.drop(['sub_grade'],axis =1, inplace =True)


# ### Univariate Analysis

# In[44]:


#create subset of numerical and categorical 
num = df_eda.select_dtypes(include = 'number').columns
category = df_eda.select_dtypes(include ='object').columns


# In[45]:


df_eda.select_dtypes(include ='object').head(2)


# #### Loan Status

# In[46]:


#plot loan status with descending order:
plt.figure(figsize=(15,10))
sns.countplot(y = df_eda['loan_status'], palette='Set2', order =df_eda['loan_status'].value_counts().index)
plt.title('Loan Status', fontsize=14)

for i,value in enumerate(df_eda['loan_status'].value_counts(normalize=True).mul(100).round(2)):
    text = f'{value} %'
    plt.text(value, i, text, fontsize=15, ha='left')

plt.xlabel('Count')
plt.ylabel('Loan Status')
plt.show()


# **Insight :**  
# - The majority of loan status distribution is current 47.95%, and fully paid 39.54%, it means that the borrower are meeting their payment obligation.  
# - There are significant number of charged off 9.08%. It means that the borrower has become delinquent on payments, and potential financial loss from the lender.
# - Late (31-120 days) is 1.48%. It means that there are delay in loan payment for 31-120 days, potential warnings that the lender has financial difficulties 
# - In grace period with 0.67%. The grace period is a time allowance for repaying the loan principal and interest within a certain period. And it is available after the payment due date (maturity). Need to do monitoring to minimize loan failure.
# - Doesn't meet the credit policy with 0.43%. It means some borrowers were proved, even though doesn't meet the standard credit criteria.  
#   
# From those insight, gonna create new feature called **loan_status** that indicates if the loan is considered good or bad.   
# - **Good loan** status is either current and fully paid.  
# - **Bad loan** status except for these 2 things.

# In[47]:


#copy new dataframe
df_loan = df_eda.copy()

#create list of good loan
good_loan = ['Current','Fully Paid']

#create new column 'target'
#if the value is 0 means good loan, if 1 means bad loan
df_loan['target'] = df_loan['loan_status'].apply(lambda x : 0 if x in good_loan else 1)


# In[48]:


#check if the column already being added
df_loan.head(2)


# In[49]:


df_loan['target'].value_counts()


# In[50]:


plt.figure(figsize=(15,10))
sns.countplot(y = df_loan['target'], palette='Set2', order =df_loan['target'].value_counts().index)
plt.title('Borrower Status Rate', fontsize=20)

for i,value in enumerate(df_loan['target'].value_counts(normalize=True).mul(100).round(2)):
    text = f'{value} %'
    plt.text(value, i, text, fontsize=15, ha='left', va='center')

plt.xlabel('Borrower Status Rate', fontsize=15)
plt.ylabel('percentage', fontsize=15)
plt.show()


# **Insight :**  
# - Good loan status got high percentage with 87.49%. It means that the bank's loan performing is good. It seems that the bank has good risk management and credit assessment.
# - Bad loan status got low percentage with 12.51%. It means the bank need to analyzing the characteristic of the borrower, so they could identify early warnings sign, and implement the mitigation from failure of pay loans from customers.

# #### Credit Purpose

# In[51]:


#plot number of loan purpose
plt.figure(figsize=(15,10))
sns.countplot(y = df_loan['purpose'], palette='Set2', order =df_loan['purpose'].value_counts().index)
plt.title('Number of Loan Purpose', fontsize=20)

for i,value in enumerate(df_loan['purpose'].value_counts(normalize=True).mul(100).round(2)):
    text = f'{value} %'
    plt.text(value, i, text, fontsize=15, ha='left', va='center')

plt.xlabel('Count', fontsize=15)
plt.ylabel('Purpose', fontsize=15)
plt.show()


# **Insight :**  
# - Debt consolidation got the highest percentage for load purpose with 58.67%. Debt consolidation is preferred because the customer can taking out a single loan or credit card to pay off multiple debts. The benefits of debt consolidation include a potentially lower interest rate and lower monthly payments.
# - Credit card got high percentage too with 22.27%. Credit card is preferred because it is typically offer all kinds of perks and benefits, including a one-time signing bonus for a new cardholder, cash back for purchases, rewards points, and frequent-flyer miles. Credit cards provide a level of safety for the user that a debit card and cash can't: fraud protection.
# - Home improvement got 5.67% for loan purpose. Home improvement is preferred because the borrower could get funds faster. Applying for a personal loan is generally a seamless process that can be completed online in minutes. Many online lenders also offer same or next-day funding, which means the borrower can get started on home improvements immediately. Also the fixed interest rate on a personal loan also means the monthly payment will stay the same, making it easier to work into spending plan each month over the life of the loan.

# #### Grade

# In[52]:


df_loan['grade'].value_counts()


# In[53]:


#plot number of grade
plt.figure(figsize=(15,10))
sns.countplot(y = df_loan['grade'], palette='Set2', order =df_loan['grade'].value_counts().index)
plt.title('Number of Grade', fontsize=20)

for i,value in enumerate(df_loan['grade'].value_counts(normalize=True).mul(100).round(2)):
    text = f'{value} %'
    plt.text(value, i, text, fontsize=15, ha='left', va='center')

plt.xlabel('Count', fontsize=15)
plt.ylabel('Grade', fontsize=15)
plt.show()


# **Insight :**  
# - Middle grade B and C got the highest percentage with 29.28% and 26.8%. It means that quality score to a loan based on a borrower's credit history, quality of the collateral, and the likelihood of repayment of the principal and interest are considered moderate.
# - Grade D with 16.45%. It means that the number of borrower that doesn't have good likelihood of repayment history is quite high.
# - Grade A with 16.01%. It means that quality score to a loan based on a borrower's credit history are considered high. It could lead with low risk of load failure.
# - Grade E,F,G got the lowest percentage. Grade E,F,G are high risk grade, because because the likelihood that the borrower will repay the loan is low. So the loan company need to tighten the criteria for loan borrowers.

# #### Loan Term

# In[54]:


df_loan['term'].value_counts()


# In[55]:


#plot number of loan term
plt.figure(figsize=(15,10))
sns.countplot(y = df_loan['term'], palette='Set2', order =df_loan['term'].value_counts().index)
plt.title('Number of Loan Term', fontsize=20)

for i,value in enumerate(df_loan['term'].value_counts(normalize=True).mul(100).round(2)):
    text = f'{value} %'
    plt.text(value, i, text, fontsize=15, ha='left', va='center')

plt.xlabel('Count', fontsize=15)
plt.ylabel('Loan Term', fontsize=15)
plt.show()


# **Insight :**  
# 36 month of loan term got the highest percentage with 72.47%. It means that short term loan are preferred by borrowers rather than long term loan. The reason could be because :  
# - Compared to long term loans, the amount of interest paid is significantly less. 
# - These loans are considered less risky compared to long term loans because of a shorter maturity date. 
# - Short term loans are the lifesavers of smaller businesses or individuals who suffer from less than stellar credit scores.

# #### Home Ownership

# In[56]:


df_loan['home_ownership'].value_counts()


# In[57]:


#plot number of home ownership
plt.figure(figsize=(15,10))
sns.countplot(y = df_loan['home_ownership'], palette='Set2', order =df_loan['home_ownership'].value_counts().index)
plt.title('Number of Home Ownership', fontsize=20)

for i,value in enumerate(df_loan['home_ownership'].value_counts(normalize=True).mul(100).round(2)):
    text = f'{value} %'
    plt.text(value, i, text, fontsize=15, ha='left', va='center')

plt.xlabel('Count', fontsize=15)
plt.ylabel('Home Ownership', fontsize=15)
plt.show()


# **Insight :**  
# - The borrower that has mortgage got the highest percentage on home ownership with 50.44%. The reason that mortgage customer is so many because a mortgage allows the customer to purchase a home without paying the full purchase price in cash. And for many people taking out a mortgage loan makes a property affordable because it would take too long to save up.  
# - The second highest is the borrower that rent their houses with 40.35%. The reason that customer choose rent rather than buying house is because no maintenance costs or repair bills, access to amenities like pool or fitness centre, no real estate taxes, and more flexibility as to where to live.
# - The borrower that own their houses is 8.9%. Only small portions for the customer that fully paid off their properties or acquired them without mortgage.

# ### Bivariate Analysis

# #### Borrower Status Rate by Grade

# In[58]:


#create cross tabulation of grade and target
grade_crosstab = pd.crosstab(df_loan['grade'],df_loan['target'], normalize='index').sort_values(by=1).round(2)
grade_crosstab


# In[59]:


#bar plot for grade and target
grade_bar = grade_crosstab.sort_values(by=1, ascending=False).plot(kind='barh', stacked=True, rot=0, colormap ='Pastel2')
grade_bar.legend(['Good', 'Bad'], bbox_to_anchor=(1,1), loc='upper left')

#add annotation
for i in grade_bar.containers:
    grade_bar.bar_label(i, label_type='center')
    
#set title and x,y axis label
plt.title('Borrower Status Rate by Grade', size=15)
plt.xlabel('Grade', size=12)
plt.ylabel('Percentage', size=12)


# **Insight :**  
# - Grade A has the most of good borrowers with 96%, and has the least of bad borrowers with only 4%.So Grade A has the least probability of loan default.
# - Grade G has the least of good borrowers with 66%, and has the most of bad borrowers with 34%. So Grade G has the most probability of loan default.
# - The lower the quality of a grade, the higher the number of bad borrowers, which will lead to a higher possibility of loan default.

# #### Borrower Status Rate by Last Credit Pull Year

# *) Notes :   
# Credit pull happens when borowers officially apply for credit, such as by filling out a credit card application. The last credit pull date shows if the borrower has recently applied for other loans or credit cards.

# In[60]:


#display needed column
df_loan[['earliest_cr_line','last_credit_pull_d','last_pymnt_d','issue_d','next_pymnt_d','target']].head(2)


# In[61]:


#convert to datetime
df_loan['last_credit_pull_d'] = pd.to_datetime(df_loan['last_credit_pull_d'], errors='coerce')


# In[62]:


#extract year from 'last_credit_pull_d'
df_loan['last_credit_pull_d_year'] = df_loan['last_credit_pull_d'].dt.year


# In[63]:


#check whether year already being extracted
df_loan['last_credit_pull_d_year'].head(2)


# In[64]:


#create cross tabulation for last_credit_pull_d_year and target
pull_d_year_ct = pd.crosstab(df_loan['last_credit_pull_d_year'],df_loan['target'])
pull_d_year_sum = pull_d_year_ct.sum(axis=1).astype(float)
pull_d_year_rate = pull_d_year_ct.div(pull_d_year_sum, axis=0)
pull_d_year_rate.index = np.arange(2007,2017)


# In[65]:


#create percentage of good and bad borrowers each year
good = pull_d_year_rate[0]
bad = pull_d_year_rate[1]


# In[66]:


#plot borrower's status rate
plt.figure(figsize=(10,7))
plt.plot(pull_d_year_rate.index, good, marker='D',linestyle ='-', color='#9FBB73' ,label = 'Good Borrower')
plt.plot(pull_d_year_rate.index, bad, marker='D',linestyle ='-', color='#EC8F5E' ,label = 'Bad Borrower')

#create percentage labels
for x,y in zip(pull_d_year_rate.index, good.round(2)):
    plt.text(x,y,f'{y*100}%', ha='center', va='bottom', color='black' )
    
for x,y in zip(pull_d_year_rate.index, bad.round(2)):
    plt.text(x,y,f'{y*100}%', ha='center', va='bottom', color='black')

#set tick labels
plt.xticks(np.arange(2007,2017,1))
plt.yticks(np.arange(0,1.1,0.1))

#set grid lines
plt.grid(True, linestyle='--', alpha=0.5)

#set title and x,y axis label
plt.title('Borrower Status Rate by Last Credit Pull Year', size=15)
plt.xlabel('Year', size=12)
plt.ylabel('Percentage', size=12)

#add legend
plt.legend()

#adjust plot margins so labels wont being cut off
plt.tight_layout()

plt.show()


# **Insight :**  
# - From 2007 to 2008, there was a slight increase 4% in the number of good borrowers. But from 2008 to 2009, there was a quite drastic decrease of 13%.
# - The upward trend for good borrower started from 2009 to 2016.This means that many borrowers pay their loans on time, and the credit selection for borrowers by loan companies is quite strict.
# - From 2007 to 2008, there was a slight decrease 5% in the number of bad borrowers. But from 2008 to 2009, there was a quite drastic increase of 21%.
# - The downward trend for bad borrower started from 2009 to 2016.This would be good signal for lenders company.

# #### Borrower Status Rate by Last Payment Date

# *) Last payment date is date of last payment received. The last payment date shows if the borrower is having difficulty making payments.

# In[67]:


#convert to datetime
df_loan['last_pymnt_d'] = pd.to_datetime(df_loan['last_pymnt_d'], errors='coerce')


# In[68]:


#extract year from 'last_pymnt_d'
df_loan['last_pymnt_d_year'] = df_loan['last_pymnt_d'].dt.year


# In[69]:


#create cross tabulation for last_pymnt_d and target
last_pymnt_d_year_ct = pd.crosstab(df_loan['last_pymnt_d_year'],df_loan['target'])
last_pymnt_d_year_sum = last_pymnt_d_year_ct.sum(axis=1).astype(float)
last_pymnt_d_year_rate = last_pymnt_d_year_ct.div(last_pymnt_d_year_sum, axis=0)
last_pymnt_d_year_rate.index = np.arange(2007,2017)


# In[70]:


#create percentage of good and bad borrowers each year
goods = last_pymnt_d_year_rate[0]
bads = last_pymnt_d_year_rate[1]


# In[71]:


#good and bad borrowers from 2008 to 2016
goods_1 = goods[1:]
bads_1 = bads[1:]


# In[72]:


#create range year from 2008 to 2016
range_year = last_pymnt_d_year_rate.index[1:]
range_year


# In[73]:


#plot borrower's status rate
plt.figure(figsize=(10,7))
plt.plot(range_year, goods_1, marker='D',linestyle ='-', color='#9FBB73' ,label = 'Good Borrower')
plt.plot(range_year, bads_1, marker='D',linestyle ='-', color='#EC8F5E' ,label = 'Bad Borrower')

#create percentage labels
for x,y in zip(range_year, goods_1.round(2)):
    plt.text(x,y,f'{y:.1%}', ha='center', va='bottom', color='black' )
    
for x,y in zip(range_year, bads_1.round(2)):
    plt.text(x,y,f'{y:.1%}', ha='center', va='bottom', color='black')

#set tick labels
plt.xticks(np.arange(2008,2017,1))
plt.yticks(np.arange(0,1.1,0.1))

#set grid lines
plt.grid(True, linestyle='--', alpha=0.5)

#set title and x,y axis label
plt.title('Borrower Status Rate by Last Payment Year', size=15)
plt.xlabel('Year', size=12)
plt.ylabel('Percentage', size=12)

#add legend
plt.legend()

#adjust plot margins so labels wont being cut off
plt.tight_layout()

plt.show()


# **Insight :**  
# - There is upward trend for good borrower by last payment year from 2008 to 2016. It means that the percentage of customers who have no difficulty in paying is getting higher. And this is good for the sustainability of the loan company's revenue.
# - There is downward trend for bad borrower by last payment year from 2008 to 2016. It means that the company's performance is very good in selecting loan applications by borrowers.
# - The difference percentage between good and bad borrowers for 2016 is really signicant with 98%. It means that the management could implement the stategy and policy in borrower eligibility and risk assessment. 

# #### Borrower's Status Interest Rate

# In[74]:


#create new dataframe
data_kde = df_loan[['last_pymnt_d_year', 'int_rate', 'target']]
data_kde.head(2)


# In[75]:


#select data from 2008 - 2016
kde_1 = data_kde[data_kde['last_pymnt_d_year']>2007].sort_values('last_pymnt_d_year', ascending = True)
kde_1.head(2)


# In[76]:


#average interest rate
int_avg = round(df_loan['int_rate'].mean(),2)

#average interest rate of good borrower
kde_good = kde_1[kde_1['target']==0]
int_good = round(kde_good['int_rate'].mean(),2)

#average interest rate of bad borrower
kde_bad = kde_1[kde_1['target']==1].sort_values('last_pymnt_d_year', ascending = True)
int_bad = round(kde_bad['int_rate'].mean(),2)

print(f'average interest rate: {int_avg}')
print(f'average good borrower int rate: {int_good}')
print(f'average bad borrower int rate: {int_bad}')


# In[77]:


#plot borrower's status rate
plt.figure(figsize=(8,6))

#create Kernel Density Estimation
sns.kdeplot(data = kde_1, x='int_rate', hue='target', palette='Set2')

#add vertical line at the average of int_rate
plt.axvline(kde_1['int_rate'].mean(), color = 'black', linestyle ='dashed', linewidth = 1, alpha=0.5)
plt.axvline(int_good, color = 'green', linestyle ='dashed', linewidth = 1, alpha=0.5)
plt.axvline(int_bad, color = 'orange', linestyle ='dashed', linewidth = 1, alpha=0.5)

#set the title
plt.title("Borrower's Status Interest Rate")

#set the legend
plt.legend(['Bad Borrower', 'Good Borrower'])

#create percentage labels
plt.text(0.62, 0.8, f"mean interest rate: {int_avg}\n"
         f"mean good borrower int rate: {int_good}\n"
         f"mean bad borrower int rate: {int_bad}"
         , bbox=dict(facecolor='w', alpha=0.5, pad=5),
        transform=plt.gca().transAxes, ha="left", va="top")

         
plt.tight_layout()
plt.show()


# Based on creditninja.com(2023), the loan interest rate obtained by the borrower depends on the credit score and loan term. A good credit score makes it possible to access lower interest rates when applying for a personal loan. Credit scores depend on the borrower's creditworthiness and financial stability as well as that of the lender. The average interest rate on personal loans for borrowers with excellent credit is between 10% and 12.5%. For a good credit score, the average rate is 13% – 16%. The lower the borrower's credit score, the higher the interest rate will be to offset the increased risk the lender assumes.
# 
# **Insight :**  
# - The average interest rate for all borrowers is 13.91%. 
# - The average interest rate for good borrowers is 13.54%. 
# - The average interest rate for bad borrowers is 15.9%.
# - The average loan interest rate for all borrowers at ID/X Partners is still relatively good because it is in the range of 13% – 16%. The reason why the average interest rate for bad borrowers is higher than for good borrowers is because the lower the borrower's credit score is, the higher the interest rates become to compensate for the increased risk the lender takes on.

# In[78]:


#function for automation
#create function for kde plot
def kda_chart(data, x, hue, palette, good, bad, title, size_x, size_y, text_all, value_all, text_good, value_good, text_bad, value_bad):
    #plot borrower's status rate
    plt.figure(figsize=(8,6))
    
    #create Kernel Density Estimation
    sns.kdeplot(data = data, x=x, hue=hue, palette=palette)
    
    #add vertical line at the average of int_rate
    plt.axvline(data[x].mean(), color = 'black', linestyle ='dashed', linewidth = 1, alpha=0.5)
    plt.axvline(good, color = 'green', linestyle ='dashed', linewidth = 1, alpha=0.5)
    plt.axvline(bad, color = 'orange', linestyle ='dashed', linewidth = 1, alpha=0.5)
    
     
    #create percentage labels
    plt.text(size_x, size_y, f"{text_all}: {value_all}\n"
         f"{text_good}: {value_good}\n"
         f"{text_bad}: {value_bad}"
         , bbox=dict(facecolor='w', alpha=0.5, pad=5),
        transform=plt.gca().transAxes, ha="left", va="top")
    
    #set the title
    plt.title(title)
    
    #set the legend
    plt.legend(['Bad Borrower', 'Good Borrower'])

    plt.tight_layout()
    plt.show()


# #### Borrower's Total Account

# *) Notes :  
# total_acc = The total number of credit lines currently in the borrower's credit file

# In[79]:


#create new dataframe
data_acc = df_loan[['last_pymnt_d_year', 'total_acc', 'target']]
data_acc.head()


# In[80]:


#average total account
acc_avg = round(data_acc['total_acc'].mean(),0)

#average total account of good borrower
data_acc_good = data_acc[data_acc['target']==0]
acc_good = round(data_acc_good['total_acc'].mean(),0)

#average total account of bad borrower
data_acc_bad = data_acc[data_acc['target']==1].sort_values('last_pymnt_d_year', ascending = True)
acc_bad = round(data_acc_bad['total_acc'].mean(),0)

print(f'average total account: {acc_avg}')
print(f'average good borrower total account: {acc_good}')
print(f'average bad borrower total account: {acc_bad}')


# In[81]:


#plot borrower's total account
kda_chart(data = data_acc, x ='total_acc',hue ='target', palette='Set2', size_x =0.55, size_y = 0.8,good=acc_good, bad=acc_bad, 
          title="Borrower's Total Account",text_all= 'mean total account', value_all=acc_avg,
          text_good = 'mean good borrower total account', value_good=acc_good,
          text_bad ='mean bad borrower total account', value_bad=acc_bad)


# **Insight :**  
# - A line of credit (LOC) is a preset borrowing limit that can be tapped into at any time. The borrower can take money out as needed until the limit is reached. According to (nerdwallet.com, 2023) suggests that 5 or more accounts — which can be a mix of cards and loans — is a reasonable number to build toward over time.
# - From the plot, average total account of good and bad borrower is 25 account, which is a lot more than recommended.  
# - Having too many open credit lines, even if borrower's not using them, can hurt their credit score by making the borrower's look more risky to lenders.  
# - Having multiple active accounts also makes it more challenging to control spending and keep track of payment due dates.  

# #### Borrower's Total Payment

# *) Notes :  
# total_pymnt = Payments received to date for total amount funded

# In[82]:


#create new dataframe
data_pymnt = df_loan[['last_pymnt_d_year', 'total_pymnt', 'target']]
data_pymnt.head()


# In[83]:


#average data payment
pymnt_avg = round(data_pymnt['total_pymnt'].mean(),2)

#average total payment of good borrower
data_pymnt_good = data_pymnt[data_pymnt['target']==0]
pymnt_good = round(data_pymnt_good['total_pymnt'].mean(),2)

#average total account of bad borrower
data_pymnt_bad = data_pymnt[data_pymnt['target']==1].sort_values('last_pymnt_d_year', ascending = True)
pymnt_bad = round(data_pymnt_bad['total_pymnt'].mean(),2)

print(f'average total payment: {pymnt_avg}')
print(f'average good borrower total payment: {pymnt_good}')
print(f'average bad borrower total payment: {pymnt_bad}')


# In[84]:


#plot borrower's total payment
kda_chart(data = data_pymnt, x ='total_pymnt',hue ='target', palette='Set2', size_x =0.5, size_y = 0.8,good=pymnt_good, 
          bad=pymnt_bad, 
          title="Borrower's Total Payment",text_all= 'mean total payment', value_all=pymnt_avg,
          text_good = 'mean good borrower total payment', value_good=pymnt_good,
          text_bad ='mean bad borrower total payment', value_bad=pymnt_bad)


# **Insight :**  
# - Total payment is Payments received to date for total amount funded. Amount financed is the actual amount of approved credit extended to a borrower in a loan from a lender, and if accepted, requires repayment by the borrower.
# - The average total amount funded for all customers is 11,512.
# - The average total amount funded for good borrowers is 12,129.
# - The average total amount funded for bad borrowers is 7,196.
# - Based on (investopedia.com, 2023), The average personal loan amount in America was 11,548 dollar in the second quarter of 2023 with average interest rate in Q2 is 11.48%.
# - The average total amount funded in ID/X Partner is relevant with the average personal loan amount in America.

# ## Feature Engineering with Weight of Evidence (WOE) and Information Value (IV)

# **Weight of evidence (WOE)** and **information value (IV)** evolved from the same logistic regression technique. **Logistic regression** is a process of modeling the probability of a discrete outcome given an input variable. The most common logistic regression models a binary outcome; something that can take two values such as true/false, yes/no, and so on. WOE and IV have been used as a benchmark to screen variables in the credit risk modeling projects such as probability of default

# **Weight of evidence (WOE)** generally described as a measure of the separation of good and bad customers. "Bad Customers" refers to the customers who defaulted on a loan. and "Good Customers" refers to the customers who paid back loan.  
# <img src="./image_IDX/woe.png" alt="woe" width = "250"/>    
# - Distribution of Goods - % of Good Customers in a particular group
# - Distribution of Bads - % of Bad Customers in a particular group
# - ln - Natural Log  
# 
# 1. Positive WOE means Distribution of Goods > Distribution of Bads
# 2. Negative WOE means Distribution of Goods < Distribution of Bads  
# Notes : Log of a number > 1 means positive value. If less than 1, it means negative value.

# **Information value (IV)** is one of the most useful technique to select important variables in a predictive model. It helps to rank variables on the basis of their importance.  
# <img src="./image_IDX/iv.png" alt="iv" width = "300"/>   
# <img src="./image_IDX/rule_iv.png" alt="rule" width = "300"/>

# In[85]:


#copy dataframe
df_fe = df_loan.copy()


# In[86]:


#create cross tabulation probability of good and bad borrowers
grade_crosstab2 = pd.crosstab(df_fe['grade'], df_fe['target'], normalize='index').sort_values(by=1).round(2)
grade_crosstab2_reset = grade_crosstab2.reset_index()
grade_crosstab2_reset.columns = ['grade','good','bad']
grade_crosstab2_reset


# In[87]:


#convert grade to string
df_fe['grade'] =  df_fe['grade'].astype(str)


# In[88]:


#delete row 0 and 47 in grade:
df_fe_new = df_fe[~(df_fe['grade'].isin(['0','47']))]


# In[89]:


#Weight of Evidence & Information Value
def woe(df,feature_name):
    feature =df.groupby(feature_name).agg(num_observation = ('target','count'), bad_loan_prob = ('target','mean'))
    feature['good_loan_prob'] = 1 - feature['bad_loan_prob']
    feature['grade_proportion'] = feature['num_observation']/feature['num_observation'].sum()
    feature['num_good_loan'] = feature['grade_proportion'] * feature['num_observation']
    feature['num_bad_loan'] = (1-feature['grade_proportion']) * feature['num_observation']
    
    #distribution of good
    feature['good_loan_prop'] = feature['num_good_loan']/feature['num_good_loan'].sum()
    
    #distribution of bad
    feature['bad_loan_prop'] = feature['num_bad_loan']/feature['num_bad_loan'].sum()
    
    #Weight of evidence (WOE)
    feature['weight of evidence'] = np.log(feature['good_loan_prop']/feature['bad_loan_prop'])
    
    #Information Value (IV)
    feature['information_value'] = (feature['good_loan_prop'] - feature['bad_loan_prop'])*feature['weight of evidence']
    feature['information_value'] = feature['information_value'].sum()
    
    #sort data
    feature = feature.sort_values('weight of evidence').reset_index()
    
    #display needed column
    feature_display = feature[[feature_name,'num_observation','good_loan_prob', 'good_loan_prop', 'bad_loan_prop', 'weight of evidence','information_value']]
    
    return feature_display


# ### WOE & IV Categorical Feature

# #### WOE & IV : `Grade`

# In[90]:


woe(df_fe_new,'grade')


# **Insight :**   
# *'grade'* is Medium predictive Power (IV in range 0.1 to 0.3)

# #### WOE & IV : `emp_length`

# In[91]:


woe(df_fe_new,'emp_length')


# **Insight :**   
# *'emp_length'* is Suspicious Predictive Power (IV > 0.5)

# #### WOE & IV : `home_ownership`

# In[92]:


woe(df_fe_new,'home_ownership')


# **Insight :**   
# *'home_ownership'* is Strong predictive Power (IV in range 0.3 to 0.5)

# #### WOE & IV : `verification_status`

# In[93]:


woe(df_fe_new,'verification_status')


# **Insight :**   
# *'verification_status'* is Not useful for prediction (IV < 0.02). So it will be dropped.

# #### WOE & IV : `purpose`

# In[94]:


woe(df_fe_new,'purpose')


# **Insight :**   
# *'purpose'* is Suspicious Predictive Power (IV > 0.5)

# #### WOE & IV : `initial_list_status`

# In[95]:


woe(df_fe_new,'initial_list_status')


# **Insight :**   
# *'initial_list_status'* is Strong predictive Power (IV in range 0.3 to 0.5)

# #### WOE & IV : `term`

# In[96]:


woe(df_fe_new,'term')


# **Insight :**   
# *'term'* is Suspicious Predictive Power (IV > 0.5)

# ### WOE & IV Numeric Feature

# In[97]:


#distribution plot
def dist(df_2, feature_name_2) :
    plt.figure(figsize=(2,4))
    sns.violinplot(df_2[feature_name_2], color = 'orange')
    print(f'number of unique value : {df_2[feature_name_2].nunique()}')
    print('Distribution :')
    print(df_2[feature_name_2].describe().T)
    plt.tight_layout()
    plt.show()


# #### WOE & IV : `loan_amnt`

# In[98]:


dist(df_fe_new, 'loan_amnt')


# In[99]:


#segment and sort data values into 10 bins
df_fe_new['loan_amnt_woe'] = pd.cut(x = df_fe_new['loan_amnt'], bins = 10)

#WOE & IV
woe(df_fe_new,'loan_amnt_woe')


# **Insight :**   
# *'loan_amnt'* is Medium predictive Power (IV in range 0.1 to 0.3)

# #### WOE & IV : `int_rate`

# In[100]:


dist(df_fe_new, 'int_rate')


# In[101]:


#segment and sort data values into 10 bins
df_fe_new['int_rate_woe'] = pd.cut(x = df_fe_new['int_rate'], bins = 10)

#WOE & IV
woe(df_fe_new,'int_rate_woe')


# **Insight :**   
# *'loan_amnt'* is Medium predictive Power (IV in range 0.1 to 0.3)

# #### WOE & IV : `annual_inc`

# In[102]:


dist(df_fe_new, 'annual_inc')


# In[103]:


#segment and sort data values into 10 bins
df_fe_new['annual_inc_woe'] = pd.cut(x = df_fe_new['annual_inc'], bins = 10)

#WOE & IV
woe(df_fe_new,'annual_inc_woe')


# In[104]:


#delete annual_inc_woe column
df_fe_new = df_fe_new.drop('annual_inc_woe', axis=1)


# In[105]:


#because the distribution of the number of observations based on bins is not evenly distributed and there is null values, 
#3 categories of annual income were created
df_fe_new['annual_inc_woe'] = np.where((df_fe_new['annual_inc']>0) & (df_fe_new['annual_inc']<200000),'Low income',
                                       np.where(df_fe_new['annual_inc']<=1500000, 'Medium income','High Income'))

#WOE & IV
woe(df_fe_new,'annual_inc_woe')


# **Insight :**   
# *'annual_inc'* information value is too high, so this variable gonna be dropped.   

# #### WOE & IV : `dti`

# In[106]:


dist(df_fe_new, 'dti')


# In[107]:


#segment and sort data values into 10 bins
df_fe_new['dti_woe'] = pd.cut(x = df_fe_new['dti'], bins = 10)

#WOE & IV
woe(df_fe_new,'dti_woe')


# **Insight :**   
# *'dti'* is Medium predictive Power (IV in range 0.1 to 0.3)

# #### WOE & IV : `delinq_2yrs`

# In[108]:


dist(df_fe_new, 'delinq_2yrs')


# In[109]:


#segment and sort data values into 10 bins
df_fe_new['delinq_2yrs_woe'] = pd.cut(x = df_fe_new['delinq_2yrs'], bins = 10)

#WOE & IV
woe(df_fe_new,'delinq_2yrs_woe')


# **Insight :**   
# *'delinq_2yrs'* information value is too high, so this variable gonna be dropped.   

# #### WOE & IV : `inq_last_6mths`

# In[110]:


dist(df_fe_new, 'inq_last_6mths')


# In[111]:


#segment and sort data values into 10 bins
df_fe_new['inq_last_6mths_woe'] = pd.cut(x = df_fe_new['inq_last_6mths'], bins = 10)

#WOE & IV
woe(df_fe_new,'inq_last_6mths_woe')


# **Insight :**   
# *'inq_last_6mths'* information value is too high, so this variable gonna be dropped.   

# #### WOE & IV : `open_acc`

# In[112]:


dist(df_fe_new, 'open_acc')


# In[113]:


#segment and sort data values into 10 bins
df_fe_new['open_acc_woe'] = pd.cut(x = df_fe_new['open_acc'], bins = 10)

#WOE & IV
woe(df_fe_new,'open_acc_woe')


# **Insight :**   
# *'open_acc'* is Suspicious Predictive Power (IV > 0.5)

# #### WOE & IV : `revol_bal`

# In[114]:


dist(df_fe_new, 'revol_bal')


# In[115]:


#segment and sort data values into 10 bins
df_fe_new['revol_bal_woe'] = pd.cut(x = df_fe_new['revol_bal'], bins = 10)

#WOE & IV
woe(df_fe_new,'revol_bal_woe')


# In[116]:


#delete revol_bal_woe column
df_fe_new = df_fe_new.drop('revol_bal_woe', axis=1)


# In[117]:


#because the distribution of the number of observations based on bins is not evenly distributed and there is null values, 
#4 categories of Total credit revolving balance were created
df_fe_new['revol_bal_woe'] = np.where((df_fe_new['revol_bal']>0) & (df_fe_new['revol_bal']<=5000),0,
                                       np.where(df_fe_new['revol_bal']<=10000, '1',
                                       np.where(df_fe_new['revol_bal']<=15000, '2','3')))

#WOE & IV
woe(df_fe_new,'revol_bal_woe')


# **Insight :**   
# *'revol_bal'* is Medium predictive Power (IV in range 0.1 to 0.3)  

# #### WOE & IV : `revol_util`

# In[118]:


dist(df_fe_new, 'revol_util')


# In[119]:


#segment and sort data values into 10 bins
df_fe_new['revol_util_woe'] = pd.cut(x = df_fe_new['revol_util'], bins = 10)

#WOE & IV
woe(df_fe_new,'revol_util_woe')


# In[120]:


#delete revol_util_woe column
df_fe_new = df_fe_new.drop('revol_util_woe', axis=1)


# In[121]:


#because the distribution of the number of observations based on bins is not evenly distributed and there is null values, 
#5 categories of revol_util_woe were created
df_fe_new['revol_util_woe'] = np.where((df_fe_new['revol_util']>0) & (df_fe_new['revol_util']<=20),'0',
                                       np.where(df_fe_new['revol_util']<=40, '1',
                                       np.where(df_fe_new['revol_util']<=60, '2',
                                       np.where(df_fe_new['revol_util']<=80, '3','4'))))

#WOE & IV
woe(df_fe_new,'revol_util_woe')


# **Insight :**   
# *'revol_util'* is Medium predictive Power (IV in range 0.1 to 0.3)

# #### WOE & IV : `total_acc`

# In[122]:


dist(df_fe_new, 'total_acc')


# In[123]:


#segment and sort data values into 10 bins
df_fe_new['total_acc_woe'] = pd.cut(x = df_fe_new['total_acc'], bins = 10)

#WOE & IV
woe(df_fe_new,'total_acc_woe')


# In[124]:


#delete total_acc_woe column
df_fe_new = df_fe_new.drop('total_acc_woe', axis=1)


# In[125]:


#because there is null values, 
#segment and sort data values into 7 bins
df_fe_new['total_acc_woe'] = pd.cut(x = df_fe_new['total_acc'], bins = 7)

#WOE & IV
woe(df_fe_new,'total_acc_woe')


# **Insight :**   
# *'total_acc'* is Medium predictive Power (IV in range 0.1 to 0.3)

# #### WOE & IV : `out_prncp`

# In[126]:


dist(df_fe_new, 'out_prncp')


# In[127]:


#segment and sort data values into 10 bins
df_fe_new['out_prncp_woe'] = pd.cut(x = df_fe_new['out_prncp'], bins = 10)

#WOE & IV
woe(df_fe_new,'out_prncp_woe')


# **Insight :**   
# *'out_prncp'* information value is too high, so this variable gonna be dropped.

# #### WOE & IV : `total_pymnt`

# In[128]:


dist(df_fe_new, 'total_pymnt')


# In[129]:


#segment and sort data values into 10 bins
df_fe_new['total_pymnt_woe'] = pd.cut(x = df_fe_new['total_pymnt'], bins = 10)

#WOE & IV
woe(df_fe_new,'total_pymnt_woe')


# **Insight :**   
# *'total_pymnt'* is Strong predictive Power (IV in range 0.3 to 0.5)

# #### WOE & IV : `total_rec_int`

# In[130]:


dist(df_fe_new, 'total_rec_int')


# In[131]:


#segment and sort data values into 10 bins
df_fe_new['total_rec_int_woe'] = pd.cut(x = df_fe_new['total_rec_int'], bins = 10)

#WOE & IV
woe(df_fe_new,'total_rec_int_woe')


# **Insight :**   
# *'total_rec_int'* information value is too high, so this variable gonna be dropped.

# #### WOE & IV : `total_rec_late_fee`

# In[132]:


dist(df_fe_new, 'total_rec_late_fee')


# In[133]:


#segment and sort data values into 10 bins
df_fe_new['total_rec_late_fee_woe'] = pd.cut(x = df_fe_new['total_rec_late_fee'], bins = 10)

#WOE & IV
woe(df_fe_new,'total_rec_late_fee_woe')


# **Insight :**   
# *'total_rec_late_fee'* information value is too high, so this variable gonna be dropped.

# #### WOE & IV : `recoveries`

# In[134]:


dist(df_fe_new, 'recoveries')


# In[135]:


#segment and sort data values into 10 bins
df_fe_new['recoveries_woe'] = pd.cut(x = df_fe_new['recoveries'], bins = 10)

#WOE & IV
woe(df_fe_new,'recoveries_woe')


# **Insight :**   
# *'recoveries'* information value is too high, so this variable gonna be dropped.

# #### WOE & IV : `last_pymnt_amnt`

# In[136]:


dist(df_fe_new, 'last_pymnt_amnt')


# In[137]:


#5 categories of last_pymnt_amnt_woe were created
df_fe_new['last_pymnt_amnt_woe'] = np.where((df_fe_new['last_pymnt_amnt']>0) & (df_fe_new['last_pymnt_amnt']<=500),'0',
                                       np.where(df_fe_new['last_pymnt_amnt']<=1000, '1',
                                       np.where(df_fe_new['last_pymnt_amnt']<=1500, '2',
                                       np.where(df_fe_new['last_pymnt_amnt']<=3500, '3','4'))))

#WOE & IV
woe(df_fe_new,'last_pymnt_amnt_woe')


# **Insight :**   
# *'last_pymnt_amnt'* is Suspicious Predictive Power (IV > 0.5)

# #### WOE & IV : `tot_coll_amt`

# In[138]:


dist(df_fe_new, 'tot_coll_amt')


# In[139]:


#4 categories of tot_coll_amt were created
df_fe_new['tot_coll_amt_woe'] = np.where((df_fe_new['tot_coll_amt']>0) & (df_fe_new['tot_coll_amt']<=50),0,
                                      np.where(df_fe_new['tot_coll_amt']<=100, '1',
                                      np.where(df_fe_new['tot_coll_amt']<=150, '2','3')))

#WOE & IV
woe(df_fe_new,'tot_coll_amt_woe')


# **Insight :**   
# *'tot_coll_amt'* information value is too high, so this variable gonna be dropped.

# #### WOE & IV : `tot_cur_bal`

# In[140]:


dist(df_fe_new, 'tot_cur_bal')


# In[141]:


#segment and sort data values into 5 bins
df_fe_new['tot_cur_bal_woe'] = pd.cut(x = df_fe_new['tot_cur_bal'], bins = 5)

#WOE & IV
woe(df_fe_new,'tot_cur_bal_woe')


# **Insight :**   
# *'tot_cur_bal_woe'* information value is too high, so this variable gonna be dropped)

# #### WOE & IV : `total_rev_hi_lim`

# In[142]:


dist(df_fe_new, 'total_rev_hi_lim')


# In[143]:


#4 categories of total_rev_hi_lim were created
df_fe_new['total_rev_hi_lim_woe'] = np.where((df_fe_new['total_rev_hi_lim']>0) & (df_fe_new['total_rev_hi_lim']<=500),0,
                                      np.where(df_fe_new['total_rev_hi_lim']<=1000, '1',
                                      np.where(df_fe_new['total_rev_hi_lim']<=2000, '2','3')))

#WOE & IV
woe(df_fe_new,'total_rev_hi_lim_woe')


# **Insight :**   
# *'total_rev_hi_lim'* information value is too high, so this variable gonna be dropped

# #### WOE & IV : `pymnt_time`

# In[144]:


dist(df_fe_new, 'pymnt_time')


# In[145]:


#4 categories of pymnt_time were created
df_fe_new['pymnt_time_woe'] = np.where((df_fe_new['pymnt_time']>0) & (df_fe_new['pymnt_time']<=1),0,
                                       np.where(df_fe_new['pymnt_time']<=6, '1',
                                       np.where(df_fe_new['pymnt_time']<=12, '2','3')))

#WOE & IV
woe(df_fe_new,'pymnt_time_woe')


# **Insight :**   
# *'pymnt_time'* is Medium predictive Power (IV in range 0.1 to 0.3)

# #### WOE & IV : `credit_pull_year`

# In[146]:


dist(df_fe_new, 'credit_pull_year')


# In[147]:


#segment and sort data values into 10 bins
df_fe_new['credit_pull_year_woe'] = pd.cut(x = df_fe_new['credit_pull_year'], bins = 10)

#WOE & IV
woe(df_fe_new,'credit_pull_year_woe')


# **Insight :**   
# *'credit_pull_year'* is Strong predictive Power (IV in range 0.3 to 0.5)

# #### Drop Feature No Needed

# There are feature that gonna be dropped because :
# - Information Value (IV) < 0.02, The variable is Not useful for prediction
# - Information Value (IV) > 0.5, The variable is Suspicious Predictive Power

# In[148]:


#drop list
drop_list = ['verification_status',
             'delinq_2yrs',
             'inq_last_6mths',
             'out_prncp',
             'total_rec_int',
             'total_rec_late_fee',
             'recoveries',
             'tot_coll_amt',
             'tot_cur_bal',
             'total_rev_hi_lim']


# In[149]:


#copy datataset
df_fe_clean = df_fe.copy()


# In[150]:


#drop list from dataset
df_fe_clean = df_fe_clean.drop(drop_list, axis = 1)


# In[151]:


print(f'Before Feature Engineering Using WOE and IV, there are {df_fe.shape[1]} columns')
print(f'After Feature Engineering Using WOE and IV, there are {df_fe_clean.shape[1]} columns')


# ## Feature Encoding

# Machine learning models can only work with numerical values. For this reason, it is necessary to transform the categorical values of the relevant features into numerical ones. This process is called **feature encoding**. 

# In[152]:


#copy new dataset
df_encode = df_fe_clean.copy()


# ### Label Encoding

# - Label encoding is probably the most basic type of categorical feature encoding method.
# - Label encoding doesn’t add any extra columns to the data but instead assigns a number to each unique value in a feature.
# - It saved a lot of room and don’t add more columns to data, resulting in a much cleaner look for the data.
# - An obvious benefit to label encoding is that it’s quick, easy, and doesn’t create a messy data frame in the way that one-hot encoding adds a lot of columns.

# In[153]:


#delete 0 and 47 row
df_encodes = df_encode[~((df_encode['home_ownership'].isin([0,47]))
                        & (df_encode['initial_list_status'].isin([0,47])))
                        ]


# In[154]:


#categorical value
cat_encode = df_encodes.select_dtypes(include = 'object')
cat_encode.head(2)


# In[155]:


#count home ownership
df_encodes['home_ownership'].value_counts()


# 'None' and 'Any' will be combined with 'Other' because have same meaning

# In[156]:


#replace row value label
df_encodes.loc[df_encodes['home_ownership'].isin(['NONE','ANY']),'home_ownership'] = 'OTHER'


# In[157]:


#check updated home_ownership
df_encodes['home_ownership'].value_counts()


# In[158]:


#value count 'term'
df_encodes['term'].value_counts()


# In[159]:


#The number of payments on the loan. Values are in months and can be either 36 or 60.
#replace term '36' with 0
#replace term '60' with 1
df_encodes['term'] = np.where(df_encodes['term'] == '36',0,1)


# In[160]:


#value count 'initial_list_status'
df_encodes['initial_list_status'].value_counts()


# In[161]:


#The initial listing status of the loan. Possible values are – Whole and Fractional
#replace initial_list_status 'f' with 0
#replace initial_list_status 'w' with 1
df_encodes['initial_list_status'] = np.where(df_encodes['initial_list_status'] == '36',0,1)


# ### One Hot Encoding Categoric

# - **One-Hot encoding** technique is used when the features are nominal(do not have any order). 
# - In one hot encoding, for every categorical feature, a new variable is created. 
# - Categorical features are mapped with a binary variable containing either 0 or 1. 
# - Here, 0 represents the absence, and 1 represents the presence of that category.
# - These newly created binary features are known as **Dummy variables.**

# In[162]:


df_encodes.head(2)


# In[163]:


#create dummy encoding
for i in [['home_ownership','purpose','emp_length','grade']]:
    onehots = pd.get_dummies(df_encodes[i], prefix = i)


# In[164]:


#display one hot encoding
onehots.head(2)


# In[165]:


onehots.info()


# ### Numerical Features Encode

# - **Encoding numerical features** refers to the process of representing numerical data in a format suitable for machine learning algorithms.
# - Numerical features are continuous values
# - The simplest form of encoding numerical columns using **Binarization**
# - In “binarization,” continuous variables are transformed into binary values (0 or 1) based on a predetermined threshold. Using this method, we can identify if a data point is above or below the threshold.

# In[166]:


#access column for numerical categories
num = df_encodes.select_dtypes(include = 'number').columns
num


# In[167]:


#divide dataset
#create new data for manual bin
manual_bin = df_encodes[['last_pymnt_amnt','revol_util','revol_bal','pymnt_time','term','target','annual_inc']]

#create new data for auto bin
auto_bin = df_encodes[num.drop(['last_pymnt_amnt','revol_util','revol_bal',
                                'pymnt_time','term','target','annual_inc',
                               'collection_recovery_fee','last_credit_pull_d_year','last_pymnt_d_year'])]


# In[168]:


manual_bin.head(4)


# In[169]:


auto_bin.head(2)


# penjelasan terkait function :
# 
# #function for segment bins
# def bins_df():
#     auto_bin['loan_amnt'] = pd.cut(auto_bin['loan_amnt'],bins=10)
#     return df_encode
# 
# #function convert to dummies
# def dummy_df():
#     dum = pd.get_dummies(auto_bin['loan_amnt'],prefix='loan_amnt')

# In[170]:


#function for segment bins
def bins_df(df, feature, bin_num):
    df[feature] = pd.cut(df[feature], bins = bin_num)
    return df

#function convert to dummies
def dummy_df(df_bin, feature):
    dum = pd.get_dummies(df_bin[feature], prefix=feature)
    return dum


# In[171]:


#segment bins & dummies
#loan_amnt
loan_amnt_bin = bins_df(auto_bin, 'loan_amnt', 10)
loan_amnt_dum = dummy_df(loan_amnt_bin , 'loan_amnt')

#int_rate
int_rate_bin = bins_df(auto_bin, 'int_rate', 10)
int_rate_dum = dummy_df(int_rate_bin , 'int_rate')

#dti
dti_bin = bins_df(auto_bin, 'dti', 10)
dti_dum = dummy_df(dti_bin , 'dti')

#open_acc
open_acc_bin = bins_df(auto_bin, 'open_acc', 10)
open_acc_dum = dummy_df(open_acc_bin , 'open_acc')

#total_acc
total_acc_bin = bins_df(auto_bin, 'total_acc', 7)
total_acc_dum = dummy_df(total_acc_bin , 'total_acc')

#total_pymnt
total_pymnt_bin = bins_df(auto_bin, 'total_pymnt', 10)
total_pymnt_dum = dummy_df(total_pymnt_bin , 'total_pymnt')

#credit_pull_year
credit_pull_year_bin = bins_df(auto_bin, 'credit_pull_year', 10)
credit_pull_year_dum = dummy_df(credit_pull_year_bin , 'credit_pull_year')


# In[172]:


#concatenate dummies feature
num_auto_bin = pd.concat([loan_amnt_dum, dti_dum, int_rate_dum, open_acc_dum, 
                          total_acc_dum, total_pymnt_dum, credit_pull_year_dum], axis= 1)


# In[173]:


#display num_auto_bin
num_auto_bin.head(2)


# In[174]:


manual_bin.describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99])


# In[175]:


#segment feature in manual_bin dataset
#revol_bal
manual_bin['revol_bal(0,5000)'] = np.where((manual_bin['revol_bal']>=0) 
                                                & (manual_bin['revol_bal']<=5000),1,0)
manual_bin['revol_bal(5000,10000)'] = np.where((manual_bin['revol_bal']>5000) 
                                                & (manual_bin['revol_bal']<=10000),1,0)
manual_bin['revol_bal(10000,15000)'] = np.where((manual_bin['revol_bal']>10000) 
                                                & (manual_bin['revol_bal']<=15000),1,0)
manual_bin['revol_bal(>15000)'] = np.where((manual_bin['revol_bal']>15000),1,0)

#revol_util
manual_bin['revol_util(0,20)'] = np.where((manual_bin['revol_util']>=0) 
                                                & (manual_bin['revol_util']<=20),1,0)
manual_bin['revol_util(20,40)'] = np.where((manual_bin['revol_util']>20) 
                                                & (manual_bin['revol_util']<=40),1,0)
manual_bin['revol_util(40,60)'] = np.where((manual_bin['revol_util']>40) 
                                                & (manual_bin['revol_util']<=60),1,0)
manual_bin['revol_util(60,80)'] = np.where((manual_bin['revol_util']>60) 
                                                & (manual_bin['revol_util']<=80),1,0)
manual_bin['revol_util(>80)'] = np.where((manual_bin['revol_util']>80),1,0)

#last_pymnt_amnt
manual_bin['last_pymnt_amnt(0,500)'] = np.where((manual_bin['last_pymnt_amnt']>=0) 
                                                & (manual_bin['last_pymnt_amnt']<=500),1,0)
manual_bin['last_pymnt_amnt(500,1000)'] = np.where((manual_bin['last_pymnt_amnt']>500) 
                                                & (manual_bin['last_pymnt_amnt']<=1000),1,0)
manual_bin['last_pymnt_amnt(1000,1500)'] = np.where((manual_bin['last_pymnt_amnt']>1000) 
                                                & (manual_bin['last_pymnt_amnt']<=1500),1,0)
manual_bin['last_pymnt_amnt(1500,3500)'] = np.where((manual_bin['last_pymnt_amnt']>1500) 
                                                & (manual_bin['last_pymnt_amnt']<=3500),1,0)
manual_bin['last_pymnt_amnt(>3500)'] = np.where((manual_bin['last_pymnt_amnt']>3500),1,0)

#pymnt_time
manual_bin['pymnt_time(0,1)'] = np.where((manual_bin['pymnt_time']>=0) 
                                                & (manual_bin['pymnt_time']<=1),1,0)
manual_bin['pymnt_time(1,6)'] = np.where((manual_bin['pymnt_time']>1) 
                                                & (manual_bin['pymnt_time']<=6),1,0)
manual_bin['pymnt_time(6,12)'] = np.where((manual_bin['pymnt_time']>6) 
                                                & (manual_bin['pymnt_time']<=12),1,0)
manual_bin['pymnt_time(>12)'] = np.where((manual_bin['pymnt_time']>12),1,0)

#annual_inc
manual_bin['annual_inc(low_income)'] = np.where((manual_bin['annual_inc']>=0) 
                                                & (manual_bin['annual_inc']<=50000),1,0)
manual_bin['annual_inc(mid_income)'] = np.where((manual_bin['annual_inc']>50000) 
                                                & (manual_bin['annual_inc']<=200000),1,0)
manual_bin['annual_inc(high_income)'] = np.where((manual_bin['annual_inc']>200000),1,0)

#drop original feature of manual_bin dataset
manual_bin_list = ['last_pymnt_amnt','revol_util','revol_bal','pymnt_time','term','target','annual_inc']
manual_bin = manual_bin.drop(manual_bin_list, axis =1)


# In[176]:


#display manual_bin
manual_bin.head(2)


# In[177]:


#create new dataframe
df_encoded = pd.concat([onehots, num_auto_bin, manual_bin, df_encodes['term'], 
                        df_encodes['initial_list_status'],df_encodes['target']], axis =1)


# In[178]:


df_encoded.head(2)


# In[179]:


#display row, column
df_encoded.shape


# ## Modelling

# ### Import Library for Modelling

# In[180]:


#split dataset
from sklearn.model_selection import train_test_split

#balance data train using SMOTE
from imblearn.over_sampling import SMOTE

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve, auc, confusion_matrix

#hyperparameter
from sklearn.model_selection import RandomizedSearchCV


# In[181]:


#import itertools
import itertools


# In[182]:


#confusion matrix function
def plot_confusion_matrix(c_matrix,
                          target_names,
                          title = 'Confusion Matrix',
                          cmap= None, 
                          normalize= True):
    
    #np.trace = sum of all the elements of a diagonal of given matrix
    accuracy = np.trace(c_matrix) / np.sum(c_matrix).astype('float')
    misclass = 1 - accuracy
    
    #get colormap instance
    if cmap is None:
        cmap = plt.get_cmap('Oranges')
        
    #image size
    plt.figure(figsize=(8,6))
    
    #display data as an image
    #data is resampled to the pixel size of the image on the figure canvas 
    #'nearest' interpolation is used if the number of display pixels is at least three times the size of the data array
    plt.imshow(c_matrix, interpolation='nearest', cmap= cmap)
    
    plt.title(title)
    
    #add a colorbar to a plot
    plt.colorbar()
    
    #set tick locations
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    
    #percentage with normalize
    #np.newaxis increase the dimensions of an array by adding new axes.
    if normalize:
        c_matrix = c_matrix.astype('float') / c_matrix.sum(axis =1)[:, np.newaxis]
    
    #threshold
    threshold = c_matrix.max()/1.5 if normalize else c_matrix.max()/2
    
    #itertools.product returns the cartesian product of the input iterables
    #The Cartesian product is the set of all combinations of elements from multiple sets
    for i,j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(c_matrix[i,j]),
                     horizontalalignment = 'center',
                     color = 'white' if c_matrix[i,j] > threshold else 'black')
        else:
            plt.text(j, i, "{:,}".format(c_matrix[i,j]),
                     horizontalalignment = 'center',
                     color = 'white' if c_matrix[i,j] > threshold else 'black')
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show() 


# In[183]:


#copy dataset
df_model = df_encoded.copy()


# In[184]:


#create data without column 'target'
x = df_model.drop(['target'], axis=1)

#create data only column 'target'
y = df_model['target']


# In[185]:


y.value_counts()


# In[186]:


#split dataset 70% training : 30% testing
#random state is a model hyperparameter used to control the randomness involved in machine learning models
#Whenever used Scikit-learn algorithm recommended to used (random_state=42) to produce the same results across a different run.
x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.3, random_state =42)

#display row and column for data train & data test
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# #### SMOTE

# **Notes:**
# - **Imbalanced Data Distribution**, generally happens when observations in one of the class are much higher or lower than the other classes
# - Standard ML techniques such as Decision Tree and Logistic Regression have a bias towards the majority class, and they tend to ignore the minority class.
# - If we have imbalanced data distribution in our dataset then our model becomes more prone to the case when minority class has negligible or very lesser recall.
# - **SMOTE (synthetic minority oversampling technique)** is one of the most commonly used oversampling methods to solve the imbalance problem.
# - SMOTE aims to balance class distribution by randomly increasing minority class examples by replicating them

# In[187]:


#handle imbalance target using SMOTE
sm = SMOTE(random_state =42)
sm.fit(x_train, y_train)
x_smote, y_smote = sm.fit_resample(x_train, y_train)

#display row and column for data smote & data test
x_smote.shape, x_test.shape, y_smote.shape, y_test.shape


# ### Train Model

# #### Binary Logistic Regression

# - Logistic regression is the appropriate regression analysis to conduct when the dependent variable is binary, means variable has only 2 outputs.
# - The reason we use logistic regression rather than linear regression because our dependent variable is binary and in linear regression, dependent variable is continuous.

# #### AUC (Area Under The Curve)

# In[188]:


#logistic regression
logreg = LogisticRegression(random_state = 42)

#fitting is equal to training
#It finds the coefficients for the equation specified via the algorithm being used
logreg.fit(x_smote, y_smote)

#after trained, the model can be used to make predictions
#predict_proba function will return the probabilities of a classification label
y_pred_proba_train = logreg.predict_proba(x_train)[:][:,1]
y_pred_proba_test = logreg.predict_proba(x_test)[:][:,1]

#AUC - ROC curve is a performance measurement for the classification problems at various threshold settings
print('AUC train probability: ', roc_auc_score(y_true= y_train , y_score= y_pred_proba_train))
print('AUC test probability: ', roc_auc_score(y_true= y_test , y_score= y_pred_proba_test))


# **AUC (Area Under The Curve) :**  
# - ROC is a probability curve and AUC represents the degree or measure of separability.
# - An excellent model has AUC near to the 1 which means it has a good measure of separability. 
# - A poor model has an AUC near 0 which means it has the worst measure of separability
# - When AUC is 0.5, it means the model has no class separation capacity whatsoever.
# 
# **Conclusion :**
# - From the result of AUC train & test probability which is 0.9 near to 1, so we've got excellent model

# #### Classification Report

# **Metrics that use to assess the quality of the model :**
# 1. Precision: Percentage of correct positive predictions relative to total positive predictions.
# 2. Recall: Percentage of correct positive predictions relative to total actual positives.
# 3. F1 Score: A weighted harmonic mean of precision and recall. The closer to 1, the better the model.
# 
# **F1 Score: 2 * (Precision * Recall) / (Precision + Recall)**
# 
# Using these three metrics, we can understand how well a given classification model is able to predict the outcomes for some response variable.
# 
# Fortunately, when fitting a classification model in Python we can use the **classification_report()** function from the sklearn library to generate all three of these metrics.

# In[189]:


#classification report
y_pred_class = []

for i in y_pred_proba_test:
    if i > 0.5:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

print(classification_report(y_true = y_test, y_pred= y_pred_class))


# **Precision**  
# - The precision tells us the accuracy of positive predictions.
# - Precision = true positive / (true positive + false positive)
# - **Conclusion: Out of all the loan status that the model predicted would get good loan, only 85% actually did.**
# 
# **Recall**
# - The recall, also named sensivity, or hit rate, tells us the fraction of correctly identified positive predictions.
# - recall = true positive / (true positive + false negative)
# - **Conclusion: Out of all the loan status that actually did get good loan, the model only predicted this outcome correctly for 65% of those loan status**
# 
# **F1 Score**  
# - The f1-score, or F measure, measures precision and recall at the same time by finding the harmonic mean of the two values.
# - This score is useful when you have opposite scores coming from precision and recall.
# - f1 score = 2 * ((recall * precision) / (recall + precision))
# - **Conclusion:   
# F1 Score = 0.73.   
# According to (spotintelligence.com, 2023) as a general rule of thumb, an F1 score of 0.7 or higher is often considered good.  
# So the model does a good job of predicting whether the loan status is considered good or bad.**

# In[190]:


from IPython.display import Image


# #### Confusion Matrix

# In[191]:


#Confusion Matrix is a performance measurement for machine learning classification problem 
#where output can be two or more classes. 
#It is a table with 4 different combinations of predicted and actual values.

Image(filename='image_IDX/confusion matrix.png', width= 300)


# In[192]:


#We describe predicted values as Positive and Negative and actual values as True and False.

Image(filename='image_IDX/actual and predict value.png', width= 300)


# In[193]:


#confusion matrix
c_matrix = confusion_matrix(y_true= y_test, y_pred= y_pred_class)
target_names = ['Good Loan','Bad Loan']

plot_confusion_matrix(c_matrix,
                          target_names,
                          title = 'Confusion Matrix',
                          cmap= None, 
                          normalize= False)


# **Insight :**
# - In the first column, the total samples in the **positive class** are 120,511 + 6,041 = 126,552
# - In the second column, the total samples in the **negative class**, are 1,954 + 11,026 = 12,980
# - The sum of the numbers in all the boxes gives the **total number of samples** evaluated 126,552 + 12,980 = 139,532
# - The **correct classifications** are the diagonal elements of the matrix 120,511 for the positive class and 11,026 for the negative class
# - 6,041 samples (bottom-left box) that were expected to be of the positive class were classified as the negative class by the model. So it is called **“False Negatives”** because the model predicted “negative,” which was wrong.
# - 1,954 samples (top-right box) were expected to be of negative class but were classified as “positive” by the model. They are thus called **“False Positives”**
# - **Accuracy rate**, which is the percentage of times a classifier is correct = 94.27%
# - **Misclassification rate**, which is the percentage of times a classifier is incorrect = 5.73%

# In[194]:


#check the total number of samples
y_test.shape


# #### Hyperparameter

# **Penalty**
# - The **"penalty"** parameter in a logistic regression model refers to a parameter used in some types of regularized logistic regression algorithms, such as **L1 (Lasso) and L2 (Ridge) regularization**.
# - This parameter determines the strength of the regularization term in the regression model, which helps to prevent overfitting.
# - In a logistic regression model, overfitting can occur when the model is too complex and fits the noise in the data.
# - This can result in poor generalization performance when the model is applied to new, unseen data.
# - The regularization term in a logistic regression model **adds a penalty** to the model's coefficients, which reduces their magnitude and helps **to prevent overfitting**
# - A higher value of the penalty parameter results in stronger regularization and a simpler model, while a lower value results in weaker regularization and a more complex model.
# - the penalty parameter in a logistic regression model is used to control the trade-off between model complexity and overfitting, and is a crucial parameter in achieving optimal model performance.
# 
# **Regularization :**
# - If all training points are correctly classified then we have problem of **overfitting (means doing perfect job on training set but performing very badly on test set, i.e. errors on train data is almost zero but errors on test data are very high).**
# - Regularization is a technique used to prevent overfitting problem.
# - The regression model which uses L1 regularization is called Lasso Regression and model which uses L2 is known as Ridge Regression.
# - If we use **L1 regularization** in Logistic Regression all the Less important features will become zero. If we use **L2 regularization** then the wi values will become small but not necessarily zero.
# - If hyper parameter(Λ) is 0 then there is no regularization term then it will overfit and if hyper parameter(Λ) is very large then it will add too much weight which leads to underfit.
# - If **L1-ratio = 0**, we have ridge regression. This means that we can treat our model as a ridge regression model
# - If **L1-ratio = 1**, we have lasso regression. Then we can solve it with the same ways we would use to solve lasso regression.
# - **Elastic net** is a combination of the regularized variants of linear regression: ridge and lasso.
# 
# **Parameter VS Hyperparameter**
# - People will often refer to all the arguments to a function as "parameters".
# - In machine learning, C is referred to as a "hyperparameter". 
# - The parameters are numbers that tells the model what to do with the features, while hyperparameters tell the model how to choose parameters.
# - Regularization generally refers the concept that there should be a complexity penalty for more extreme parameters. 
# - The idea is that just looking at the training data and not paying attention to how extreme one's parameters are leads to overfitting. 
# - A high value of C tells the model to give high weight to the training data, and a lower weight to the complexity penalty. 
# - A low value tells the model to give more weight to this complexity penalty at the expense of fitting to the training data.
# - Basically, a high C means "Trust this training data a lot", while a low value says "This data may not be fully representative of the real world data, so if it's telling you to make a parameter really large, don't listen to it".

# In[195]:


#penalty and C
param = {'penalty':['None', 'l2', 'l1', 'elasticnet'],
        'C': [float(x) for x in np.linspace(start=0, stop=1, num=75)]}

logreg = LogisticRegression()

#RandomizedSearchCV randomly passes the set of hyperparameters and 
#calculate the score and gives the best set of hyperparameters which gives the best score as an output.
logreg_clf = RandomizedSearchCV(estimator= logreg, 
                                param_distributions = param, 
                                scoring= 'roc_auc', 
                                cv=5, 
                                random_state =42)

search_logreg = logreg_clf.fit(x_smote, y_smote)

#best parameter
search_logreg.best_params_


# **Insight:**  
# - The “C” hyperparameter controls the strength of the regularization.
# - A smaller value for “C” (e.g. C=0.01) leads to stronger regularization and a simpler model, 
# - while a larger value (e.g. C=1.0) leads to weaker regularization and a more complex model.
# - Best parameter we've got is L2 (Ridge) regularization with 'C' is 0.027 which is near to 0, and leads to stronger regularization and a simpler model

# ### Retrain with Best Hyperparameter Tuning

# In[196]:


#best parameter
best_params = search_logreg.best_params_

#logistic regression
logreg_tuning = LogisticRegression(**best_params)

#fitting is equal to training
#It finds the coefficients for the equation specified via the algorithm being used
logreg_tuning.fit(x_smote, y_smote)

#after trained, the model can be used to make predictions
#predict_proba function will return the probabilities of a classification label
y_train_pred_proba = logreg_tuning.predict_proba(x_train)[:][:,1]
y_test_pred_lr_proba = logreg_tuning.predict_proba(x_test)[:][:,1]

#AUC - ROC curve is a performance measurement for the classification problems at various threshold settings
print('AUC train probability: ', roc_auc_score(y_true= y_train , y_score= y_train_pred_proba))
print('AUC test probability: ', roc_auc_score(y_true= y_test , y_score= y_test_pred_lr_proba))


# **Conclusion :**
# - From the result of AUC train & test probability which is 0.9 near to 1, so we've got excellent model

# In[197]:


#classification report
y_pred_class_2 = []

for i in y_test_pred_lr_proba:
    if i > 0.5:
        y_pred_class_2.append(1)
    else:
        y_pred_class_2.append(0)

print(classification_report(y_true = y_test, y_pred= y_pred_class_2))


# **Conclusion**
# - **Precision** :Out of all the loan status that the model predicted would get good loan, only 84% actually did.
# - **Recall**: Out of all the loan status that actually did get good loan, the model only predicted this outcome correctly for 64% of those loan status
# - **F1 Score** = 0.72 > 0.7. The model does a good job of predicting whether the loan status is considered good or bad

# In[198]:


#confusion matrix
c_matrix = confusion_matrix(y_true= y_test, y_pred= y_pred_class_2)
target_names = ['Good Loan','Bad Loan']

plot_confusion_matrix(c_matrix,
                          target_names,
                          title = 'Confusion Matrix',
                          cmap= None, 
                          normalize= False)


# **Insight :**
# - In the first column, the total samples in the **positive class** are 120,440 + 6,226 = 126,666
# - In the second column, the total samples in the **negative class**, are 2,025 + 10,841 = 12,866
# - The sum of the numbers in all the boxes gives the **total number of samples** evaluated 126,666 + 12,866 = 139,532
# - The **correct classifications** are the diagonal elements of the matrix 120,440 for the positive class and 10,841 for the negative class
# - 6,226 samples (bottom-left box) that were expected to be of the positive class were classified as the negative class by the model. So it is called **“False Negatives”** because the model predicted “negative,” which was wrong.
# - 2,025 samples (top-right box) were expected to be of negative class but were classified as “positive” by the model. They are thus called **“False Positives”**
# - **Accuracy rate**, which is the percentage of times a classifier is correct = 94.09%
# - **Misclassification rate**, which is the percentage of times a classifier is incorrect = 5.91%

# ### Show Coefficient Value Each Feature with Statsmodel Logistic Regression

# In[199]:


import statsmodels.api as sm

#before we fit the model, we need to use the sm.add_constant(X) function, 
#which adds a column of constants to the X dataframe, before passing that into the Logit() function. 
#This is a structural requirement so that the Logit() estimation can be performed properly.
x2 = sm.add_constant(x_smote)

#Statsmodels provides a Logit() function for performing logistic regression.
est = sm.Logit(endog= y_smote ,exog= x2)

#model that fits the data well if the differences between the observed values and the model's predicted values are small and unbiased
#model fitting is a measure of how well a machine learning model generalizes to similar data to that on which it was trained. 
#model that is well-fitted produces more accurate outcomes. 
#model that is overfitted matches the data too closely. 
#model that is underfitted doesn’t match closely enough.
#sklearn fit method uses the training data as an input to train the machine learning model.
#The ‘fit’ method trains the algorithm on the training data, after the model is initialized.
#Broyden, Fletcher, Goldfarb, and Shanno, or BFGS Algorithm algorithms for numerical optimization and 
#BFGS is commonly used to fit machine learning algorithms such as the logistic regression algorithm.
est2 = est.fit(method= 'bfgs')
print(est2.summary())


# **Insight :**  
# - Logistic Regression, like binary and multinomial logistic regression, uses maximum likelihood estimation, which is an iterative procedure, and we had 35 iteration.
# - Variabel dependen (terikat / Y) adalah variabel yang dipengaruhi atau yang menjadi akibat karena adanya variabel bebas (variabel independent / bebas / X).
# - dependent variable = target(loan_status)
# - independent variable = home_ownership, purpose, emp_length, grade, loan_amnt, dti, int_rate, open_acc, total_acc, total_pymnt, credit_pull_year, revol_bal, revol_util, last_pymnt_amnt, pymnt_time, annual_inc, term, initial_list_status
# 
# **Coef**  
# - These are the values for the logistic regression equation for predicting the dependent variable from the independent variable. They are in log-odds units. 
# 
# - The **prediction equation** is:  
# log(p/1-p) = b0 + (b1 * home_ownership_MORTGAGE) + (b2 * home_ownership_OTHER) + ..... + (bn * initial_list_status) 
# 
# - Where **p** is the probability. Expressed in terms of the variables used in this example, **the logistic regression equation** is :  
# log(p/1-p) = (-0.2864 * home_ownership_MORTGAGE) + (0.0128 * home_ownership_OTHER) + ..... + (12.0088 * initial_list_status)
# 
# - Because these coefficients are in log-odds units, they are often **difficult to interpret**, so they are often converted into **odds ratios**
# 
# 
# **These estimates tell the amount of increase in the predicted log odds of loan status = 1 that would be predicted by a 1 unit increase in the predictor, holding all other predictors constant.**   
# 
# For example :  
# home_ownership_MORTGAGE – The coefficient (or parameter estimate) for the variable home_ownership_MORTGAGE is 0.0128.  This means that for a one-unit increase in home_ownership_MORTGAGE, we expect a 0.0128 increase in the log-odds of the dependent variable loan_status, holding all other independent variables constant.
# 

# In[200]:


#converting statsmodel summary object to Pandas Dataframe
df_importance = pd.read_html(est2.summary().tables[1].as_html(),
                             header = 0, 
                             index_col =0)[0]
#find odds ratio
for i in df_importance['coef']:
    if i == 0:
        df_importance['odds_ratio']==0
    else:
        df_importance['odds_ratio'] = np.exp(df_importance['coef'])

#show probability contribution
#filtering only ['P>|z|'] <= 0.05
#['P>|z|'] provide the z-value and 2-tailed p-value used in testing the null hypothesis that the coefficient (parameter) is 0.
#alpha is 0.05, coefficients having a p-value of 0.05 or less would be statistically significant 
#i.e, we can reject the null hypothesis and say that the coefficient is significantly different from 0
df_importance[df_importance['P>|z|'] <= 0.05].sort_values('odds_ratio', ascending= False)


# **Odds ratio in Logistic Regression**  
# - Probabilities range between 0 and 1
# - for mid_income, the odds of being good loans are 4.23 times as large as the odds for low and high income being good loans.

# In[201]:


#if we display ['P>|z|'] > 0.05, we can see that coef is equal to 0, and accept null hypothesis
df_importance[df_importance['P>|z|'] > 0.05].sort_values('odds_ratio', ascending= False)


# ### ROC Curve

# In[202]:


#false & true positive rate
fpr, tpr, tr = roc_curve(y_true= y_test, y_score= y_test_pred_lr_proba)

#roc auc score
auc = roc_auc_score(y_true= y_test, y_score= y_test_pred_lr_proba)

#plot ROC Curve
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr, label= f'AUC = {auc:.3f}', color='orange')
plt.plot(fpr,fpr, linestyle = '--', color='grey')
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.title('ROC Curve', fontsize=15)
plt.legend()
plt.tight_layout()


# **Insight :**  
# - The ROC curve shows the performance of a binary classifier, which plots the True Positive rate (TPR) against the False Positive rate (FPR).
# - The ROC AUC score is the area under the ROC curve. It sums up how well a model can produce relative scores to discriminate between positive or negative instances across all classification thresholds. 
# - The ROC AUC score ranges from 0 to 1, where 0.5 indicates random guessing, and 1 indicates perfect performance.
# - From the plot, we've got AUC score = 0.93, which is near to 1, indicates good performance.

# ### Kolmogorov-Smirnov

# In[203]:


import scikitplot as skplt

#logistic regression with best parameter
#predict_proba = probability estimates
y_pred_proba = logreg_tuning.predict_proba(x_test)

#plot kolmogorov smirnov
plt.figure(figsize=(7,5))
skplt.metrics.plot_ks_statistic(y_test, y_pred_proba)


# **Insight:**  
# - Even if ROC AUC is the most widespread metric for class separation, but it is also good to plot Kolmogorov Smirvov to know how well the model separates the predictions of the two different classes
# - the ROC AUC score goes from 0.5 to 1.0, while KS statistics range from 0.0 to 1.0.
# - K-S should be a high value (Max =1.0) when the fit is good and a low value (Min = 0.0) when the fit is not good.
# - From the plot, we've got KS Statistic = 0.736, so we considered it as 'medium' dataset, which mean even though it doesn't have perfect separation, but there is enough overlap to confuse the classifier, and has wide gap between the class CDF (positive & negative instances).

# ### Score Card

# In[204]:


#reset index data statsmodel summary object
df_importance = df_importance.reset_index()


# In[205]:


df_importance.head()


# In[206]:


#rename column 
df_importance = df_importance.rename(columns = {'index':'feature'})


# In[207]:


df_importance.head(2)


# In[208]:


#create new column 
df_importance['feature_name'] = df_importance['feature'].str.split('_').str[:-1]
df_importance['feature_name'] = df_importance['feature_name'].str.join('_')


# In[209]:


df_importance


# In[210]:


#fix wrong feature_name
df_importance.at[124,'feature_name'] = 'term'
df_importance.at[125,'feature_name'] = 'initial_list_status'


# In[211]:


df_importance


# In[212]:


#copy dataset
df_scorecard = df_importance.copy()


# In[213]:


#define min and max score according to FICO
min_score = 300
max_score = 850


# In[214]:


#group by sum of minimum 
min_sum_coef = df_scorecard.groupby('feature_name')['coef'].min().sum()

#group by sum of maximum 
max_sum_coef = df_scorecard.groupby('feature_name')['coef'].max().sum()

#calculate credit score
df_scorecard['score_calculation'] = ((df_scorecard['coef']) * (max_score - min_score) / (max_sum_coef - min_sum_coef)).round(3)


# In[215]:


df_scorecard


# In[216]:


#fix wrong row
df_scorecard.at[121,'feature_name'] = 'annual_inc'
df_scorecard.at[122,'feature_name'] = 'annual_inc'
df_scorecard.at[123,'feature_name'] = 'annual_inc'


# In[217]:


df_scorecard


# In[218]:


#group by sum of minimum 
min_sum_coef = df_scorecard.groupby('feature_name')['coef'].min().sum()

#group by sum of maximum 
max_sum_coef = df_scorecard.groupby('feature_name')['coef'].max().sum()

#calculate credit score
df_scorecard['score_calculation'] = ((df_scorecard['coef']) * (max_score - min_score) / (max_sum_coef - min_sum_coef)).round(3)


# In[219]:


min_sum_score_pre1 = df_scorecard.groupby('feature_name')['score_calculation'].min().sum()
max_sum_score_pre1 = df_scorecard.groupby('feature_name')['score_calculation'].max().sum()


# In[220]:


print(min_sum_score_pre1)
print(max_sum_score_pre1)


# In[221]:


min_sum_score_pre2 = df_scorecard.groupby('feature_name').agg({'score_calculation':['min','max']})


# In[222]:


min_sum_score_pre2 = min_sum_score_pre2.reset_index()


# In[223]:


min_sum_score_pre2


# In[224]:


df_scorecard


# In[225]:


#fix wrong row
df_scorecard.at[5,'feature_name'] = 'purpose'
df_scorecard.at[6,'feature_name'] = 'purpose'
df_scorecard.at[8,'feature_name'] = 'purpose'
df_scorecard.at[10,'feature_name'] = 'purpose'
df_scorecard.at[14,'feature_name'] = 'purpose'
df_scorecard.at[15,'feature_name'] = 'purpose'


# In[226]:


#group by sum of minimum 
min_sum_coef = df_scorecard.groupby('feature_name')['coef'].min().sum()

#group by sum of maximum 
max_sum_coef = df_scorecard.groupby('feature_name')['coef'].max().sum()

#calculate credit score
df_scorecard['score_calculation'] = ((df_scorecard['coef']) * (max_score - min_score) / (max_sum_coef - min_sum_coef)).round(3)


# In[227]:


min_sum_score_pre1 = df_scorecard.groupby('feature_name')['score_calculation'].min().sum()
max_sum_score_pre1 = df_scorecard.groupby('feature_name')['score_calculation'].max().sum()
print(min_sum_score_pre1)
print(max_sum_score_pre1)


# In[228]:


df_scorecard.groupby('feature_name').agg({'score_calculation':['min','max']}).reset_index()


# In[229]:


min_sum_coef


# In[230]:


max_sum_coef


# In[231]:


df_scorecard2 = df_scorecard.copy()


# In[232]:


min_sum_coef = df_scorecard2.groupby('feature_name')['coef'].min().sum()

#group by sum of maximum 
max_sum_coef = df_scorecard2.groupby('feature_name')['coef'].max().sum()

#calculate credit score
df_scorecard2['score_calculation'] = ((df_scorecard2['coef']) * (max_score - min_score) / (max_sum_coef - min_sum_coef)).round(3)


# In[233]:


min_sum_score_pre2 = df_scorecard2.groupby('feature_name')['score_calculation'].min().sum()
max_sum_score_pre2 = df_scorecard2.groupby('feature_name')['score_calculation'].max().sum()
print(min_sum_score_pre2)
print(max_sum_score_pre2)


# In[234]:


num_min = min_score - min_sum_score_pre2
num_max = max_score - max_sum_score_pre2
print(num_min)
print(num_max)


# In[235]:


#add column intercept
df_scorecard2.loc[len(df_scorecard2.index)] = ['intercept',0,0,0,0,0,0,0,'intercept',num_min]


# In[236]:


df_scorecard2


# In[237]:


#check min and max score
min_sum_score_pre2 = df_scorecard2.groupby('feature_name')['score_calculation'].min().sum().round()
max_sum_score_pre2 = df_scorecard2.groupby('feature_name')['score_calculation'].max().sum().round()
print(min_sum_score_pre2)
print(max_sum_score_pre2)


# In[238]:


#exclude intercept
df_high_score = df_scorecard2[:126]

#sort value to top 5
high_score = df_high_score.sort_values('score_calculation',ascending= False).round(0).head(5)

#create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(15,8))

#plot bar chart
sns.barplot(data =high_score, x='score_calculation', y='feature', palette='flare')

#label
plt.bar_label(ax.containers[0])
plt.xlabel('Score')
plt.title('Top 5 Highest Score Features', fontsize=15, weight ='extra bold')


# **Insight:**  
# Features that make high contribution to increase credit score are:
# - initial list status
# - last payment amount
# - total payment
# 
# The result is in line with how FICO credit score is calculated that based on five factors: payment history, amount owed, new credit, length of credit history, and credit mix.

# In[239]:


#sort value to the 5 lowest
low_score = df_high_score.sort_values('score_calculation',ascending= True).round(0).head(5)

#create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(15,8))

#plot bar chart
sns.barplot(data =low_score, x='score_calculation', y='feature', palette='flare')

#label
plt.bar_label(ax.containers[0])
plt.xlabel('Score')
plt.title('Top 5 Lowest Score Features', fontsize=15, weight ='extra bold')


# **Insight:**  
# Features that make contribution to decrease credit score are:
# - last payment amount
# - loan amount
# - payment time
# 
# The result is in line with how FICO credit score is calculated that based on five factors: payment history, amount owed, new credit, length of credit history, and credit mix.

# In[240]:


x_smote.columns


# In[241]:


#new dataset without target(loan_status)
data_fico = df_model[x_smote.columns]

#copy dataset
df_score = data_fico.copy()

#display
df_score.head(2)


# In[242]:


df_score.shape


# In[243]:


#insert intercept column
df_score.insert(loc= 0 ,column= 'intercept', value=1)


# In[244]:


#dataframe score
scorecard_score = df_scorecard2['score_calculation']


# In[245]:


scorecard_score.shape


# In[246]:


#reshape to row,column
scorecard_score = scorecard_score.values.reshape(127,1)


# In[247]:


scorecard_score


# In[248]:


#calculate score with matrix multiplication
#dot product of arrays with numpy
y_scores = df_score.dot(scorecard_score)


# In[249]:


#concatenate pandas
score_card_df = pd.concat([df_score, y_scores], axis=1)


# In[250]:


score_card_df.head(1)


# In[251]:


#rename column
score_card_df.rename(columns = {0:'credit_score'}, inplace = True)


# In[252]:


score_card_df.head(1)


# In[253]:


df.columns


# In[254]:


#copy column id and member id
df_ori = pd.read_csv(r'D:\Dokumen\Portfolio Project\Credit Risk IDX Partners/loan_data_2007_2014.csv')
df_id = df_ori[['id','member_id']].copy()


# In[255]:


#merge credit_score with member id
credit_score_with_id = pd.merge(df_id, score_card_df, left_index=True, right_index=True)


# In[256]:


credit_score_with_id.head(2)


# In[257]:


#create dataframe with selected columns
result_credit_score = credit_score_with_id[['id','member_id', 'credit_score']]


# In[258]:


#take sample
result_credit_score.sample(10)


# In[259]:


#create dataframe with selected columns
credit_score_with_id_2 = credit_score_with_id[['id', 'credit_score']]


# In[260]:


#merge raw dataframe to clean data
df_result = pd.merge(df_ori, df_clean[['id','pymnt_time','credit_pull_year']], on='id')


# In[261]:


#merge score to raw dataframe
df_result = pd.merge(df_result, credit_score_with_id_2, on='id')


# In[262]:


#take sample
df_result.sample(10)


# ## Visualization Based On Feature Importance Model

# In[263]:


df_vis = df_result.copy()


# ### FICO Score
# A FICO score is a credit score created by the Fair Isaac Corporation (FICO). Lenders use borrowers’ FICO scores along with other details on borrowers’ credit reports to assess credit risk and determine whether to extend credit.
# 
# FICO scores take into account data in five areas to determine a borrower's credit worthiness: payment history, the current level of indebtedness, types of credit used, length of credit history, and new credit accounts

# In[264]:


Image(filename='image_IDX/fico score.jpg', width= 500)


# In[265]:


df_vis.head(1)


# In[266]:


#create colum score_group
df_vis['score_group'] = np.where((df_vis['credit_score'] >=300) & (df_vis['credit_score'] <=579), 'Poor (300-579)',
                                np.where((df_vis['credit_score'] >=580) & (df_vis['credit_score'] <=669), 'Fair (580-669)',
                                np.where((df_vis['credit_score'] >=670) & (df_vis['credit_score'] <=739), 'Good (670-739)',
                                np.where((df_vis['credit_score'] >=740) & (df_vis['credit_score'] <=799), 'Very Good (740-799)', 'Excellent (800-850)'))))


# In[267]:


df_vis.head(1)


# In[268]:


#group by score_group with id customer
score_groupy = df_vis.groupby('score_group').agg(num_cust=('id','count')).reset_index()


# In[269]:


score_groupy


# In[270]:


total_num_cust = score_groupy['num_cust'].sum()
total_num_cust


# In[271]:


score_groupy['percentage'] = round((score_groupy['num_cust']/total_num_cust)*100, 2)


# In[272]:


score_groupy = score_groupy.sort_values('percentage', ascending = False)
score_groupy.reset_index(drop = True)


# In[273]:


#create bar plot
fig, ax = plt.subplots(figsize = (9, 7))
sns.barplot(y= score_groupy['score_group'], x=score_groupy['percentage'], orient='h', palette='flare')
plt.bar_label(ax.containers[0], fmt='%.1f%%')
plt.title('Credit Score Category',fontsize=15, weight ='extra bold')
plt.show()


# **Insight:**  
# - the highest number of borrowers is in the loan poor category at 73.1%.
# - Then followed by Fair at 17.6%, Good at 8.1%, Very good at 0.8%, and excelent at 0.4%
# - Loan poor category demonstrates to lenders that their customer is a risky borrower

# In[274]:


Image(filename='image_IDX/fico table.png', width= 500)


# ## Credit Score with Loan Status

# ### Adjusting Label to Visualization

# In[275]:


#copy dataset 
dv_vis_2 = df_vis.copy()


# In[276]:


#define good loan
good_loan = ['Current','Fully Paid']


# In[277]:


dv_vis_2.head(2)


# In[278]:


#adjusting label on loan_status
dv_vis_2['loan_status'] = dv_vis_2['loan_status'].apply(lambda x: 'Good Loan' if x in good_loan else 'Bad Loan')


# In[279]:


#selected columns
dv_vis_2[['loan_amnt', 'last_pymnt_amnt', 'pymnt_time', 'int_rate']].describe(percentiles=[0.1,0.25,0.5,0.75,0.9,0.95,0.99])


# In[280]:


#adjusting label on loan_amnt
dv_vis_2['loan_amnt_fc'] = np.where((dv_vis_2['loan_amnt']>= 465)  & (dv_vis_2['loan_amnt']< 3950),'465-3950',
                                np.where((dv_vis_2['loan_amnt']>= 3950)  & (dv_vis_2['loan_amnt']< 7400),'3950-7400',
                                np.where((dv_vis_2['loan_amnt']>= 7400)  & (dv_vis_2['loan_amnt']< 10850),'7400-10850',
                                np.where((dv_vis_2['loan_amnt']>= 10850)  & (dv_vis_2['loan_amnt']< 14300),'10850-14300',
                                np.where((dv_vis_2['loan_amnt']>= 14300)  & (dv_vis_2['loan_amnt']< 17750),'14300-17750',
                                np.where((dv_vis_2['loan_amnt']>= 17750)  & (dv_vis_2['loan_amnt']< 21200),'17750-21200',
                                np.where((dv_vis_2['loan_amnt']>= 21200)  & (dv_vis_2['loan_amnt']< 24650),'21200-24650',
                                np.where((dv_vis_2['loan_amnt']>= 24650)  & (dv_vis_2['loan_amnt']< 28100),'24650-28100',
                                np.where((dv_vis_2['loan_amnt']>= 28100)  & (dv_vis_2['loan_amnt']< 31550),'28100-31550', '31550-35000')))))))))


# In[281]:


#check
dv_vis_2[['loan_amnt','loan_amnt_fc']].head(3)


# In[282]:


#adjusting label on last_pymnt_amnt
dv_vis_2['last_pymnt_amnt_fc'] = np.where((dv_vis_2['last_pymnt_amnt']>= 0)  & (dv_vis_2['last_pymnt_amnt']< 500),'0-500',
                                    np.where((dv_vis_2['last_pymnt_amnt']>= 500)  & (dv_vis_2['last_pymnt_amnt']< 1000),'500-1000',
                                    np.where((dv_vis_2['last_pymnt_amnt']>= 1000)  & (dv_vis_2['last_pymnt_amnt']< 1500),'1000-1500',
                                    np.where((dv_vis_2['last_pymnt_amnt']>= 1500)  & (dv_vis_2['last_pymnt_amnt']< 3500),'1500-3500', '>3500'))))


# In[283]:


#check
dv_vis_2[['last_pymnt_amnt','last_pymnt_amnt_fc']].head(3)


# In[284]:


#adjusting label on pymnt_time
dv_vis_2['pymnt_time_fc'] = np.where((dv_vis_2['pymnt_time']>= 0)  & (dv_vis_2['pymnt_time']<= 1),'1 Month',
                                np.where((dv_vis_2['pymnt_time']>= 2)  & (dv_vis_2['pymnt_time']<= 6),'2-6 Month',
                                np.where((dv_vis_2['pymnt_time']>= 7)  & (dv_vis_2['pymnt_time']<= 12),'7-12 Month', 'Over 1 year')))


# In[285]:


#check
dv_vis_2[['pymnt_time','pymnt_time_fc']].head(5)


# In[286]:


dv_vis_2['int_rate'].min()


# In[287]:


dv_vis_2['int_rate'].max()


# In[288]:


#adjusting label on int_rate
dv_vis_2['int_rate_fc'] = np.where((dv_vis_2['int_rate']>= 5.3)  & (dv_vis_2['int_rate']< 7.5),'5.3-7.5',
                                np.where((dv_vis_2['int_rate']>= 7.5)  & (dv_vis_2['int_rate']< 9.5),'7.5-9.5',
                                np.where((dv_vis_2['int_rate']>= 9.5)  & (dv_vis_2['int_rate']< 11.5),'9.5-11.5',
                                np.where((dv_vis_2['int_rate']>= 11.5)  & (dv_vis_2['int_rate']< 13.5),'11.5-13.5',
                                np.where((dv_vis_2['int_rate']>= 13.5)  & (dv_vis_2['int_rate']< 15.5),'13.5-15.5',
                                np.where((dv_vis_2['int_rate']>= 15.5)  & (dv_vis_2['int_rate']< 17.5),'15.5-17.5',
                                np.where((dv_vis_2['int_rate']>= 17.5)  & (dv_vis_2['int_rate']< 19.5),'17.5-19.5',
                                np.where((dv_vis_2['int_rate']>= 19.5)  & (dv_vis_2['int_rate']< 21.5),'19.5-21.5',
                                np.where((dv_vis_2['int_rate']>= 21.5)  & (dv_vis_2['int_rate']< 23.5),'21.5-23.5','24-26')))))))))


# In[289]:


#check
dv_vis_2[['int_rate','int_rate_fc']].head(3)


# ### Create Orderlist

# In[290]:


dv_vis_2['emp_length'].value_counts()


# In[291]:


order_list_loan_amnt = ['465-3950','3950-7400','7400-10850','10850-14300','14300-17750',
                   '17750-21200','21200-24650','24650-28100','28100-31550','31550-35000']
order_list_last_pymnt_amnt = ['0-500','500-1000','1000-1500','1500-3500','>3500']
order_list_pymnt_time = ['1 Month','2-6 Month','7-12 Month','Over 1 year']
order_list_int_rate = ['5.3-7.5','7.5-9.5','9.5-11.5','11.5-13.5','13.5-15.5',
                       '15.5-17.5','17.5-19.5','19.5-21.5','21.5-23.5','24-26']
order_list_emp = ['< 1 year','1 year','2 years','3 years','4 years','5 years',
                  '6 years','7 years','8 years','9 years','10+ years']


# ### Loan Status Visualization

# score_groupy = df_vis.groupby('score_group').agg(num_cust=('id','count')).reset_index()
# total_num_cust = score_groupy['num_cust'].sum()
# total_num_cust
# score_groupy['percentage'] = round((score_groupy['num_cust']/total_num_cust)*100, 2)
# #create bar plot
# fig, ax = plt.subplots(figsize = (9, 7))
# sns.barplot(y= score_groupy['score_group'], x=score_groupy['percentage'], orient='h', palette='flare')
# plt.bar_label(ax.containers[0], fmt='%.1f%%')
# plt.title('Credit Score Category',fontsize=15, weight ='extra bold')
# plt.show()

# In[296]:


#groupby loan_status
loan_status_group = dv_vis_2.groupby('loan_status').agg(num_cust=('id','count')).reset_index()
total_num_cust_loan_status = loan_status_group['num_cust'].sum()
loan_status_group['percentage'] = round((loan_status_group['num_cust']/total_num_cust_loan_status)*100, 2)
loan_status_group = loan_status_group.sort_values('percentage', ascending = False)
loan_status_group


# In[297]:


#plot loan_status
fig, ax = plt.subplots(figsize = (10, 7))
sns.barplot(y= loan_status_group['loan_status'], x=loan_status_group['percentage'], orient='h', palette='flare')
plt.bar_label(ax.containers[0], fmt='%.1f%%')
plt.title('Loan Status Category',fontsize=15, weight ='extra bold')
plt.show()


# In[298]:


#automatization function
def d_frame(group):
    df_group = dv_vis_2.groupby(group).agg(num_cust=('id','count')).reset_index()
    total_num_cust = df_group['num_cust'].sum()
    df_group['percentage'] = round((df_group['num_cust']/total_num_cust)*100, 2)
    df_group = df_group.sort_values('percentage', ascending = False)
    return df_group


# In[299]:


#groupby loan_status
d_frame('loan_status')


# In[300]:


def plot(group, title):
    df_group = dv_vis_2.groupby(group).agg(num_cust=('id','count')).reset_index()
    total_num_cust = df_group['num_cust'].sum()
    df_group['percentage'] = round((df_group['num_cust']/total_num_cust)*100, 2)
    df_group = df_group.sort_values('percentage', ascending = False)
    
    fig, ax = plt.subplots(figsize = (10, 7))
    sns.barplot(y= df_group[group], x=df_group['percentage'], orient='h', palette='flare')
    plt.bar_label(ax.containers[0], fmt='%.1f%%')
    plt.title(title,fontsize=15, weight ='extra bold')
    plt.show()


# In[301]:


plot('loan_status', 'Loan Status Category')


# ### Bad Loan Rate Based On Loan Amount

# In[302]:


dv_vis_2.head(1)


# In[303]:


#groupby df_loan
df_loan = dv_vis_2.groupby(['score_group','loan_status','loan_amnt_fc']).agg(num_cust= ('id','count')).reset_index()
df_loan.head(3)


# In[304]:


total_cust_loan = df_loan.groupby(['loan_amnt_fc']).agg(total_cust= ('num_cust','sum')).reset_index()
total_cust_loan


# In[305]:


#merge df_loan with total_cust_loan
df_loan_group = df_loan.merge(total_cust_loan, on='loan_amnt_fc', how='inner')
df_loan_group.head()


# In[306]:


#select loan_status bad loan
df_bad_loan_rate = df_loan_group[df_loan_group['loan_status'] == 'Bad Loan']
df_bad_loan_rate.head()


# In[307]:


#percentage
df_bad_loan_rate['bad_loan_rate'] = round((df_bad_loan_rate['num_cust'] / df_bad_loan_rate['total_cust'])*100, 2)
df_bad_loan_rate = df_bad_loan_rate.sort_values('bad_loan_rate', ascending=False).reset_index()
df_bad_loan_rate.head()


# In[308]:


#import module
from matplotlib import style


# In[322]:


plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (20, 12))

#plot data
sns.barplot(data = df_bad_loan_rate, x='loan_amnt_fc', 
            y='bad_loan_rate', hue='score_group', 
            orient='v', palette='OrRd', ci=None, order= order_list_loan_amnt)

#axvline
plt.axvline(x= -0.55, ls='--', color='red')
plt.axvline(x= 2.5, ls='--', color='red')
plt.stackplot(np.arange(-0.5,3.5),[[25000]], color='red', alpha=0.1)

#text
plt.text(x=0, y=13, s='This loan is too risky', 
         fontsize=16, color='red',va='center',weight='bold')

#set y axes
plt.ylim(0,15)

#title
plt.title("Bad Loan Rate on Loan Amount \nBased on Borrower's Score Status",
          fontsize=18, weight ='extra bold')

#show percentage
plt.bar_label(ax.containers[0], fmt='%.1f%%')
plt.bar_label(ax.containers[1], fmt='%.1f%%')
plt.bar_label(ax.containers[2], fmt='%.1f%%')
plt.bar_label(ax.containers[3], fmt='%.1f%%')

#label axis
plt.xlabel('Loan Amount', fontsize=14)
plt.ylabel('Bad Loan Rate (%)', fontsize=14)

#legend
plt.legend(title='Status', loc='upper right')
plt.show()


# **Insight :**  
# - Customers who have a bad credit score with a loan amount ranging from 465-10,850 have the potential to become a bad loan in the future

# ### Bad Loan Rate Based On Interest Rate

# In[310]:


def df_bad_loan(df_fc):
    #selected column
    df_int = dv_vis_2.groupby(['score_group','loan_status',df_fc]).agg(num_cust= ('id','count')).reset_index()
    
    #total customer
    total_cust_int = df_int.groupby([df_fc]).agg(total_cust= ('num_cust','sum')).reset_index()
    
    #merge
    df_int_group = df_int.merge(total_cust_int, on=df_fc, how='inner')
    
    #select bad loan
    df_bad_loan_rate = df_int_group[df_int_group['loan_status'] == 'Bad Loan']
    
    #bad loan rate
    df_bad_loan_rate['bad_loan_rate'] = round((df_bad_loan_rate['num_cust'] / df_bad_loan_rate['total_cust'])*100, 2)
    
    #sort
    df_bad_loan_rate = df_bad_loan_rate.sort_values('bad_loan_rate', ascending=False).reset_index()
    
    return df_bad_loan_rate


# In[311]:


#bad loan rate based on int_rate_fc
df_int_rate = df_bad_loan('int_rate_fc')
df_int_rate


# In[323]:


def plot_bad_loan(data_df, x, y, hue, order, x_line1, x_line2, r1, r2, x_text, y_text, text, y_ax, title, xlabel):
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize = (20, 12))
    
    #plot data
    sns.barplot(data = data_df, x=x, 
                y= y, hue= hue, 
                orient='v', palette='OrRd', ci=None, order=order)
    
    #axvline
    plt.axvline(x= x_line1, ls='--', color='red')
    plt.axvline(x= x_line2, ls='--', color='red')
    plt.stackplot(np.arange(r1,r2),[[25000]], color='red', alpha=0.1)

    #text
    plt.text(x=x_text, y=y_text, s=text, 
             fontsize=16, color='red',va='center',weight='bold')

    #set y axes
    plt.ylim(0,y_ax)
    
    #title
    plt.title(title,
              fontsize=18, weight ='extra bold')

    #show percentage
    plt.bar_label(ax.containers[0], fmt='%.1f%%')
    plt.bar_label(ax.containers[1], fmt='%.1f%%')
    plt.bar_label(ax.containers[2], fmt='%.1f%%')
    plt.bar_label(ax.containers[3], fmt='%.1f%%')

    #label axis
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Bad Loan Rate (%)', fontsize=14)

    #legend
    plt.legend(title='Status', loc='upper right')
    plt.show()


# In[328]:


plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (20, 12))

#plot data
sns.barplot(data = df_int_rate, x='int_rate_fc', 
            y='bad_loan_rate', hue='score_group', 
            orient='v', palette='OrRd', ci=None, order= order_list_int_rate)

#axvline
plt.axvline(x= 1.5, ls='--', color='red')
plt.axvline(x= 6.5, ls='--', color='red')
#plt.stackplot(np.arange(1,8),[[25000]], color='red', alpha=0.1)

#text
plt.text(x=3, y=13, s='This loan is too risky', 
         fontsize=16, color='red',va='center',weight='bold')

#set y axes
plt.ylim(0,15)

#title
plt.title("Bad Loan Rate on Interest Rate \nBased on Borrower's Score Status",
          fontsize=18, weight ='extra bold')

#show percentage
plt.bar_label(ax.containers[0], fmt='%.1f%%')
plt.bar_label(ax.containers[1], fmt='%.1f%%')
plt.bar_label(ax.containers[2], fmt='%.1f%%')
plt.bar_label(ax.containers[3], fmt='%.1f%%')

#label axis
plt.xlabel('Interest Rate', fontsize=14)
plt.ylabel('Bad Loan Rate (%)', fontsize=14)

#legend
plt.legend(title='Status', loc='upper right')
plt.show()


# **Insight:**  
# - Customers who have a bad credit score with interest rate ranging from 9.5% to 19.5% have the potential to become a bad loan in the future

# ### Bad Loan Rate Based On Payment Time

# In[330]:


#bad loan rate based on payment time
df_pymnt_time = df_bad_loan('pymnt_time_fc')
df_pymnt_time


# In[338]:


plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(figsize = (20, 12))

#plot data
sns.barplot(data = df_pymnt_time, x='pymnt_time_fc', 
            y='bad_loan_rate', hue='score_group', 
            orient='v', palette='OrRd', ci=None, order= order_list_pymnt_time)

#axvline
plt.axvline(x= 0.6, ls='--', color='red')
plt.axvline(x= 3.4, ls='--', color='red')
#plt.stackplot(np.arange(0.6,4),[[25000]], color='red', alpha=0.1)

#text
plt.text(x=1, y=13, s='This loan is too risky', 
         fontsize=16, color='red',va='center',weight='bold')

#set y axes
plt.ylim(0,15)

#title
plt.title("Bad Loan Rate on Payment Time \nBased on Borrower's Score Status",
          fontsize=18, weight ='extra bold')

#show percentage
plt.bar_label(ax.containers[0], fmt='%.1f%%')
plt.bar_label(ax.containers[1], fmt='%.1f%%')
plt.bar_label(ax.containers[2], fmt='%.1f%%')
plt.bar_label(ax.containers[3], fmt='%.1f%%')

#label axis
plt.xlabel('Payment Time', fontsize=14)
plt.ylabel('Bad Loan Rate (%)', fontsize=14)

#legend
plt.legend(title='Status', loc='upper left')
plt.show()


# **Insight:**  
# - The longer the customer's payment time takes to pay off the loan, the riskier customer's gonna potentially to be bad loan

# ### Recommendation

# - Loan companies can build a robust and effective credit scoring model machine learning using variety of methods and criteria to assess the creditworthiness of potential customers. The goal is to minimize the risk of lending to individuals who are unlikely to repay their loans.
# - One of method to evaluate a borrower incorporates both qualitative and quantitative measures is the 5 C's of credit (Character, Capacity, Capital, Collateral, and Conditions)

# In[340]:


Image(filename='image_IDX/5c.png', width= 600)

