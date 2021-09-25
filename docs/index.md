# Bike Share Rides Data Analysis

Analysis of a case study that was part of the Google Data Analytics Certificate.

Dataset describes a bike-share company in Chicago that is seeking to understand its user demographics and redesign it's marketing strategy to increase the number of users opting in for annual memberships. Dataset only covers data for 2019.


```python
import numpy as np
import pandas as pd
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
%matplotlib inline
```


```python
import datetime
```


```python
bks_df = pd.read_csv('bike_share_rawdata.csv')
```


```python
bks_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trip_id</th>
      <th>start_time</th>
      <th>end_time</th>
      <th>bikeid</th>
      <th>tripduration</th>
      <th>from_station_id</th>
      <th>from_station_name</th>
      <th>to_station_id</th>
      <th>to_station_name</th>
      <th>usertype</th>
      <th>gender</th>
      <th>birthyear</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21742443</td>
      <td>01-01-2019 00:04</td>
      <td>01-01-2019 00:11</td>
      <td>2167</td>
      <td>390</td>
      <td>199</td>
      <td>Wabash Ave &amp; Grand Ave</td>
      <td>84</td>
      <td>Milwaukee Ave &amp; Grand Ave</td>
      <td>Subscriber</td>
      <td>Male</td>
      <td>1989.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21742444</td>
      <td>01-01-2019 00:08</td>
      <td>01-01-2019 00:15</td>
      <td>4386</td>
      <td>441</td>
      <td>44</td>
      <td>State St &amp; Randolph St</td>
      <td>624</td>
      <td>Dearborn St &amp; Van Buren St (*)</td>
      <td>Subscriber</td>
      <td>Female</td>
      <td>1990.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21742445</td>
      <td>01-01-2019 00:13</td>
      <td>01-01-2019 00:27</td>
      <td>1524</td>
      <td>829</td>
      <td>15</td>
      <td>Racine Ave &amp; 18th St</td>
      <td>644</td>
      <td>Western Ave &amp; Fillmore St (*)</td>
      <td>Subscriber</td>
      <td>Female</td>
      <td>1994.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21742446</td>
      <td>01-01-2019 00:13</td>
      <td>01-01-2019 00:43</td>
      <td>252</td>
      <td>1783</td>
      <td>123</td>
      <td>California Ave &amp; Milwaukee Ave</td>
      <td>176</td>
      <td>Clark St &amp; Elm St</td>
      <td>Subscriber</td>
      <td>Male</td>
      <td>1993.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21742447</td>
      <td>01-01-2019 00:14</td>
      <td>01-01-2019 00:20</td>
      <td>1170</td>
      <td>364</td>
      <td>173</td>
      <td>Mies van der Rohe Way &amp; Chicago Ave</td>
      <td>35</td>
      <td>Streeter Dr &amp; Grand Ave</td>
      <td>Subscriber</td>
      <td>Male</td>
      <td>1994.0</td>
    </tr>
  </tbody>
</table>
</div>



Based on the raw data provided we can explore the following questions:

- How many users are there in each user type (Subscriber/Customer)?
- How many users are there in each gender category?
- What is the average age of each user?
- What is the distribution of rides by age?
- What is the distribution of rides by day of week and month?
- What is the trip duration? What is the average trip duration across user types, ages and gender?
- When do trips usually begin? What time periods are the busiest?
- Which bike stations do the most rides begin/end at?

# Data Cleaning and Preparation

In this stage the data is checked for accuracy and completeness prior to beginning the analysis. 

- Removing extraneous data and outliers.
- Filling in missing values.
- Conforming data to a standardized pattern.
- Identifying errors revealed when new variables are created.
- Deleting data that cannot be corrected.

Note: There are no unique identifiers for each user so the data does not account for multiple trips made by users. Therefore, it will not be able to tell us how many unique users are associated with the ride data. 

# Checking for missing values


```python
bks_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 365069 entries, 0 to 365068
    Data columns (total 12 columns):
     #   Column             Non-Null Count   Dtype  
    ---  ------             --------------   -----  
     0   trip_id            365069 non-null  int64  
     1   start_time         365069 non-null  object 
     2   end_time           365069 non-null  object 
     3   bikeid             365069 non-null  int64  
     4   tripduration       365069 non-null  int64  
     5   from_station_id    365069 non-null  int64  
     6   from_station_name  365069 non-null  object 
     7   to_station_id      365069 non-null  int64  
     8   to_station_name    365069 non-null  object 
     9   usertype           365069 non-null  object 
     10  gender             345358 non-null  object 
     11  birthyear          347046 non-null  float64
    dtypes: float64(1), int64(5), object(6)
    memory usage: 33.4+ MB
    

The gender and birthyear columns appear to have missing values. 

The missing values originate from a variety of reasons:
- The user may have forgotten to enter the value.  
- For missing gender values, the user may have identified with a different gender identify than the two options provided. 
- Hardware or software error in the bikes is affecting accuracy of trip data and so on.

As we are interested in Customer data it is important to identify how many of the missing vales belong to the Customers. Additionally, it is recommended to make both the above values a required field for user data collection purposes to avoid missing data in the future. 


```python
##Rows with missing value for gender

bks_df['gender'].isnull().value_counts() 
```




    False    345358
    True      19711
    Name: gender, dtype: int64




```python
#Number of Customers with missing gender
bks_df[(bks_df['gender'].isnull()==True)& (bks_df['usertype']=="Customer")].count()
```




    trip_id              17228
    start_time           17228
    end_time             17228
    bikeid               17228
    tripduration         17228
    from_station_id      17228
    from_station_name    17228
    to_station_id        17228
    to_station_name      17228
    usertype             17228
    gender                   0
    birthyear              103
    dtype: int64



The majority <i>(17228/19711)</i> of the missing gender values belong to the <b>Customer</b> data.


```python
##Rows with missing value for birthyear

bks_df['birthyear'].isnull().value_counts() 
```




    False    347046
    True      18023
    Name: birthyear, dtype: int64




```python
#Number of Customers with missing birthyears
bks_df[(bks_df['birthyear'].isnull()==True)& (bks_df['usertype']=="Customer")].count()
```




    trip_id              17126
    start_time           17126
    end_time             17126
    bikeid               17126
    tripduration         17126
    from_station_id      17126
    from_station_name    17126
    to_station_id        17126
    to_station_name      17126
    usertype             17126
    gender                   1
    birthyear                0
    dtype: int64




```python
#Number of Subscribers with missing birthyears
bks_df[(bks_df['birthyear'].isnull()==True)& (bks_df['usertype']=="Subscriber")].count()
```




    trip_id              897
    start_time           897
    end_time             897
    bikeid               897
    tripduration         897
    from_station_id      897
    from_station_name    897
    to_station_id        897
    to_station_name      897
    usertype             897
    gender                 0
    birthyear              0
    dtype: int64



The majority <i>(17126/18023)</i> of the missing birthyear values belong to the <b>Customer</b> data.

### Replacing Missing Values

As there are a significant number of Customer data points in the missing data, we need to replace the missing data to get a better picture of the customer behavior. The missing data for the Subscribers is relatively less significant compared to the number of Subscribers.

- We can replace missing gender values with a "Other Gender Identity" since we do not have the specific context stating otherwise. 
- We can replace missing birthyear vales with the average age for Customers. However, this will skew the age customer data towards the average of the customers who were not missing birthyear values.



```python
bks_df['gender'] = bks_df['gender'].fillna("Other Gender Identity")
```


```python
## We have replaced all missing gender values

bks_df['gender'].value_counts() 
```




    Male                     278440
    Female                    66918
    Other Gender Identity     19711
    Name: gender, dtype: int64




```python
#Identifying average birthyear for each usertype
avg_age = bks_df.groupby('usertype')['birthyear'].mean().round()
avg_age
```




    usertype
    Customer      1989.0
    Subscriber    1982.0
    Name: birthyear, dtype: float64




```python
#The following line of code helps identify all values in the birthyear column that are null for customer data:
#bks_df.loc[(bks_df['usertype']=='Customer') & (bks_df['birthyear'].isnull()==True),['birthyear']]

#Replacing missing values with the average customer age
bks_df.loc[(bks_df['usertype']=='Customer') & (bks_df['birthyear'].isnull()==True),['birthyear']] = bks_df.loc[(bks_df['usertype']=='Customer') & (bks_df['birthyear'].isnull()==True),['birthyear']].fillna(1989)
```


```python
##Rows with missing value for birthyear have been replaced

bks_df['birthyear'].isnull().value_counts() 
```




    False    364172
    True        897
    Name: birthyear, dtype: int64



The only remaining values that are mising are the subscribers with missing birthyear.

### Deleting Missing Values


```python
bks_df.dropna(subset=['birthyear'], inplace=True)
```


```python
## We have deleted all rows with missing birthyear values

bks_df['birthyear'].isnull().value_counts() 
```




    False    364172
    Name: birthyear, dtype: int64




```python
bks_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 364172 entries, 0 to 365068
    Data columns (total 12 columns):
     #   Column             Non-Null Count   Dtype  
    ---  ------             --------------   -----  
     0   trip_id            364172 non-null  int64  
     1   start_time         364172 non-null  object 
     2   end_time           364172 non-null  object 
     3   bikeid             364172 non-null  int64  
     4   tripduration       364172 non-null  int64  
     5   from_station_id    364172 non-null  int64  
     6   from_station_name  364172 non-null  object 
     7   to_station_id      364172 non-null  int64  
     8   to_station_name    364172 non-null  object 
     9   usertype           364172 non-null  object 
     10  gender             364172 non-null  object 
     11  birthyear          364172 non-null  float64
    dtypes: float64(1), int64(5), object(6)
    memory usage: 36.1+ MB
    

All missing values have been dealt with.

## Correcting formatting issues in data


```python
#Converting start time and end time to datetime values 

bks_df['start_time'] = pd.to_datetime(bks_df['start_time'])
bks_df['end_time'] = pd.to_datetime(bks_df['end_time'])
```


```python
#Converting birthyear to int format
bks_df['birthyear'] = bks_df['birthyear'].astype(int)
```

## Checking for other issues with values

### Trip start and end times


```python
#Checking for any instances where end time is earlier than the start time

(bks_df['end_time'] > bks_df['start_time']).value_counts()
```




    True     364105
    False        67
    dtype: int64



The error could originate from a variety of reasons:
- Start and end times may have been swapped
- Start and end times may have been entered incorrectly
- Hardware or software error in the bikes is affecting accuracy of trip data and so on. 

As the source of the error is unknown, correcting the values is not possible so the associated rows will be deleted.


```python
#Deleting rows where end time is earlier than start time

bks_df.drop(bks_df[bks_df['end_time'] < bks_df['start_time']].index, inplace = True)
```


```python
#Rows have been sucessfully deleted

(bks_df['end_time'] > bks_df['start_time']).value_counts()
```




    True    364105
    dtype: int64



### Trip Duration

We have a column with <b>"tripduration"</b> that appears to be in seconds. However we have to check if the values are consistent with the actual trip duration.


```python
# Calculating Trip duration in a new column
bks_df['trip_duration'] = bks_df['end_time'] - bks_df['start_time']
```

The trip duration calculated will be as a timedelta value which indicates the absolute time difference (for example "1 day 10:10:10")


```python
#Checking if any trips exceed 1 day

(bks_df['trip_duration']>pd.Timedelta("1 days")).value_counts() 
```




    False    363720
    True        385
    Name: trip_duration, dtype: int64




```python
#Checking the longest trip duration

bks_df['trip_duration'].max()
```




    Timedelta('302 days 12:38:00')



The bike share trips are meant for commutes around the city and it would not make sense for a trip to exceed one day. Therefore, we have incorrect values that need to be removed. 


```python
#Deleting rows where end time is earlier than start time

bks_df.drop(bks_df[bks_df['trip_duration']>pd.Timedelta("1 days")].index, inplace = True)
```


```python
#Rows with duration trips duration exceeding 1 day have been deleted

(bks_df['trip_duration']>pd.Timedelta("1 days")).value_counts() 
```




    False    363720
    Name: trip_duration, dtype: int64



Now that we have calculated the trip duration and removed the erroneous values, we can convert it to seconds


```python
#Creating a new column where we will convert 'trip_duration' into seconds
bks_df['trip_duration_seconds'] = bks_df['trip_duration'].dt.total_seconds()
```


```python
#Comparing values in the trip_duration_seconds to tripduration
(bks_df['trip_duration_seconds'] == bks_df['tripduration']).value_counts()
```




    False    359096
    True       4624
    dtype: int64



It appears the majority of the values do not match, which presents three possibilities:
1. The <b>tripduration</b> values in the raw data may be capturing actual ride time in which case they may not include time the user is taking to unlock the bike, get on the bike etc. In this instance we can expect that the <b>tripduration</b> values to be lower than <b>trip_duration_seconds</b> values.
2. The <b>tripduration</b> values may be inaccurate, which could be leading to the discrepancy. 
3. The <b>start_time</b> and <b>end_time</b> values may be inaccurate, which could be leading to the discrepancy. 


```python
#Comparing values in the trip_duration_seconds to tripduration to check if tripduration is always less than trip_duration_seconds
(bks_df['trip_duration_seconds'] > bks_df['tripduration']).value_counts()
```




    False    183346
    True     180374
    dtype: int64



The <b>tripduration</b> values are not consistently lower than the <b>tripduration</b> values, so we can rule out the first possibility. 

In the absence of specific context, it is not possible to verify if the second or third possibility are true. Therefore, we will make the assumption that the <b>start_time</b> and <b>end_time</b> values are accurate and delete the <b>tripduration</b> column.


```python
#Deleting the tripduration column
bks_df.drop('tripduration',axis=1, inplace=True)
```

## Creating new features

In this stage we are adding new features that will provide more insight into the data.

### Age


```python
from datetime import date

bks_df['age'] = date.today().year - bks_df['birthyear']
```

### Day of week


```python
#Identifying which day of the week the ride occured

bks_df['day_of_week'] = bks_df['end_time'].dt.day_name()
```

### Month


```python
#Identifying which month the ride occured

bks_df['month'] = bks_df['end_time'].apply(lambda time: time.month)
```


```python
#We needs to convert the values in the Month column from numbers to names of Months

dmap = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
```


```python
#Mapping our new dictionary to the Month column in the Dataframe

bks_df['month'] = bks_df['month'].map(dmap)
```

# Data Analysis and Visualization

In this stage, we will examine the data to identify any patterns, trends and relationships between the variables. It will help us analyze the data and extract insights that can be used to make decisions.

Data Visualization will gives us a clear idea of what the data means by giving it visual context.

### Checking for any correlation in the data


```python
#Checking for any obvious correlation in the numeric variables
sns.heatmap(bks_df.corr())
```




    <AxesSubplot:>




    
![png](output_72_1.png)
    


There does not appear to be any significant and meaningful correlation between variables

### User Types

The users are divided into two types: <b> Subscribers</b> and <b>Customer</b>. 
    
- Users who purchase an annual membership fee are referred to as Subscribers 
- Users who purchase single-ride or full-day passes are referred to as Customers


```python
#Number of trips associated with users in each category

bks_df['usertype'].value_counts()
```




    Subscriber    340685
    Customer       23035
    Name: usertype, dtype: int64




```python
Subscriber = 340685
Customer = 23035

round(Subscriber/Customer,1)
```




    14.8



There are <b>14.8 times</b> more Subscribers than Customers


```python
user_count = bks_df['usertype']
sns.countplot(x=user_count,data=bks_df,palette='viridis')
plt.title("Number of Users of Each Type", fontsize=20)
```




    Text(0.5, 1.0, 'Number of Users of Each Type')




    
![png](output_79_1.png)
    


The users are overwhelmingly in the <b>Subscriber</b> category. The marketing strategy is therefore aimed at converting the relatively small number of users from Customers to Subscribers.

### Gender


```python
#Number of trips associated with users of each gender

bks_df['gender'].value_counts()
```




    Male                     278169
    Female                    66851
    Other Gender Identity     18700
    Name: gender, dtype: int64




```python
gender_df = bks_df['gender'].value_counts()

explode = (0, 0.1, 0.2)

gender_df.plot.pie(figsize=(6, 6), autopct="%.1f",fontsize=20,labels=None, legend=True, explode=explode).set_ylabel('')
plt.title("Percentage of Users in Each Gender Category", fontsize=20)

# autopct="%.1f" shows the percentage to 1 decimal place 
#.set_ylabel('') and can be added to remove the usertype label on the left of the chart.set_ylabel('')
# The explosion array specifies the fraction of the radius with which to offset each slice.
```




    Text(0.5, 1.0, 'Percentage of Users in Each Gender Category')




    
![png](output_83_1.png)
    


## Age

<b> Age statistics for all users</b>


```python
print("Youngest user is ", bks_df['age'].min(),"years old")
print("Oldest user is ", bks_df['age'].max(),"years old")
print("Average user is ", round(bks_df['age'].mean()),"years old")
print("Average standard deviation in age is", round(bks_df['age'].std()),"years")
```

    Youngest user is  18 years old
    Oldest user is  121 years old
    Average user is  39 years old
    Average standard deviation in age is 11 years
    

<b> Age statistics by gender</b>


```python
# Identifying the average age by gender 

bks_df.groupby('gender')['age'].mean().round()
```




    gender
    Female                   38.0
    Male                     40.0
    Other Gender Identity    33.0
    Name: age, dtype: float64




```python
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)

sns.histplot(data=bks_df, x="age",binwidth=2,kde=True)
plt.title("Age Breakdown for Rides", fontsize=20)
```




    Text(0.5, 1.0, 'Age Breakdown for Rides')




    
![png](output_89_1.png)
    


Users <b>between the ages of 32-34</b> make the most rides overall.


```python
graph = sns.FacetGrid(bks_df, col="usertype", height=6)
graph.map_dataframe(sns.histplot,x="age",binwidth=2)

#Setting the title for the FacetGrid 
graph.fig.subplots_adjust(top=0.85)
graph.fig.suptitle('Age breakdown by Usertype for Rides', fontsize=20)
```




    Text(0.5, 0.98, 'Age breakdown by Usertype for Rides')




    
![png](output_91_1.png)
    


- Subscribers <b>between the ages of 32-34</b> make the most rides.
- Customers <b>between the ages of 31-33</b> make the most rides.


```python
graph = sns.FacetGrid(bks_df, row="gender",col="usertype", height=6)
graph.map_dataframe(sns.histplot,x="age",binwidth=2)

#Setting the title for the FacetGrid 
graph.fig.subplots_adjust(top=0.85)
graph.fig.suptitle('Age breakdown by Usertype and Gender for Rides', fontsize=20)
```




    Text(0.5, 0.98, 'Age breakdown by Usertype and Gender for Rides')




    
![png](output_93_1.png)
    


<b>Subscribers</b>

- Male users <b>between the ages of 32-34</b> make the most rides.
- Female users <b>between the ages of 31-33</b> make the most rides.
- Other Gender Identity users <b>between the ages of 38-40</b> make the most rides.

<b>Customers</b>

- Male users <b>between the ages of 27-29</b> make the most rides.
- Female users <b>between the ages of 25-27</b> make the most rides.
- Other Gender Identity users <b>between the ages of 31-33</b> make the most rides

## Day of Week


```python
#Number of rides on any given day of the week
bks_df['day_of_week'].value_counts()
```




    Thursday     66543
    Tuesday      59524
    Wednesday    57915
    Friday       57651
    Monday       47113
    Sunday       38558
    Saturday     36416
    Name: day_of_week, dtype: int64




```python
fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

# order = bks_df['day_of_week'].value_counts().index helps us sort the count plot by the value counts

sns.countplot(x='day_of_week',data=bks_df,order = bks_df['day_of_week'].value_counts().index,palette='viridis')
plt.title("Bike Share Ride Breakdown by Day of Week", fontsize=20)
```




    Text(0.5, 1.0, 'Bike Share Ride Breakdown by Day of Week')




    
![png](output_97_1.png)
    


<b>Thursdays</b> are the busiest days of the week for rides

## Month


```python
bks_df['month'].value_counts()
```




    March        127161
    February      65249
    January       54228
    November      15997
    April         15044
    May           14915
    August        14769
    June          13483
    July          13382
    December      11520
    September      9087
    October        8885
    Name: month, dtype: int64




```python
fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#order = bks_df['month'].value_counts().index helps us sort the count plot by the value counts

sns.countplot(x='month',data=bks_df,palette='viridis')
plt.title("Bike Share Ride Breakdown by Month", fontsize=20)
```




    Text(0.5, 1.0, 'Bike Share Ride Breakdown by Month')




    
![png](output_101_1.png)
    


There is a steep increase in rides starting from January with <b>March</b> being the busiest month for rides. The ridership drops off in April and does not fluctuate as significantly for the rest of the year.

## Trip Duration

<b> Trip Duration Statistics </b>


```python
max_trip = bks_df['trip_duration'].max()
min_trip = bks_df['trip_duration'].min()
mean_trip = bks_df['trip_duration'].mean()
print("The longest trip duration was",max_trip,"\nThe shortest trip duration was",min_trip,"\nThe average trip duration was",mean_trip)
```

    The longest trip duration was 0 days 23:29:00 
    The shortest trip duration was 0 days 00:01:00 
    The average trip duration was 0 days 00:12:36.275156713
    


```python
# Analyzing the trip duration times to identify any trends

fig_dims = (10, 5)
fig, ax = plt.subplots(figsize=fig_dims)

trip_duration_counts = bks_df['trip_duration'].value_counts()

trip_duration_counts.plot()

plt.title("Ride Duration Times", fontsize=20)
```




    Text(0.5, 1.0, 'Ride Duration Times')




    
![png](output_106_1.png)
    



```python
#Checking how many trips exceed the 1 hour mark
(bks_df['trip_duration']<pd.Timedelta("1 hour")).value_counts() 
```




    True     359683
    False      4037
    Name: trip_duration, dtype: int64



As seen in the plot and the data above the <b>majority of trips do not exceed an hour.</b>

### Trip Duration by User Type


```python
#Identifying the average trip duration for each user type
user_df = bks_df.groupby('usertype')['trip_duration_seconds'].mean().round()

#Converting the series user_df to a dataframe
user_df = user_df.to_frame()

#Converting 'trip_duration_seconds' back into timedelta format for readability 
user_df['average_trip_duration'] = user_df['trip_duration_seconds'].apply(lambda time: (datetime.timedelta(seconds = time))) 

#Dropping 'trip_duration_seconds' as it is no longer needed
user_df.drop('trip_duration_seconds',axis=1, inplace=True)

#Identifying number of trips for each user type
user_df['number_of_trips'] = bks_df['usertype'].value_counts()

user_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>average_trip_duration</th>
      <th>number_of_trips</th>
    </tr>
    <tr>
      <th>usertype</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Customer</th>
      <td>0 days 00:34:23</td>
      <td>23035</td>
    </tr>
    <tr>
      <th>Subscriber</th>
      <td>0 days 00:11:08</td>
      <td>340685</td>
    </tr>
  </tbody>
</table>
</div>




```python
cus = pd.to_timedelta("0 days 00:34:23")
sub = pd.to_timedelta("0 days 00:11:08")

round(cus/sub,1)
```




    3.1



On average, <b>Customers make trips that are 3.1 times longer than Subscribers.</b>

<b>Note:</b> it would be useful to know the frequency of trips made by individual users to understand whether individual Customers trips more or less often than individual Subscribers. The marketing strategy may differ depending on the result. For example it is harder to convince a customer who takes a long trip once every few months to purchase an annual membership as opposed to a customer who takes a long trip more frequently. 

However, as mentioned before there are no unique identifiers for each user so the data does not account for multiple trips made by users. Therefore, it will not be able to tell us how many unique users are associated with the ride data. 

### Trip Duration, Number of Trips and Demographic Data


```python
#Identifying the average trip duration for each user type
user_df = bks_df.groupby(['usertype','gender'])['trip_duration_seconds'].mean().round()

#Converting the series user_df to a dataframe
user_df = user_df.to_frame()

#Converting 'trip_duration_seconds' back into timedelta format for readability 
user_df['average_trip_duration'] = user_df['trip_duration_seconds'].apply(lambda time: (datetime.timedelta(seconds = time))) 

#Dropping 'trip_duration_seconds' as it is no longer needed
user_df.drop('trip_duration_seconds',axis=1, inplace=True)

#Identifying number of trips for each user type
user_df['number_of_trips'] = bks_df.groupby('usertype')['gender'].value_counts()

#Identifying average age by usertype and gender
user_df['average_age'] = bks_df.groupby(['usertype','gender'])['age'].mean().round()

user_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>average_trip_duration</th>
      <th>number_of_trips</th>
      <th>average_age</th>
    </tr>
    <tr>
      <th>usertype</th>
      <th>gender</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Customer</th>
      <th>Female</th>
      <td>0 days 00:35:40</td>
      <td>1871</td>
      <td>31.0</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>0 days 00:29:08</td>
      <td>4045</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>Other Gender Identity</th>
      <td>0 days 00:35:29</td>
      <td>17119</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">Subscriber</th>
      <th>Female</th>
      <td>0 days 00:11:53</td>
      <td>64980</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>0 days 00:10:54</td>
      <td>274124</td>
      <td>40.0</td>
    </tr>
    <tr>
      <th>Other Gender Identity</th>
      <td>0 days 00:19:45</td>
      <td>1581</td>
      <td>43.0</td>
    </tr>
  </tbody>
</table>
</div>



In the <b>Customer</b> category
- Female users are making the longest trips.
- Average age range is 31-32.

In the <b>Subscriber</b> category
- Other Gender Identity users are making the longest trips.
- Average age range is 38-43.

Overall,
- On average, Customers are make longer trips than Subscribers in each gender category. 
- On average, Customers are younger than Subscribers. 
- Male users make the most trips in both categories. 

### Average Trip Duration by Day of Week


```python
#Identifying the average trip on any given day of the week
day_of_week = bks_df.groupby(['usertype','day_of_week'])['trip_duration_seconds'].mean().round()

#Converting the series day_of_week to a dataframe
day_of_week = day_of_week.to_frame()

#Converting 'trip_duration_seconds' back into timedelta format for readability 
day_of_week['average_trip_duration'] = day_of_week['trip_duration_seconds'].apply(lambda time: (datetime.timedelta(seconds = time))) 

#Dropping 'trip_duration_seconds' as it is no longer needed
day_of_week.drop('trip_duration_seconds',axis=1, inplace=True)

#Sorting values by'average_trip_duration'
day_of_week.sort_values(by='average_trip_duration')

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>average_trip_duration</th>
    </tr>
    <tr>
      <th>usertype</th>
      <th>day_of_week</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">Subscriber</th>
      <th>Friday</th>
      <td>0 days 00:10:44</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>0 days 00:10:58</td>
    </tr>
    <tr>
      <th>Monday</th>
      <td>0 days 00:10:59</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>0 days 00:11:02</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>0 days 00:11:14</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>0 days 00:11:31</td>
    </tr>
    <tr>
      <th>Saturday</th>
      <td>0 days 00:11:46</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Customer</th>
      <th>Monday</th>
      <td>0 days 00:32:39</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>0 days 00:33:15</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>0 days 00:33:18</td>
    </tr>
    <tr>
      <th>Saturday</th>
      <td>0 days 00:33:31</td>
    </tr>
    <tr>
      <th>Friday</th>
      <td>0 days 00:33:43</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>0 days 00:34:37</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>0 days 00:38:39</td>
    </tr>
  </tbody>
</table>
</div>



Note: Directly using <b> bks_df.groupby(['usertype','day_of_week'])['trip_duration'].mean() </b> will not work because Python will not have any numeric data to aggregate

- On average, <b>Customers</b> make the longest trips on <b>Saturday</b> and the shortest trips on <b>Friday</b>
- On average, <b>Subscribers</b> make the longest trips on <b>Sunday</b> and the shortest trips on <b>Friday</b>

### Number of Trips by Day of Week


```python
#Identifying number of trips on a given day of the week by user type
number_of_trips = bks_df.groupby('usertype')['day_of_week'].value_counts()

#Converting the series number_of_trips to a dataframe
number_of_trips = number_of_trips.to_frame()

number_of_trips
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>day_of_week</th>
    </tr>
    <tr>
      <th>usertype</th>
      <th>day_of_week</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="7" valign="top">Customer</th>
      <th>Saturday</th>
      <td>4752</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>3722</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>3637</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>3145</td>
    </tr>
    <tr>
      <th>Friday</th>
      <td>2800</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>2773</td>
    </tr>
    <tr>
      <th>Monday</th>
      <td>2206</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">Subscriber</th>
      <th>Thursday</th>
      <td>63398</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>56751</td>
    </tr>
    <tr>
      <th>Friday</th>
      <td>54851</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>54193</td>
    </tr>
    <tr>
      <th>Monday</th>
      <td>44907</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>34921</td>
    </tr>
    <tr>
      <th>Saturday</th>
      <td>31664</td>
    </tr>
  </tbody>
</table>
</div>



- <b>Customers</b> make the most trips on <b>Saturday</b> and the least trips on <b>Monday</b>
- <b>Subscribers</b> make the most trips on <b>Thursday</b> and the least trips on <b>Saturday</b>

### Trip Start Times


```python
# Analyzing the trip start times to identify how many trips begin at various times throughout the day.

# Isolating the time portion from bks_df['start_time']
start_time = bks_df['start_time'].dt.time 

start_time_counts = start_time.value_counts()

#X-axis ticks will have to be set manually since the plot is difficult to read otherwise

#Creating a series that will be populated using a for loop 
time_series = []
for x in range(24):
    #Note {:02d} ensures there are leading zeros (01,02,etc.) which is consistent with time formatting
    time_series.append("{:02d}:00:00".format(x))

time_series.append("23:59:00")

fig_dims = (20, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#The x-ticks are set to the time series just created
start_time_counts.plot(xticks=time_series)
plt.title("Ride Start Times throughout the Day", fontsize=20)
```




    Text(0.5, 1.0, 'Ride Start Times throughout the Day')




    
![png](output_125_1.png)
    


As seen above, the number of rides peaks at two different points on any given day, which correspond roughly with daily commute times where users may be heading to and from their workplaces.

<b>Morning Commute:</b> Rides begins to rise after 5 AM, reaching a peak between 8 AM and 9 AM and then plateau around 10 AM.

<b>Late Afternoon/Evening Commute:</b> Rides begins to rise after 3 PM, reaching a peak sometime after 5 PM peak and then continue to decrease until the end of the day. 


```python
# Analyzing the trip start times to identify how many trips begin at various times throughout the day.

#Looking at Customer rides specifically
customer_trips = bks_df[bks_df["usertype"]=="Customer"]

# Isolating the time portion from bks_df['start_time']
start_time = customer_trips['start_time'].dt.time 

start_time_counts = start_time.value_counts()

#X-axis ticks will have to be set manually since the plot is difficult to read otherwise

#Creating a series that will be populated using a for loop 
time_series = []
for x in range(24):
    #Note {:02d} ensures there are leading zeros (01,02,etc.) which is consistent with time formatting
    time_series.append("{:02d}:00:00".format(x))

time_series.append("23:59:00")

fig_dims = (20, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#The x-ticks are set to the time series just created
start_time_counts.plot(xticks=time_series)
plt.title("Customer Ride Start Times throughout the Day", fontsize=20)
```




    Text(0.5, 1.0, 'Customer Ride Start Times throughout the Day')




    
![png](output_127_1.png)
    


Customer trips follow a different trend than the data overall. 

The rides began to rise after 6 AM, reaching a peak sometime after 3 PM and then continue to decrease until the end of the day. 

### Trip Start/End Locations

<b>Top ten bike stations where rides begin</b>


```python
from_stations_df = bks_df['from_station_name'].value_counts().head(10)
from_stations_df = from_stations_df.to_frame()

#The index for this dataframe contains the station names so we want to put that information in a seaprate column
#Resetting the index (also creates a new column called 'index')
from_stations_df.reset_index(level=0, inplace=True)

#Renaming 'from_station_name' which contains value counts to 'number_of_trips_start' and renaming the 'index' column to 'start_station_name'
from_stations_df.rename(columns={'from_station_name': 'number_of_trips_start', 'index':'start_station_name'}, inplace=True)

from_stations_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>start_station_name</th>
      <th>number_of_trips_start</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Clinton St &amp; Washington Blvd</td>
      <td>7653</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Clinton St &amp; Madison St</td>
      <td>6512</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Canal St &amp; Adams St</td>
      <td>6335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Columbus Dr &amp; Randolph St</td>
      <td>4651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Canal St &amp; Madison St</td>
      <td>4569</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kingsbury St &amp; Kinzie St</td>
      <td>4389</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Michigan Ave &amp; Washington St</td>
      <td>3986</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Franklin St &amp; Monroe St</td>
      <td>3509</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dearborn St &amp; Monroe St</td>
      <td>3243</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LaSalle St &amp; Jackson Blvd</td>
      <td>3234</td>
    </tr>
  </tbody>
</table>
</div>



<b>Top ten bike stations where rides end</b>


```python
to_stations_df = bks_df['to_station_name'].value_counts().head(10)
to_stations_df = to_stations_df.to_frame()

#Resetting the index 
to_stations_df.reset_index(level=0, inplace=True)

#Renaming columns
to_stations_df.rename(columns={'to_station_name': 'number_of_trips_end', 'index':'end_station_name'}, inplace=True)

to_stations_df 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>end_station_name</th>
      <th>number_of_trips_end</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Clinton St &amp; Washington Blvd</td>
      <td>7657</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Clinton St &amp; Madison St</td>
      <td>6840</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Canal St &amp; Adams St</td>
      <td>6741</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Canal St &amp; Madison St</td>
      <td>4870</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Michigan Ave &amp; Washington St</td>
      <td>4408</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kingsbury St &amp; Kinzie St</td>
      <td>4368</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LaSalle St &amp; Jackson Blvd</td>
      <td>3302</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Clinton St &amp; Lake St</td>
      <td>3293</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Dearborn St &amp; Monroe St</td>
      <td>3133</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Clinton St &amp; Jackson Blvd (*)</td>
      <td>3111</td>
    </tr>
  </tbody>
</table>
</div>



There's a lot of overlap in the top 10 stations where rides begin/end so it is useful to identify which stations appear in both lists.

<b>Stations that appear on both lists</b>


```python
#Creating two series each with the names of the top 10 stations
start = from_stations_df['start_station_name']
end = to_stations_df['end_station_name']

#Identifying the intersection of both series
pd.Series(list(set(start) & set(end)))
```




    0    Michigan Ave & Washington St
    1       LaSalle St & Jackson Blvd
    2         Dearborn St & Monroe St
    3             Canal St & Adams St
    4        Kingsbury St & Kinzie St
    5    Clinton St & Washington Blvd
    6         Clinton St & Madison St
    7           Canal St & Madison St
    dtype: object



The bike stations listed are <b>points where the most rides both begin and end</b>, which makes them ideal locations for marketing promotions.  

We can further explore what the top stations are by usertype. As we are primarily interested in converting Customers to Subscribers, it is valuable to know which stations marketing efforts should be focused on to target customers.

<b>Top ten bike stations where rides begin for <i>Customers</i> </b>


```python
#Identifying Customers
customers_from_stations = bks_df[bks_df['usertype']=="Customer"]

#Identifying top 5 stations where Customers begin trips from
customers_from_stations = customers_from_stations.groupby('usertype')['from_station_name'].value_counts().head(10)
customers_from_stations = customers_from_stations.to_frame()

#Renaming columns
customers_from_stations.rename(columns={'from_station_name': 'number_of_trips'}, inplace=True)

customers_from_stations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>number_of_trips</th>
    </tr>
    <tr>
      <th>usertype</th>
      <th>from_station_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">Customer</th>
      <th>Streeter Dr &amp; Grand Ave</th>
      <td>1218</td>
    </tr>
    <tr>
      <th>Lake Shore Dr &amp; Monroe St</th>
      <td>1140</td>
    </tr>
    <tr>
      <th>Shedd Aquarium</th>
      <td>833</td>
    </tr>
    <tr>
      <th>Millennium Park</th>
      <td>622</td>
    </tr>
    <tr>
      <th>Michigan Ave &amp; Oak St</th>
      <td>386</td>
    </tr>
    <tr>
      <th>Adler Planetarium</th>
      <td>361</td>
    </tr>
    <tr>
      <th>Dusable Harbor</th>
      <td>342</td>
    </tr>
    <tr>
      <th>Michigan Ave &amp; Washington St</th>
      <td>342</td>
    </tr>
    <tr>
      <th>Field Museum</th>
      <td>298</td>
    </tr>
    <tr>
      <th>Buckingham Fountain (Temp)</th>
      <td>256</td>
    </tr>
  </tbody>
</table>
</div>



<b>Top five bike stations where rides end for <i>Customers</i> </b>


```python
#Identifying Customers
customers_to_stations = bks_df[bks_df['usertype']=="Customer"]

#Identifying top 5 stations where Customers begin trips from
customers_to_stations = customers_to_stations.groupby('usertype')['to_station_name'].value_counts().head(10)
customers_to_stations = customers_to_stations.to_frame()

#Renaming columns
customers_to_stations.rename(columns={'to_station_name': 'number_of_trips'}, inplace=True)

customers_to_stations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>number_of_trips</th>
    </tr>
    <tr>
      <th>usertype</th>
      <th>to_station_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">Customer</th>
      <th>Streeter Dr &amp; Grand Ave</th>
      <td>1931</td>
    </tr>
    <tr>
      <th>Lake Shore Dr &amp; Monroe St</th>
      <td>891</td>
    </tr>
    <tr>
      <th>Millennium Park</th>
      <td>818</td>
    </tr>
    <tr>
      <th>Shedd Aquarium</th>
      <td>639</td>
    </tr>
    <tr>
      <th>Michigan Ave &amp; Oak St</th>
      <td>447</td>
    </tr>
    <tr>
      <th>Michigan Ave &amp; Washington St</th>
      <td>409</td>
    </tr>
    <tr>
      <th>Theater on the Lake</th>
      <td>348</td>
    </tr>
    <tr>
      <th>Adler Planetarium</th>
      <td>295</td>
    </tr>
    <tr>
      <th>Lake Shore Dr &amp; North Blvd</th>
      <td>286</td>
    </tr>
    <tr>
      <th>Wabash Ave &amp; Grand Ave</th>
      <td>242</td>
    </tr>
  </tbody>
</table>
</div>



<b>Stations that appear on both lists</b>


```python
#As the station names are part of the MultiIndex for the data frames we need to rest the index
#Makes the username a new column
customers_from_stations.reset_index(level=0, inplace=True)
customers_to_stations.reset_index(level=0, inplace=True)

#Makes the to_station_name and from_station_name a new column
customers_from_stations.reset_index(level=0, inplace=True)
customers_to_stations.reset_index(level=0, inplace=True)
```


```python
#Creating two series each with the names of the top 10 stations
start = customers_from_stations['from_station_name']
end = customers_to_stations['to_station_name']

#Identifying the intersection of both series
pd.Series(list(set(start) & set(end)))
```




    0    Michigan Ave & Washington St
    1           Michigan Ave & Oak St
    2                 Millennium Park
    3               Adler Planetarium
    4                  Shedd Aquarium
    5         Streeter Dr & Grand Ave
    6       Lake Shore Dr & Monroe St
    dtype: object



The bike stations listed are <b>points where the most rides both begin and end for <i>Customers</i></b>, which makes them ideal locations for marketing promotions <i>specifically targeted</i> at them.

# Summary of Customer Data

- There are 23,035 customer. 
- There are 14.8 times more Subscribers than Customers

<b>Age</b>
- On average, Customers are younger than Subscribers.
- Customers between the ages of 31-33 make the most rides.

    - Male users between the ages of 27-29 make the most rides.
    - Female users between the ages of 25-27 make the most rides.
    - Other Gender Identity users between the ages of 31-33 make the most rides

<b>Trip Duration</b>

- The majority of trips do not exceeed an hour.
- On average, Customers make trips that are 3.1 times longer than Subscribers.

    - Male users make the most trips.
    - Female users are making the longest trips.
    - Average age range is 31-32.

<b>Day of Week</b>
- On average, Customers make the longest trips on Saturday and the shortest trips on Friday
- Customers make the most trips on Saturday and the least trips on Monday

<b>Start Times and Stations</b>
- The rides began to rise after 6 AM, reaching a peak sometime after 3 PM and then continue to decrease until the end of the day.
- The bike stations listed are points where the most rides both begin and end for Customers, which makes them ideal locations for marketing promotions specifically targeted at them.
- The bike stations listed are points where the most rides both begin and end, which makes them ideal locations for marketing promotions.

    - Streeter Dr & Grand Ave
    - Michigan Ave & Washington St
    - Adler Planetarium
    - Lake Shore Dr & Monroe St
    - Shedd Aquarium
    - Millennium Park
    - Michigan Ave & Oak St




# Reccomendations

<b>Focus on what is valuable to Customers</b>

- Customers make longer trips than Subscribers. Longer trips will have a higher cost for the customer. The marketing team could highlight the decrease in cost from switching to becoming a Subscriber. The trip duration does not vary significantly for Customers regardless of the day of the week. Therefore, depending on the frequency of use by each customer, the customer would stand to save a significant amount by switching.
<br>
- Customers have a slight preference for weekend trips, The marketing teams could offer deals for Customers who choose to become Subscribers on a weekend. 
<br>
- Customers also tend to be younger than Subscribers (primarily in their early 30's) so the marketing campaign should cater to the interests of this demographic. Customers may be using the bike share service for personal use such as running errands, travelling to meet someone etc. The marketing team could survey Customers to find out their interests and offer promotions such as discounts at grocery stores, a discounted annual price for referring friends etc. depending on their interests. 
<br>
- The majority of Customers and Subscribers appear to be men. Therefore, the marketing team may want to change their marketing campaign to attract female users and users of other gender identities and expand their user base.
<br>
- Customer rides also appear to not follow the traditional commuting patterns, which may suggest that the customers either do not work traditional office jobs that require them to commute to and from work. Therefore the key time period to market promotions would be around 3 PM when the most rides are occuring. 
<br>
- The Marketing team should distribute promotional materials at the stations identified that see the most traffic, where the majority of Customer tier trips begin and end. These locations are frequented the most by customers and provides an opportunity to entice them to make the switch to become Subscribers.

<b>Additional Data necessary</b><br>
The company should associate a unique customer id with each trip to identify individual customer behaviors. The unique identifiers would reveal information on the frequency of bike trips made by each customer. For example, it would reveal how many trips each customer makes a week on average, which may reveal if there are specific differences or similarities in the two tiers of customers.
