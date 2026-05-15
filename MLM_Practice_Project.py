# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:46:35 2026

@author: Luche_Cameron
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('mlm_leads_mock_dataset.csv')

# checking if data is clean
print('\n' , df.isnull().sum())
print('\nThe amount of duplicated rows is :' , df.duplicated().sum())


df['lead_id'] = df['lead_id'].astype('str')
df['w_converted'] = df['converted'].map({
    0 : 'Not Converted' ,
    1 : 'Successfully Converted'
    })

print(df.info())
print(df.describe())


''' What is the overall conversion rate? '''
#overall conversion insight
overall_conversion= df.groupby('w_converted')['lead_id'].count()
print(f' \nThe overall conversion is {overall_conversion} \n ')
#calculating the rate 

conversion_rate_overall = df['converted'].mean() * 100
print(f'Overall Conversion Rate : {conversion_rate_overall : .2f} %')


#piechart showing the overall conversion rate
labels = ['Not Converted' , 'Successfully Converted']
explode = [0.1, 0]

plt.pie(overall_conversion , labels = labels , 
        explode = explode , autopct = '%.0f%%')
plt.title('Overall Conversion Rate Of Leads At WhimsyLu')
plt.legend(title="Converted",
          loc="center left", 
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()

''' Which platforms generate the highest conversion rates? '''

# Creating a crosstab to segment platforms of conversion rate 
conversion_platform = pd.crosstab(
    df['platform'] , df['w_converted'] , margins = True
    )
print('\n', conversion_platform,'\n')

platform_conversion_rate = df.groupby('platform')['converted'].mean() * 100
print('\n The platform conversion rate is \n' ,platform_conversion_rate)

platform_interest = pd.crosstab (
    df['platform'] , df['interest_level'] , margins = True
    )
print('\n' ,platform_interest, '\n')

# barchart for platform insights 
plt.figure(figsize = (10,5))
ax = platform_conversion_rate.sort_values(ascending=False).plot( kind = 'bar' )

# Adding annotation to the bar grapph for readability
for i, value in enumerate(platform_conversion_rate.sort_values(ascending=False)):
    ax.text(i, value + 0.5, f'{value:.1f}%', ha='center')

    
    
plt.title('Avergave Platform Conversion Rate(%)')
plt.xlabel('Platform')
plt.ylabel('Average Rate')
plt.show()

#interest level impact 
avg_interest_conversion_rate  = df.groupby('interest_level')['converted'].mean()
print('\n The average conversion rate based on interest is : \n' ,
      avg_interest_conversion_rate * 100)

sns.barplot(x = 'interest_level' , y='converted' , data = df)
plt.show()
# correctionn to conversion
cols = ['time_to_contact_hours' , 'num_followups', 'converted' , 'age' ]
corr = df[cols].corr()
sns.heatmap(corr , annot = True , cmap = 'coolwarm')
plt.show()

# lead source insights 
source_conversion = df.groupby('lead_source')['converted'].mean() * 100
print('\nThe average rate of gained customers based on the lead source are (in percentage):\n' , source_conversion.round(2))

source_conversion_unstacked = df.groupby(['lead_source','converted']).size().unstack()
source_conversion_unstacked.plot(kind = 'bar')

plt.title('Leads Conversion by Source')
plt.xlabel('The Source of the Lead')
plt.ylabel('Amount of Leads')

plt.xticks(rotation = 0)
plt.legend(title = 'Converted' , labels = ['No' , 'Yes'])
plt.show()

# Which follow up count has the highest success rate 

success_conversion_rate = df.groupby('num_followups')['converted'].mean().round(3)
print('\n The average percentage of customers gained based on the number of follow-ups is:\n ' ,
      success_conversion_rate * 100 )


#Does the response time affect conversion?
bins = range(0,150,5)
df['time_bin'] = pd.cut(df['time_to_contact_hours'] , bins = bins)
time_conversion = df.groupby('time_bin')['converted'].mean().reset_index()
time_conversion['converted'] = time_conversion['converted'] * 100

time_conversion['time_bin'] = time_conversion['time_bin'].astype(str)
plt.figure(figsize = (10 , 5))

sns.lineplot(
    data = time_conversion,
    x = 'time_bin', 
    y = 'converted',
    marker = 'o')

plt.xticks(rotation = 90)
plt.ylabel('Conversion Rate (%)')
plt.xlabel('TIme to contact (Hours)')
plt.title('Conversion Rate vs Response Time')

plt.show()

#combined insights and analysis

platform_interest = df.groupby(['platform' , 'interest_level'])['converted'].mean().unstack()
sns.heatmap(platform_interest , annot = True , cmap = 'coolwarm')
plt.title('Platform vs Interest Level Conversion Heatmap')
plt.show()


''' 
---INSIGHT---

-The analysis reveals that while the business maintains an overall conversion rate
 of 37.6%, there is a massive performance gap between high-intent leads and people 
 just looking at the products with high-interest prospects being three times more 
 likely to convert.
 
-Currently, efficiency is hindered by a "speed to lead" issue, as conversions 
 drop significantly after the first 10 hours, and a marketing paradox where free
 Organic leads are outperforming Paid ones by nearly 10%.
 
-To maximize revenue, the strategy should shift toward a Lead Scoring System that
 prioritizes immediate human contact for high-value leads while using automation
 for lower-interest groups. 
 
-By enforcing a 4-hour response SLA, standardizing a "Rule of 3" follow-up cadence, 
and focusing high-intent ad spend on Instagram, the business can capitalize on
 its most successful segments without increasing its total marketing budget.


'''


