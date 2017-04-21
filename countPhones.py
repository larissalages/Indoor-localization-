#This code count the number os different phones in each reference point.

import pandas as pd
import csv

#convert csv file in an data frame
df = pd.read_csv('trainingData.csv')

#Group the location attributes
grouped_df = df.groupby(['LONGITUDE','LATITUDE','FLOOR','BUILDINGID','SPACEID','RELATIVEPOSITION'])

#Convert to list
l_grouped_df = list(grouped_df)
print len(l_grouped_df)

#know the number os smartphones used to measure the same location
for i in range(len(l_grouped_df)):
	print len(l_grouped_df[i][1].PHONEID.unique())
	print "\n"

#print df.iloc[0][:]
