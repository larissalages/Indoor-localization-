#Calculate how many measurements each cell phone has.
import pandas as pd
import csv

#convert csv file in an data frame
df = pd.read_csv('trainingData.csv')

grouped_df = list(df.groupby(['PHONEID']))

for i in range(len(grouped_df)):
	print "Measures:" + str(len(grouped_df[i][1])) + ", PHONEID" + str((grouped_df[i][1]).PHONEID.unique())
	print "\n"