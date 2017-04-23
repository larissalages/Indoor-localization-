
import pandas as pd
import csv

#Calculate how many measurements each cell phone has
def show_number_measurements(grouped_df):
	for i in range(len(grouped_df)):
		print "Measures:" + str(len(grouped_df[i][1])) + ", PHONEID" + str((grouped_df[i][1]).PHONEID.unique())
	print "\n"
#---------------------------------------------------------------------------------------------------------------

#Create a list of data frames. Each smartphone has its own data frame
def create_phone_df(df,grouped_df):
	list_phones = df.PHONEID.unique()
	df_phone = []

	j=0
	for i in range(0,24):
		if (i in list_phones):
			df_phone.append(grouped_df[j][1])
			j=j+1
		else:
			df_phone.append([])

	return df_phone, list_phones
#---------------------------------------------------------------------------------------------------------------

def undersampling(df_phone, phones_used):

	minimum = 10000000
	und_df_phone = []

	for i in phones_used:
		
		#find the smaller data frame
		if(len(df_phone[i]) < minimum):
			minimum = len(df_phone[i])
			ind_min = i

	#unsampling the others data frames so they are the same size		
	for i in phones_used:
		if(i != ind_min):
			und_df_phone.append(df_phone[i].sample(n=minimum))
		else:
			und_df_phone.append(df_phone[i])	

	return und_df_phone	

#---------------------------------------------------------------------------------------------------------------
def main():

	phones_used = [6,7,13,14]

	#convert csv file in an data frame
	df = pd.read_csv('trainingData.csv')

	#group by pohneID
	grouped_df = list(df.groupby(['PHONEID']))

	show_number_measurements(grouped_df)

	#create a data frame for each phone
	df_phone, list_phones = create_phone_df(df,grouped_df)
	

	und_df_phone = undersampling(df_phone,phones_used)


if __name__ == "__main__":
    main()


#to do a list of data frames. Each smartphone has its own data frame
