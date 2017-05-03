
import pandas as pd
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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
def shuffle(und_df_phone):

	for i in range(len(und_df_phone)):
		und_df_phone[i] = und_df_phone[i].sample(frac=1)

	return und_df_phone	 

#---------------------------------------------------------------------------------------------------------------
def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

#---------------------------------------------------------------------------------------------------------------
def floor_classifier(predictions,train,test): #TODO: TESTAR
	
	scores_floor = []

	knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance')

	#for each building
	for i in range(3):
		
		new_train = train.loc[train['BUILDINGID'] == i] #select for training only buildings with that label (0,1, or 2)
		indexes = [x for x in range(len(predictions)) if predictions[x]==i] #get the position of the samples that have building == i
		
		if (indexes): #if list is not empty
			#training, samples with building == i 
			X_train = new_train.ix[:,0:519]
			Y_train = new_train['FLOOR']
			knn.fit(X_train,Y_train)
			
			#testing samples with prediction building == i
			new_test = test.iloc[indexes,:]
			X_test = new_test.ix[:,0:519] 
			Y_test = new_test['FLOOR']

			
			scores_floor.append(knn.score(X_test , Y_test))
	 
	
	return np.mean(scores_floor)		

#---------------------------------------------------------------------------------------------------------------
def KFold(k, und_df_phone):

	und_df_phone = shuffle(und_df_phone)
	phone = []
	
	#split the data frame of each smartphone
	for j in range(len(und_df_phone)): 
		phone.append(np.array_split(und_df_phone[j],k)) #the first dimension of phone is each phone, the second is the splits data frames from that smatphone

	knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance')

	hit_rate_build = init_list_of_objects(len(und_df_phone)) #creating a empty list with size len(und_df_phone)
	hit_rate_floor = init_list_of_objects(len(und_df_phone)) #creating a empty list with size len(und_df_phone)

	for i in range(k):
		#separate each smartphone's data frame in test and train
		test = [] #list of data frames
		train =pd.DataFrame()		
		for j in range(len(und_df_phone)):
			test.append(phone[j][i])
			#Join the train set
			for x in range(k):
				if x != i:
					train = pd.concat([train,phone[j][x]])	
		
		#Training KNN, with total training set				
		data_train = train.ix[:,0:519]
		target_train = train['BUILDINGID']
		#knn - training		
		knn.fit(data_train,target_train)
		
		#test all phones
		for j in range(len(und_df_phone)):
			#only pick up from test set the phone that you will be evaluated
			data_test = test[j].ix[:,0:519] 
			target_test = test[j]['BUILDINGID']	

			#predictions and scores for each smartphone
			predictions = knn.predict(data_test)
			#classification for building 
			hit_rate_build[j].append(knn.score(data_test , target_test)) 
			#classification for floor
			hit_rate_floor[j].append( floor_classifier(predictions,train,test[j]) )
			predictions = []  

			
	print np.mean(hit_rate_floor[0])
	print np.mean(hit_rate_floor[1])
	print np.mean(hit_rate_floor[2])
	print np.mean(hit_rate_floor[3])	



#---------------------------------------------------------------------------------------------------------------
def main():

	#defines
	phones_used = [6,7,13,14]
	k=10

	#convert csv file in an data frame
	df = pd.read_csv('trainingData.csv')

	#group by pohneID
	grouped_df = list(df.groupby(['PHONEID']))

	#show_number_measurements(grouped_df)

	#create a data frame for each phone
	df_phone, list_phones = create_phone_df(df,grouped_df)
	
	#Doing undersampling
	und_df_phone = undersampling(df_phone,phones_used)

	KFold(k, und_df_phone)



if __name__ == "__main__":
    main()


#to do a list of data frames. Each smartphone has its own data frame
