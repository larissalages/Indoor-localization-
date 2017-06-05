
import pandas as pd
import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from geopy.distance import vincenty
from pyproj import Proj
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r *1000 #return in meters
#---------------------------------------------------------------------------------------------------------------      
def regression_allset(train,test,method): #Only for tests

	if(method==1):
		ml_lon = KNeighborsRegressor(n_neighbors=5, weights = 'distance')
		ml_lat = KNeighborsRegressor(n_neighbors=5, weights = 'distance')
	elif(method==2):
		ml_lon = SVR(C=1.0, epsilon=0.2,kernel='linear')
		ml_lat = SVR(C=1.0, epsilon=0.2,kernel='linear')
			
	X_train = train.ix[:,0:519]
	Y_train_lon = train['LONGITUDE']
	Y_train_lat = train['LATITUDE']

	ml_lon.fit(X_train,Y_train_lon) 
	ml_lat.fit(X_train,Y_train_lat)                                   
					
	X_test = test.ix[:,0:519] 
	Y_test_lon = test['LONGITUDE']
	Y_test_lat = test['LATITUDE']	

	#Turn into list
	predicts_lon = ml_lon.predict(X_test).tolist()
	predicts_lat = ml_lat.predict(X_test).tolist()

	Y_test_lon = Y_test_lon.values.tolist()
	Y_test_lat = Y_test_lat.values.tolist()

	error = []

	for j in range(len(X_test)):
			
		#change the latitude and longitude unit
		myProj = Proj("+proj=utm +zone=23K, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
		lon_pred,lat_pred = myProj(predicts_lon[j], predicts_lat[j], inverse=True)
		lon_Y, lat_Y = myProj(Y_test_lon[j], Y_test_lat[j], inverse=True)

		#join in a unique list
		Y = []
		Y.append(lon_Y)
		Y.append(lat_Y)
		predict = []
		predict.append(lon_pred)
		predict.append(lat_pred)			

		#The distance between the two latitudes is the error
		distance = vincenty(Y, predict).meters 
#		distance = haversine(lon_Y, lat_Y, lon_pred, lat_pred)
		#If you want to use haversine distance, uncomment the line below
		#print haversine(lon_Y, lat_Y, lon_pred, lat_pred)

		error.append(distance)	
	#print(np.mean(distance))
	
	return np.mean(error)
#--------------------------------------------------------------------------------------------------------------
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
#return the number of hits
def compare(Y_test_build, predictions_build, Y_test_floor, predictions_floor):

	hits = 0
	#if tests and predictions have the same number of building and the same number of floor, the algorithm hit
	for i in range(len(Y_test_floor)):
		if(Y_test_build[i] == predictions_build[i] and Y_test_floor[i] == predictions_floor[i]):
			hits = hits +1

	return hits

#---------------------------------------------------------------------------------------------------------------

def save_vec(hit_rate_build_mlp,hit_rate_floor_mlp,hit_rate_build_knn, hit_rate_floor_knn):

	np.save("build_mlp.npy",hit_rate_build_mlp)
	np.save("floor_mlp.npy",hit_rate_floor_mlp)

	np.save("build_knn.npy",hit_rate_build_knn)
	np.save("floor_knn.npy",hit_rate_floor_knn)
#---------------------------------------------------------------------------------------------------------------
def load_vec():
	hit_rate_build_mlp = np.load("build_mlp.npy")
	hit_rate_floor_mlp = np.load("floor_mlp.npy")

	hit_rate_build_knn = np.load("build_knn.npy")
	hit_rate_floor_knn = np.load("floor_knn.npy")

#---------------------------------------------------------------------------------------------------------------

def KFold(k, und_df_phone):

	#und_df_phone = shuffle(und_df_phone)
	phone = []
	
	#split the data frame of each smartphone
	for j in range(len(und_df_phone)): 
		phone.append(np.array_split(und_df_phone[j],k)) #the first dimension of "phone" is each phone, the second is the splits data frames from that smatphone

	#GridSearch	
	rf_building= RandomForestClassifier(n_estimators=50,n_jobs=2)
	rf_floor = RandomForestClassifier(n_estimators=50,n_jobs=2)

	#creating a empty list with size len(und_df_phone)
	hit_rate_build_rf = init_list_of_objects(len(und_df_phone))
	hit_rate_floor_rf = init_list_of_objects(len(und_df_phone))
	real_hit_rate_rf = init_list_of_objects(len(und_df_phone))
	list_predictions_build = init_list_of_objects(len(und_df_phone))
	list_predictions_floor= init_list_of_objects(len(und_df_phone))

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
		
		#Training with total training set				
		data_train = train.ix[:,0:519]
		target_train_build= train['BUILDINGID'].values
		target_train_floor= train['FLOOR'].values
		#Training		
		rf_building.fit(data_train,target_train_build)
		rf_floor.fit(data_train,target_train_floor)

		#test all phones
		for j in range(len(und_df_phone)):
			#only pick up from test set the phone that you will be evaluated
			data_test = test[j].ix[:,0:519] 
			target_test_build = test[j]['BUILDINGID'].values
			target_test_floor = test[j]['FLOOR'].values

			#classification for building 
			hit_rate_build_rf[j].append(rf_building.score(data_test , target_test_build))
			predictions_build = rf_building.predict(data_test).tolist()
			list_predictions_build[j].append(predictions_build) 

			hit_rate_floor_rf[j].append(rf_floor.score(data_test , target_test_floor))
			predictions_floor = rf_floor.predict(data_test).tolist()
			list_predictions_floor[j].append(predictions_floor)

			successful_amount = compare(target_test_build, predictions_build, target_test_floor, predictions_floor)
			real_hit_rate_rf[j].append( successful_amount/float(len(data_test)) )
			
	#saving data
	np.save("list_predictions_floor_rf.npy",list_predictions_floor)
	np.save("list_predictions_build_rf.npy",list_predictions_build)
	np.save("hit_rate_build_rf.npy",hit_rate_build_rf)
	np.save("hit_rate_floor_rf.npy",hit_rate_floor_rf)
	np.save("real_hit_rate_rf.npy",real_hit_rate_rf)

	print "hit rate building RF"
	print str(np.mean(hit_rate_build_rf[0])) + " - " +  str(np.std(hit_rate_build_rf[0]))
	print str(np.mean(hit_rate_build_rf[1])) + " - " +  str(np.std(hit_rate_build_rf[1]))
	print str(np.mean(hit_rate_build_rf[2])) + " - " +  str(np.std(hit_rate_build_rf[2]))
	print str(np.mean(hit_rate_build_rf[3])) + " - " +  str(np.std(hit_rate_build_rf[3]))
	print " "			

		
	print "hit rate floor RF"
	print str(np.mean(hit_rate_floor_rf[0])) + " - " +  str(np.std(hit_rate_floor_rf[0]))
	print str(np.mean(hit_rate_floor_rf[1])) + " - " +  str(np.std(hit_rate_floor_rf[1]))
	print str(np.mean(hit_rate_floor_rf[2])) + " - " +  str(np.std(hit_rate_floor_rf[2]))
	print str(np.mean(hit_rate_floor_rf[3])) + " - " +  str(np.std(hit_rate_floor_rf[3]))
	print " "			

	print "hit rate REAL RF"
	print str(np.mean(real_hit_rate_rf[0])) + " - " +  str(np.std(real_hit_rate_rf[0]))
	print str(np.mean(real_hit_rate_rf[1])) + " - " +  str(np.std(real_hit_rate_rf[1]))
	print str(np.mean(real_hit_rate_rf[2])) + " - " +  str(np.std(real_hit_rate_rf[2]))
	print str(np.mean(real_hit_rate_rf[3])) + " - " +  str(np.std(real_hit_rate_rf[3]))
	print " "			
"""
	print "Best Params Building"
	print rf_building.best_params_


	print "Best Params"
	print rf_floor.best_params_
"""


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
