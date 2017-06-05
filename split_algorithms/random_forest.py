
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
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor

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
def regression_allset(Y_test_lon,Y_test_lat,X_test,ml_lon,ml_lat): #Only for tests
                              					
	#Turn into list
	predicts_lon = ml_lon.predict(X_test)
	predicts_lat = ml_lat.predict(X_test)



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

		#If you want to use haversine distance, uncomment the line below
#		distance = haversine(lon_Y, lat_Y, lon_pred, lat_pred)		


		error.append(distance)	

	
	return np.mean(error),predicts_lon,predicts_lat
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

	lon = RandomForestRegressor(n_estimators=500,n_jobs=3)
	lat = RandomForestRegressor(n_estimators=500,n_jobs=3)	

	#creating a empty list with size len(und_df_phone)
	mean_error = init_list_of_objects(len(und_df_phone))
	list_pred_lon = init_list_of_objects(len(und_df_phone))
	list_pred_lat = init_list_of_objects(len(und_df_phone))

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
		X_train = train.ix[:,0:519].fillna(0).as_matrix().astype(float)
		Y_train_lon = train['LONGITUDE'].fillna(0).as_matrix().astype(float)
		Y_train_lat = train['LATITUDE'].fillna(0).as_matrix().astype(float)


		lon.fit(X_train,Y_train_lon) 
		lat.fit(X_train,Y_train_lat) 



		#test all phones
		for j in range(len(und_df_phone)):
			#only pick up from test set the phone that you will be evaluated
			data_test = test[j].ix[:,0:519].fillna(0).as_matrix().astype(float)
			Y_test_lon = test[j]['LONGITUDE'].fillna(0).as_matrix().astype(float)
			Y_test_lat = test[j]['LATITUDE'].fillna(0).as_matrix().astype(float)

			error, pred_lon, pred_lat = regression_allset(Y_test_lon,Y_test_lat,data_test,lon,lat)
			mean_error[j].append( error )
			list_pred_lon[j].append(pred_lon)
			list_pred_lat[j].append(pred_lat)

	#save data
	np.save("pred_lon.npy", list_pred_lon)
	np.save("pred_lat.npy", list_pred_lat)			
	np.save("mean_error_RandomForest.npy", mean_error)

	print "mean error regression RandomForest"
	print str(np.mean(mean_error[0])) + " - " +  str(np.std(mean_error[0]))
	print str(np.mean(mean_error[1])) + " - " +  str(np.std(mean_error[1]))
	print str(np.mean(mean_error[2])) + " - " +  str(np.std(mean_error[2]))
	print str(np.mean(mean_error[3])) + " - " +  str(np.std(mean_error[3]))
	print " "		
"""
	print "Best Params Lon"
	print lon.best_params_
	print "Best Params Lat"
	print lat.best_params_
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
