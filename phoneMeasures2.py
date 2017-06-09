
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
def regression_allset(X_test,Y_test,machine_learn): #Only for tests

	#Turn into list
	predicts_lon_lat = machine_learn.predict(X_test).tolist()
	Y_test = Y_test.values.tolist()

	mean_error = []

	for j in range(len(predicts_lon_lat)):
			
		#change the latitude and longitude unit
		myProj = Proj("+proj=utm +zone=23K, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
		lon_pred,lat_pred = myProj(predicts_lon_lat[j][0], predicts_lon_lat[j][1], inverse=True)
		lon_Y, lat_Y = myProj(Y_test[j][0], Y_test[j][1], inverse=True)

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
		#print haversine(lon_Y, lat_Y, lon_pred, lat_pred)
		print "distance"
		print distance
		mean_error.append(distance)	
	#print(np.mean(distance))
	1/0	
	return mean_error
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
def floor_classifier(predictions,train,test,method):
		
	successful_amount = 0

	if(method==1):
		machine_learn = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
	elif(method==2):
		#machine_learn = MLPClassifier(solver='sgd',learning_rate = 'adaptive',verbose='true',activation='tanh',alpha=1e-5)		
		machine_learn = MLPClassifier(solver='sgd',learning_rate = 'adaptive',verbose='false',activation='tanh',alpha=1e-5,max_iter=400) #THE BEST
		#machine_learn = MLPClassifier(hidden_layer_sizes=(100,5), solver='sgd',learning_rate = 'adaptive',verbose='true',activation='tanh',alpha=1e-5,max_iter=500)
		#model = MLPClassifier(learning_rate = 'adaptive')
		#solvers = ['lbfgs', 'sgd', 'adam']
		#activations = ['identity', 'logistic', 'tanh', 'relu']
		#max_its = [200,400,600]
		#machine_learn = GridSearchCV(estimator=model, param_grid=dict(activation =activations,max_iter=max_its),n_jobs=7) #GRID


	#for each building
	for i in range(3):
		
		new_train = train.loc[train['BUILDINGID'] == i] #select for training only buildings with that label (0,1, or 2)
		indexes = [x for x in range(len(predictions)) if predictions[x]==i] #get the position of the samples that have building == i
		
		if (indexes): #if list is not empty
			#training, samples with building == i 
			X_train = new_train.ix[:,0:519]
			Y_train = new_train['FLOOR']
			machine_learn.fit(X_train,Y_train)                                   
			
			#testing samples w ith prediction building == i
			new_test = test.iloc[indexes,:]
			X_test = new_test.ix[:,0:519] 

			Y_test_floor = new_test['FLOOR']
			Y_test_build = new_test['BUILDINGID']
			#if(method ==2):
				#print "best score:"
				#print machine_learn.best_score_
			predictions_floor = machine_learn.predict(X_test)
			#Accumulate the number of hits 
			successful_amount = compare(Y_test_build.tolist(), predictions[indexes].tolist(), Y_test_floor.tolist(), predictions_floor.tolist()) + successful_amount
	
	return successful_amount/float(len(test))		

#---------------------------------------------------------------------------------------------------------------

def regression_subset(predictions,train,test,method): 
	
	
	mean_error = []
	if(method==1):
		machine_learn = KNeighborsRegressor(n_neighbors=5, weights = 'distance')
	elif(method==2):
		machine_learn = MLPRegressor(random_state=0) #THE BEST
	#for each building
	for i in range(3):
		
		new_train = train.loc[train['BUILDINGID'] == i] #select for training only buildings with that label (0,1, or 2)
		indexes = [x for x in range(len(predictions)) if predictions[x]==i] #get the position of the samples that have building == i

		
		if (indexes): #if list is not empty
			#training, samples with building == i 
			X_train = new_train.ix[:,0:519]
			Y_train = new_train[['LONGITUDE','LATITUDE']]
			machine_learn.fit(X_train,Y_train)
		
			#testing samples with prediction building == i
			new_test = test.iloc[indexes,:]
			X_test = new_test.ix[:,0:519] 
			Y_test = new_test[['LONGITUDE','LATITUDE']]

			#Turn into list
			predicts_lon_lat = machine_learn.predict(X_test).tolist()
			Y_test = Y_test.values.tolist()

			distance = []
			for j in range(len(predicts_lon_lat)):
				print "lon"
				print predicts_lon_lat[j][0]
				print "lat"
				print predicts_lon_lat[j][1]				
				#change the latitude and longitude unit
				myProj = Proj("+proj=utm +zone=23K, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
				lon_pred,lat_pred = myProj(predicts_lon_lat[j][0], predicts_lon_lat[j][1], inverse=True)
				lon_Y, lat_Y = myProj(Y_test[j][0], Y_test[j][1], inverse=True)
			
				#join in a unique list
				Y = []
				Y.append(lon_Y)
				Y.append(lat_Y)
				predict = []
				predict.append(lon_pred)
				predict.append(lat_pred)			

				#The distance between the two latitudes is the error
				distance.append(vincenty(Y, predict).meters)

				#If you want to use haversine distance, uncomment the line below
				#print haversine(lon_Y, lat_Y, lon_pred, lat_pred)

			mean_error.append(np.mean(distance))	
			#print(np.mean(distance))
	1/0	
	return np.mean(mean_error)

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

	knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
	knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
	mlp = MLPClassifier(solver='sgd',learning_rate = 'invscaling',verbose='true',activation='tanh')
#	mlp_reg = MLPRegressor(solver='sgd',learning_rate = 'invscaling',verbose='true',activation='tanh')
	model = MLPRegressor(learning_rate = 'adaptive')
	solvers = ['lbfgs', 'sgd', 'adam']
	activations = ['identity', 'logistic', 'tanh', 'relu']
	max_its = [200,400,600]
	mlp_reg = GridSearchCV(estimator=model, param_grid=dict(activation =activations,max_iter=max_its),n_jobs=7) #GRID
 
	#knn_reg = KNeighborsRegressor(n_neighbors=5, weights = 'distance')

	#creating a empty list with size len(und_df_phone)
	hit_rate_build_knn = init_list_of_objects(len(und_df_phone)) 
	hit_rate_floor_knn = init_list_of_objects(len(und_df_phone))
	hit_rate_build_mlp = init_list_of_objects(len(und_df_phone))
	hit_rate_floor_mlp = init_list_of_objects(len(und_df_phone)) 
	mean_error_knn = init_list_of_objects(len(und_df_phone))
	mean_error_mlp = init_list_of_objects(len(und_df_phone))

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
		target_train_reg = train[['LONGITUDE','LATITUDE']]

		#knn - training		
#		knn.fit(data_train,target_train)
#		knn_reg.fit(data_train,target_train_reg)
#		mlp.fit(data_train, target_train)
		mlp_reg.fit(data_train,target_train_reg)
		#test all phones
		for j in range(len(und_df_phone)):
			#only pick up from test set the phone that you will be evaluated
			data_test = test[j].ix[:,0:519] 
			target_test = test[j]['BUILDINGID']	
			target_test_reg = test[j][['LONGITUDE','LATITUDE']]
			#predictions and scores for each smartphone (predictions about building)
#			predictions_knn = knn.predict(data_test)
#			predictions_mlp = mlp.predict(data_test)
			#classification for building 
#			hit_rate_build_knn[j].append(knn.score(data_test , target_test))
#			hit_rate_build_mlp[j].append(mlp.score(data_test , target_test)) 
			#classification for floor
#			hit_rate_floor_knn[j].append( floor_classifier(predictions_knn,train,test[j],1) )
#			hit_rate_floor_mlp[j].append( floor_classifier(predictions_mlp,train,test[j],2) )
			#regression to found latitude and longitude
#			mean_error_knn[j].append(regression_subset(predictions_knn,train,test[j],1))
#			mean_error_knn[j].append(regression_allset(data_test,target_test_reg,knn_reg))
			mean_error_mlp[j].append(regression_allset(data_test,target_test_reg,mlp_reg))
			predictions_knn = []

	print "mean error regression MLP"
	print str(np.mean(mean_error_mlp[0])) + " - " + str(np.std(mean_error_mlp[0]))
	print str(np.mean(mean_error_mlp[1])) + " - " + str(np.std(mean_error_mlp[1]))
	print str(np.mean(mean_error_mlp[2])) + " - " + str(np.std(mean_error_mlp[2]))
	print str(np.mean(mean_error_mlp[3])) + " - " + str(np.std(mean_error_mlp[3]))

#---------------------------------------------------------------------------------------------------------------
def main():

	#defines
	phones_used = [6,7,13,14]
	k=10

	#convert csv file in an data frame
	df = pd.read_csv('trainingData.csv')

	#group by pohneID
	grouped_df = list(df.groupby(['PHONEID']))

	show_number_measurements(grouped_df)
	1/0
	#create a data frame for each phone
	df_phone, list_phones = create_phone_df(df,grouped_df)
	
	#Doing undersampling
	und_df_phone = undersampling(df_phone,phones_used)

	KFold(k, und_df_phone)



if __name__ == "__main__":
    main()


#to do a list of data frames. Each smartphone has its own data frame
