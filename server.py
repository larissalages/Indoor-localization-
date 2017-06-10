
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
		ml_lon = SVR(C=1.0, epsilon=0.2)
		ml_lat = SVR(C=1.0, epsilon=0.2)
	elif(method==3):
		#machine_learn = MLPClassifier(solver='sgd',learning_rate = 'adaptive',verbose='true',activation='tanh',alpha=1e-5)		
		#machine_learn = MLPClassifier(solver='sgd',learning_rate = 'adaptive',verbose='false',activation='tanh',alpha=1e-5,max_iter=400) #THE BEST
		#machine_learn = MLPClassifier(hidden_layer_sizes=(100,5), solver='sgd',learning_rate = 'adaptive',verbose='true',activation='tanh',alpha=1e-5,max_iter=500)
		model = MLPClassifier(learning_rate = 'adaptive',activation='logistic')
		solvers = ['lbfgs', 'sgd', 'adam']
		random_states = [0,2,4,6,8,9]
		max_its = [200,400,600]
		ml_lon = GridSearchCV(estimator=model, param_grid=dict(max_iter=max_its,solver=solvers,random_state=random_states)) #GRID
		ml_lat = GridSearchCV(estimator=model, param_grid=dict(max_iter=max_its,solver=solvers,random_state=random_states),n_jobs=7) #GRID					

	X_train = train.ix[:,0:519]
	Y_train_lon = np.asarray(train['LONGITUDE'], dtype="|S6")
	Y_train_lat = np.asarray(train['LATITUDE'], dtype="|S6")

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
#reorder the list
def put_list(pred_old, index, pred_new):

	for i in range(len(index)):
		pred_new[index[i]] = pred_old[i]

	return pred_new	

#---------------------------------------------------------------------------------------------------------------
def floor_classifier(predictions,train,test,method):
		
	successful_amount = 0
	pred_floor_ordered = init_list_of_objects(len(predictions))

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
			pred_floor_ordered = put_list(predictions_floor, indexes, pred_floor_ordered)

			#Accumulate the number of hits 
			successful_amount = compare(Y_test_build.tolist(), predictions[indexes].tolist(), Y_test_floor.tolist(), predictions_floor.tolist()) + successful_amount
	
	return successful_amount/float(len(test)), pred_floor_ordered		

#---------------------------------------------------------------------------------------------------------------
def coord_regression(predictions_b,predictions,train,test,method):
		
	mean_error = []

	if(method==1):
		machine_learn = KNeighborsRegressor(n_neighbors=5, weights = 'distance')
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
	for j in range(3):
		new_train1 = train.loc[train['BUILDINGID'] == j] #select for training only buildings with that label (0,1, or 2)
		ind = [x for x in range(len(predictions_b)) if predictions_b[x]==j] #get the position of the samples that have building == i	
		new_test1 = test.iloc[ind,:]

		if(ind):
		#for each floor
			for i in range(5):
				
				new_train2 = new_train1.loc[new_train1['FLOOR'] == i]
				if(not new_train2.empty): 
					indexes = [x for x in range(len(predictions)) if (predictions[x]==i and predictions_b[x]==j)] #get the position of the samples that have building == i
				else:
					index = []

				if (indexes): #if list is not empty

					X_train = new_train2.ix[:,0:519]
					Y_train = new_train2[['LONGITUDE','LATITUDE']]
					machine_learn.fit(X_train,Y_train)                                   
					
					#testing samples with prediction building == i
					new_test2 = test.iloc[indexes,:]
					X_test = new_test2.ix[:,0:519] 
					Y_test = new_test2[['LONGITUDE','LATITUDE']]

					#Turn into list
					predicts_lon_lat = machine_learn.predict(X_test).tolist()
					Y_test = Y_test.values.tolist()

					distance = []
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
						distance.append(vincenty(Y, predict).meters)
						print "distance"
						print distance
						#If you want to use haversine distance, uncomment the line below
						#print haversine(lon_Y, lat_Y, lon_pred, lat_pred)

					mean_error.append(np.mean(distance))	
					#print(np.mean(distance))
		
	return np.mean(mean_error)
#---------------------------------------------------------------------------------------------------------------

def regression_subset(predictions,train,test,method): 
	
	
	mean_error = []
	if(method==1):
		machine_learn = KNeighborsRegressor(n_neighbors=5, weights = 'distance')
	elif(method==2):
		machine_learn = MLPRegressor(random_state=0) 
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
	mlp = MLPClassifier(solver='sgd',learning_rate = 'invscaling',verbose='true',activation='tanh')
	

	#creating a empty list with size len(und_df_phone)
	hit_rate_build_knn = init_list_of_objects(len(und_df_phone)) 
	hit_rate_floor_knn = init_list_of_objects(len(und_df_phone))
	hit_rate_build_mlp = init_list_of_objects(len(und_df_phone))
	hit_rate_floor_mlp = init_list_of_objects(len(und_df_phone)) 
	mean_error_knn = init_list_of_objects(len(und_df_phone))
	mean_error_mlp = init_list_of_objects(len(und_df_phone))
	mean_error_svr = init_list_of_objects(len(und_df_phone))

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
		target_train_build= train['BUILDINGID']

		#knn - training		
		knn.fit(data_train,target_train_build)
#		mlp.fit(data_train, target_train)

		#test all phones
		for j in range(len(und_df_phone)):
			#only pick up from test set the phone that you will be evaluated
			data_test = test[j].ix[:,0:519] 
			target_test_build = test[j]['BUILDINGID']

			#predictions and scores for each smartphone (predictions about building)
#			predictions_knn = knn.predict(data_test)
#			predictions_mlp = mlp.predict(data_test)
			#classification for building 
#			hit_rate_build_knn[j].append(knn.score(data_test , target_test_build))
#			hit_rate_build_mlp[j].append(mlp.score(data_test , target_test)) 
			#classification for floor
#			hit_rate, predictions_floor_knn = floor_classifier(predictions_knn,train,test[j],1)
#			hit_rate_floor_knn[j].append( hit_rate )
#			hit_rate_floor_mlp[j].append( floor_classifier(predictions_mlp,train,test[j],2) )
			#regression to found latitude and longitude
			#mean_error_knn[j].append(coord_regression(predictions_knn,predictions_floor_knn,train,test[j],1))
#			mean_error_knn[j].append(regression_subset(predictions_knn,train,test[j],1))
#			mean_error_knn[j].append( regression_allset(train,test[j],1) )
			mean_error_mlp[j].append(regression_allset(train,test[j],3))
#			mean_error_svr[j].append( regression_allset(train,test[j],2) )
			predictions_knn = []

	np.save("error_mlp.npy",mean_error_mlp)
	print "mean error regression MLP"
	print str(np.mean(mean_error_mlp[0])) + " - " + str(np.std(mean_error_mlp[0]))
	print str(np.mean(mean_error_mlp[1])) + " - " + str(np.std(mean_error_mlp[1]))
	print str(np.mean(mean_error_mlp[2])) + " - " + str(np.std(mean_error_mlp[2]))
	print str(np.mean(mean_error_mlp[3])) + " - " + str(np.std(mean_error_mlp[3]))


"""
	print "mean error regression knn"
	print str(np.mean(mean_error_knn[0])) + " - " +  str(np.std(mean_error_knn[0]))
	print str(np.mean(mean_error_knn[1])) + " - " +  str(np.std(mean_error_knn[1]))
	print str(np.mean(mean_error_knn[2])) + " - " +  str(np.std(mean_error_knn[2]))
	print str(np.mean(mean_error_knn[3])) + " - " +  str(np.std(mean_error_knn[3]))
	print " "
"""
"""
	print "mean error regression SVR"
	print str(np.mean(mean_error_svr[0])) + " - " +  str(np.std(mean_error_svr[0]))
	print str(np.mean(mean_error_svr[1])) + " - " +  str(np.std(mean_error_svr[1]))
	print str(np.mean(mean_error_svr[2])) + " - " +  str(np.std(mean_error_svr[2]))
	print str(np.mean(mean_error_svr[3])) + " - " +  str(np.std(mean_error_svr[3]))
	print " "			
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