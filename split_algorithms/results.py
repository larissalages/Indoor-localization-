import numpy as np

#real hit of techniques
real_hit_rate_knn = np.load("real_hit_rate_knn.npy")
real_hit_rate_rf = np.load("real_hit_rate_rf.npy")
real_hit_rate_svm = np.load("real_hit_rate_svm.npy")

print "hit rate REAL KNN"
print str(np.mean(real_hit_rate_knn[0])) + " - " +  str(np.std(real_hit_rate_knn[0]))
print str(np.mean(real_hit_rate_knn[1])) + " - " +  str(np.std(real_hit_rate_knn[1]))
print str(np.mean(real_hit_rate_knn[2])) + " - " +  str(np.std(real_hit_rate_knn[2]))
print str(np.mean(real_hit_rate_knn[3])) + " - " +  str(np.std(real_hit_rate_knn[3]))
print " "		

print "hit rate REAL RF"
print str(np.mean(real_hit_rate_rf[0])) + " - " +  str(np.std(real_hit_rate_rf[0]))
print str(np.mean(real_hit_rate_rf[1])) + " - " +  str(np.std(real_hit_rate_rf[1]))
print str(np.mean(real_hit_rate_rf[2])) + " - " +  str(np.std(real_hit_rate_rf[2]))
print str(np.mean(real_hit_rate_rf[3])) + " - " +  str(np.std(real_hit_rate_rf[3]))
print " "	

print "hit rate REAL SVM"
print str(np.mean(real_hit_rate_svm[0])) + " - " +  str(np.std(real_hit_rate_svm[0]))
print str(np.mean(real_hit_rate_svm[1])) + " - " +  str(np.std(real_hit_rate_svm[1]))
print str(np.mean(real_hit_rate_svm[2])) + " - " +  str(np.std(real_hit_rate_svm[2]))
print str(np.mean(real_hit_rate_svm[3])) + " - " +  str(np.std(real_hit_rate_svm[3]))
print " "
print " "	
print " "		

#-------------------------------------------------------------------------------------------------
#Mean Error

mean_error_knn = np.load("mean_error_knn.npy")
mean_error_rf = np.load("mean_error_RandomForest.npy")
mean_error_svr = np.load("mean_error_svr.npy")

print "mean error regression knn"
print str(np.mean(mean_error_knn[0])) + " - " +  str(np.std(mean_error_knn[0]))
print str(np.mean(mean_error_knn[1])) + " - " +  str(np.std(mean_error_knn[1]))
print str(np.mean(mean_error_knn[2])) + " - " +  str(np.std(mean_error_knn[2]))
print str(np.mean(mean_error_knn[3])) + " - " +  str(np.std(mean_error_knn[3]))
print " "

print "mean error regression RandomForest"
print str(np.mean(mean_error_rf[0])) + " - " +  str(np.std(mean_error_rf[0]))
print str(np.mean(mean_error_rf[1])) + " - " +  str(np.std(mean_error_rf[1]))
print str(np.mean(mean_error_rf[2])) + " - " +  str(np.std(mean_error_rf[2]))
print str(np.mean(mean_error_rf[3])) + " - " +  str(np.std(mean_error_rf[3]))
print " "	

print "mean error regression SVR"
print str(np.mean(mean_error_svr[0])) + " - " +  str(np.std(mean_error_svr[0]))
print str(np.mean(mean_error_svr[1])) + " - " +  str(np.std(mean_error_svr[1]))
print str(np.mean(mean_error_svr[2])) + " - " +  str(np.std(mean_error_svr[2]))
print str(np.mean(mean_error_svr[3])) + " - " +  str(np.std(mean_error_svr[3]))
print " "
#-------------------------------------------------------------------------------------------------
media = []
for i in range(4):
	media.append( np.mean(real_hit_rate_knn[i]) )

import plotly.plotly as py
import plotly.graph_objs as go

data = [go.Bar(
            x=['A', 'B', 'C', 'D'],
            y= media
    )]

py.plot(data, filename='basic-bar')