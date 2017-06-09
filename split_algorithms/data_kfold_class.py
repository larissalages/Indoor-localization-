import numpy as np

#real hit of techniques
real_hit_rate_knn = np.load("real_hit_rate_knn.npy")
real_hit_rate_rf = np.load("real_hit_rate_rf.npy")
real_hit_rate_svm = np.load("real_hit_rate_svm.npy")

file = open("data_class_svm.txt", "w")
for i in range(4):
	for j in range(10):
			print str(real_hit_rate_svm[i][j])
			file.write("%s" % str(real_hit_rate_svm[i][j]))
			file.write("%s" % '\n')

		#file.write("Purchase Amount: %s" % TotalAmount)

file.close()