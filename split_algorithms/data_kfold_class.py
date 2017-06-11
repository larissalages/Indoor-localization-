import numpy as np

#real hit of techniques
mean_error_knn = np.load("mean_error_knn.npy")
mean_error_rf = np.load("mean_error_RandomForest.npy")
mean_error_svr = np.load("mean_error_svr.npy")

def write_in_file(method, file):

	for i in range(4):
		for j in range(10):
			file.write("%s" % str(method[i][j]))
			file.write("%s" % '\n')

	return file

def main():
	file_knn = open("data_reg_knn.txt", "w")
	file_rf = open("data_reg_rf.txt", "w")
	file_svm = open("data_reg_svm.txt", "w")

	file_knn = write_in_file(mean_error_knn, file_knn)
	file_rf = write_in_file(mean_error_rf, file_rf)
	file_svm = write_in_file(mean_error_svr, file_svm)



	file_knn.close()
	file_rf.close()
	file_svm.close()

if __name__ == "__main__":
    main()

