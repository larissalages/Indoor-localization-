import numpy as np

mean_error_knn = np.load("mean_error_knn.npy")
mean_error_RandomForest = np.load("mean_error_RandomForest.npy")
mean_error_svr = np.load("mean_error_svr.npy")

def write_in_file(celular_id, file):

	for j in range(10): #for each fold
		file.write("%s" % str(mean_error_knn[celular_id][j]))
		file.write("%s" % '\n')
		file.write("%s" % str(mean_error_RandomForest[celular_id][j]))
		file.write("%s" % '\n')
		file.write("%s" % str(mean_error_svr[celular_id][j]))
		file.write("%s" % '\n')

	return file	

def main():

	file0 = open("data_reg_cel0.txt", "w")
	file1 = open("data_reg_cel1.txt", "w")
	file2 = open("data_reg_cel2.txt", "w")
	file3 = open("data_reg_cel3.txt", "w")

	write_in_file(0, file0)
	write_in_file(1, file1)
	write_in_file(2, file2)
	write_in_file(3, file3)

	file0.close()
	file1.close()
	file2.close()
	file3.close()

if __name__ == "__main__":
    main()
