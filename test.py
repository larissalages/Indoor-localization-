import pandas as pd
import numpy as np


x = [[1,2,3,6],[1,4,3,6],[1,2,3,6],[3,7,9,2],[3,5,8,10],[3,7,8,55]]
vetor = np.array(x)


df3 = pd.DataFrame(vetor,columns=['a', 'b', 'c', 'd'])

grouped_df = df3.groupby(['a','c'])
l_df = list(grouped_df)

print l_df
print "\n"
print "teste"

#print len(l_df)

for i in range(len(l_df)):
	print len(l_df[i][1].d.unique())
	print "\n"