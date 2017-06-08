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

#mean_error_knn = np.load("mean_error_knn.npy")
mean_error_rf = np.load("mean_error_RandomForest.npy")
mean_error_svr = np.load("mean_error_svr.npy")
"""
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
"""
#-------------------------------------------------------------------------------------------------
"""
media = []
for i in range(4):
	media.append( np.mean(real_hit_rate_svm[i]) )

import plotly.plotly as py
import plotly.graph_objs as go

x = ['GT-S5360', 'GT-S6500', 'HTC Wildfire', 'LT22i']
y = media
y = [i * 100 for i in y]
y = [ '%.2f' % i for i in y ]
y = [i+'%' for i in y]

data = [go.Bar(
            x=x,y=y,
            marker=dict(
        	color=['rgba(222,45,38,0.8)', 'rgb(136, 222, 254)','rgb(0, 216, 173)', 'rgb(255, 245, 173)'],
                line=dict(
                    color='rgb(0, 2, 0)',
                    width=1.5),
            ),
            opacity=0.6
        )]

layout = go.Layout(
	title='Taxas de acerto - Fase 2 - SVM',
	yaxis=dict(
		range=[80, 100]
    ),
    annotations=[
        dict(x=xi,y=yi,
             text=str(yi),
             xanchor='center',
             yanchor='bottom',
             showarrow=False,
        ) for xi, yi in zip(x, y)]
)


fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='bar-direct-labels')
"""
#--------------------------------------------------------------------------------------------------------------
import plotly.plotly as py
import plotly.graph_objs as go


#-----------------------------------------------------------------------

error= np.load("error_rf.npy")
x = error[3]

trace1 = go.Histogram(
    x=x,
    xbins=dict(
        start=0,
        end=80,
        size=3
    ),
    marker=dict(
        color='#A3FF99',
            line=dict(
                	color='rgb(0, 2, 0)',
                	width=1.5),
    ),

    opacity=0.75
)
data = [trace1]
layout = go.Layout(
    title='Erro Random Forest- LT22i',
    xaxis=dict(
        autotick=False,
        ticks='outside',
        tick0=0,
        dtick=3,
        title='Erro (m)',
        range=[0, 50]

    ),
    yaxis=dict(

        title='Quantidade de amostras',
        range=[0, 120]
    ),

)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='rf_histogram_cel4')


"""
data = [go.Histogram(x=x)]

py.plot(data, filename='basic histogram')
"""