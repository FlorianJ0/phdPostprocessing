#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:59:47 2017

@author: florian
"""

import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import matplotlib.cm as cm
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
import traces

localData = np.load("localdata.npy")
globalData = np.load("globaldata.npy")

g = 0
for i in xrange(globalData[:, 1].shape[0]):
    # print globalData[138-i,-1]
    if globalData[138 - i, -1] != 'nan':
        g = globalData[138 - i, -1]

    else:
        globalData[138 - i, -1] = g

# compute sorting index
ind = np.lexsort((globalData[:, 2].astype(float), globalData[:, -1].astype(float)))
indLoc = np.ones(localData.shape[0])*np.nan
index = np.indices([600])
for i in xrange(globalData.shape[0]):
    indLoc[(i) * 600: (i + 1) * 600 ] = (ind[i]*600)+index

globalData = globalData[ind]
indLoc = indLoc.astype(int)
localData = localData[indLoc]

globalVarNames = ['lastScan', 'patientId', 'scanId', 'ptName', 'AAA', 'AGE', 'SEXE', 'IMC', \
                  'Psys', 'Pdias', 'HTA', 'nRxantiHTA', 'DLP', 'STATINES', 'volLum', 'voTH', \
                  'vTot', 'vTot_monthly', 'vTot_monthly_2percentthreshold', 'areaLum', \
                  'shapeLum', 'dmaxLum', 'd0Lum', 'davgLum', 'dmaxTH', 'dmaxTH_50mmthreshold', \
                  'd0TH', 'davgTH', 'volLum_monthly', 'areaLum_monthly', 'shapeLum_monthly' \
    , 'dmaxLum_monthly', 'dmaxTH_monthly', 'dmaxTH_monthly5mmthreshold', \
                  'volTH_monthly', 'tortuosity', 'tortLum_monthly', 'curvature', \
                  'curvLum_monthly', 'Divergence_average_max', 'Gradients_average_max', \
                  'OSI_max', 'PatchArea_max', 'RRT_max', 'TAWSS_max', 'Thrombus_thickness_max', \
                  'ECAP_max', 'Divergence_average_min', 'Gradients_average_min', \
                  'OSI_min', 'PatchArea_min', 'RRT_min', 'TAWSS_min', 'Thrombus_thickness_min', \
                  'ECAP_min', 'Divergence_average_mean', 'Gradients_average_mean', \
                  'OSI_mean', 'PatchArea_mean', 'RRT_mean', 'TAWSS_mean', 'Thrombus_thickness_mean', \
                  'ECAP_mean', 'Divergence_average_std', 'Gradients_average_std', 'OSI_std', \
                  'PatchArea_std', 'RRT_std', 'TAWSS_std', 'Thrombus_thickness_std', 'ECAP_std', \
                  'localLumAreaVarMean', 'localLumAreaVarMax', \
                  ' localLumAreaVarMin', 'localThThVarMean', 'localThThVarMax', 'localThThVarMin', \
                  ' localDivAvgVarMean', 'localDivAvgVarMax', 'localDivAvgVarMin', 'localGradAvgVarMean', \
                  'localGradAvgVarMax', 'localGradAvgVarMin', ' localOSIVarMean', 'localOSIVarMax', \
                  'localOSIVarMin', 'localRRTVarMean', 'localRRTVarMax', 'localRRTVarMin', \
                  'localTAWSSVarMean', ' localTAWSSVarMax', 'localTAWSSVarMin', 'localECAPVarMean', \
                  'localECAPVarMax', 'localECAPVarMin', 'thrombusCoverage', \
                  'thrombusCoverageVar', 'dt', 'CummulativeRisk', 'dMaxGrowthRegression']

localVarNames = ['AbscissaMetric', 'AngularMetric', 'BoundaryMetric', 'ClippingArray', \
                 'DistanceToCenterlines', 'Divergence_average', 'Gradients_average', \
                 'GroupIds', 'HarmonicMapping', 'OSI', 'PatchArea', 'RRT', 'Sector', \
                 'Slab', 'StretchedMapping', 'TAWSS', 'th_thickness', 'localLumAreaVar', \
                 'localThThVar', 'localDivAvgVar', 'localGradAvgVar', 'localOSIVar', \
                 'localRRTVar', 'localTAWSSVar', 'localECAPVar', 'ECAP', 'localDistToCtrl', \
                 'localDistToCtrlVar', 'thrombusGrowthThresh', 'stretchGrowthThresh', 'disToCtrlGrowthThresh']

AAAvsnoAAA = 0
riskvsnorisk = 1

idDmaxth = globalVarNames.index('dmaxTH_50mmthreshold')
idVolmax = globalVarNames.index('vTot_monthly_2percentthreshold')
CummulativeRisk = globalVarNames.index('CummulativeRisk')

localDataAAA = localData[:-5400, :]
localDatanoAAA = localData[-5399:, :]

globalDataAAA = globalData[:-10, :]
globalDatanoAAA = globalData[-9:, :]

globalDataAAARisky = globalDataAAA[globalDataAAA[:, CummulativeRisk] == 'True']
# globalDataAAARisky = globalDataAAARisky[globalDataAAARisky[:,idVolmax]=='True']
scanIdrisky = globalDataAAARisky[:, 2]
globalDataAAAnotRisky = globalDataAAA
nrowremoved = 0
for i in xrange(globalDataAAA.shape[0]):
    if globalDataAAA[i, 2] in scanIdrisky:
        #        print 'risk', i, i-nrowremoved
        globalDataAAAnotRisky = np.delete(globalDataAAAnotRisky, i - nrowremoved, 0)
        nrowremoved += 1

if AAAvsnoAAA:
    for i in xrange(globalData.shape[1]):
        if i not in [0, 3, 4, 18, 33, 6, 98, 25]:
            aaa = globalDataAAA[:, i].astype(float)
            noaaa = globalDatanoAAA[:, i].astype(float)
            #        print pdcolumnlist[i]
            #        print 'variance =', np.var(aaa), np.var(noaaa)
            #        print 'mean =', np.mean(aaa), np.mean(noaaa)
            #        print 'stdev =', np.std(aaa), np.std(noaaa)
            s, p = stats.ttest_ind(aaa, noaaa, equal_var=False, nan_policy='omit')
            if p < 0.001:
                p = '<0.001'
            # print globalVarNames[i],',',np.nanmean(aaa),',',np.nanstd(aaa), ',',np.nanmean(noaaa),',',np.nanstd(noaaa),',',p
    print '\n\n'

    for i in xrange(localData.shape[1]):
        if i not in [28, 29, 30]:
            aaa = localDataAAA[:, i].astype(float)
            noaaa = localDatanoAAA[:, i].astype(float)
            #        print pdcolumnlist[i]
            #        print 'variance =', np.var(aaa), np.var(noaaa)
            #        print 'mean =', np.mean(aaa), np.mean(noaaa)
            #        print 'stdev =', np.std(aaa), np.std(noaaa)
            s, p = stats.ttest_ind(aaa, noaaa, equal_var=False, nan_policy='omit')
            if p < 0.001:
                p = '<0.001'
            # print localVarNames[i],',',np.nanmean(aaa),',',np.nanstd(aaa), ',',np.nanmean(noaaa),',',np.nanstd(noaaa),',',p

if riskvsnorisk:
    for i in xrange(globalData.shape[1]):
        if i not in [0, 3, 4, 18, 33, 6, 98, 25]:
            #            print i
            risk = globalDataAAARisky[:, i].astype(float)
            norisk = globalDataAAAnotRisky[:, i].astype(float)
            #        print pdcolumnlist[i]
            #        print 'variance =', np.var(aaa), np.var(noaaa)
            #        print 'mean =', np.mean(aaa), np.mean(noaaa)
            #        print 'stdev =', np.std(aaa), np.std(noaaa)
            s, p = stats.ttest_ind(risk, norisk, equal_var=False, nan_policy='omit')
            # if p < 0.001:
            #     p = '<0.001'
            # print globalVarNames[i],',',np.nanmean(risk),',',np.nanstd(risk), ',',np.nanmean(norisk),',',np.nanstd(norisk),',',p
    print '\n\n'

nscans = []
globalData[:, 1] = globalData[:, 1].astype(int)
prev = globalData[0, 1]
cunt = 0
for i in xrange(globalData.shape[0]):
    if globalData[i, 1] == prev:
        cunt += 1
    else:
        nscans.append(cunt)
        cunt += 1
    prev = globalData[i, 1]

dt = np.empty([139])
globalData = np.insert(globalData, 100, 0, axis=1)

# lastcolumn = time (0, t1, t2+t1 etc)
prev = 0
for i in xrange(globalData[:, -1].shape[0]):
    globalData[i, -1] = prev
    if globalData[i, -4] == 'nan':
        prev = 0
    else:
        prev = globalData[i, -4].astype(float) + prev

a = np.split(globalData, nscans, axis=0)

time_seriesSlow = traces.TimeSeries()
time_seriesFast = traces.TimeSeries()

varID = 52
varname = globalVarNames[varID]
minVar = np.nanmin(globalData[:, varID].astype(float)) - 0.1 * np.nanmin(globalData[:, varID].astype(float))
maxVar = np.nanmax(globalData[:, varID].astype(float)) + 0.1 * np.nanmax(globalData[:, varID].astype(float))

p1 = figure(x_range=(-1, np.max(globalData[:, -1].astype(float) + 1)), y_range=(minVar, maxVar))
p1.grid.grid_line_alpha = 0.3
p1.xaxis.axis_label = 'Follow-up time (month)'
p1.yaxis.axis_label = varname
for i in xrange(np.max(globalData[:, 1].astype(int))):
    # b= np.append(a[i][:,0], np.indices([a[i][:,0].shape[0]]).T.reshape(a[i][:,0].shape[0]), axis=1)
    index = list(np.indices([a[i][:, 0].shape[0]])[0])
    df2 = pd.DataFrame(a[i][:, [varID, -3, -1]], index=index)
    if df2[1][0] == 'True':
        for k in xrange(a[i][:, -1].shape[0]):
            time_seriesFast[a[i][k, -1].astype(float)] = a[i][k, varID].astype(float)
        color = '#ffb3b3'
        # print 'Grand'
    else:
        for k in xrange(a[i][:, -1].shape[0]):
            time_seriesSlow[a[i][k, -1].astype(float)] = a[i][k, varID].astype(float)

        color = '#b3b3ff'
    if i > 0:
        p1.line(df2[2], df2[0], color=color, line_width=4)
        # print df2[0]

quickAvg = time_seriesFast.moving_average(1, window_size=30, pandas=True)
quickAvg = pd.DataFrame(quickAvg, columns=[varname])

slowAvg = time_seriesSlow.moving_average(1, window_size=30, pandas=True)
slowAvg = pd.DataFrame(slowAvg, columns=[varname])

p1.line(quickAvg.index, quickAvg[varname], color='#ff0000', line_width=4)
p1.line(slowAvg.index, slowAvg[varname], color='#0000ff', line_width=4)

# show(p1)


varID = 9
varname = localVarNames[varID]

p2 = figure(title='test')

# print indLoc, indLoc.shape, type(indLoc), list(indLoc[:])
l = list(indLoc[:].astype(int))

preums = 0
nscans.append(139)
index = list(np.indices([599])[0])
for i in nscans:
    n = i - prev
    id = list(np.indices([n])[0] + prev)
    # print prev, i, id, n
    npArr = np.empty([599, n])
    for k in xrange(n):
        # print (id[0] + k) * 600, (id[0] + k + 1) * 600 - 1
        npArr[:, k] = localData[(id[0] + k) * 600:(id[0] + k + 1) * 600 - 1, varID]
        df3 = pd.DataFrame(npArr, index=index)
        # print 'df3', df3.shape, npArr.shape

    prev = i
print np.max(localData[-1200:-600,varID].astype(float)),np.min(localData[-1200:-600:,varID].astype(float))
# print df3
# plt.show()
# print slow.shape, fast.shape
# sns.lmplot(slow[])
#    for i in xrange(localData.shape[1]):
#        if i not in [28, 29, 30]:
#            risk= localDataAAA[:, i].astype(float)
#            noaaa=localDatanoAAA[:, i].astype(float)
#    #        print pdcolumnlist[i]
#    #        print 'variance =', np.var(aaa), np.var(noaaa)
#    #        print 'mean =', np.mean(aaa), np.mean(noaaa)
#    #        print 'stdev =', np.std(aaa), np.std(noaaa)
##            s,p = stats.ttest_ind(aaa, noaaa, equal_var=False, nan_policy='omit')
##            if p<0.001:
##                p = '<0.001'
##            print localVarNames[i],',',np.nanmean(aaa),',',np.nanstd(aaa), ',',np.nanmean(noaaa),',',np.nanstd(noaaa),',',p
