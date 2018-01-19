#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:59:47 2017

@author: florian
"""

import os
from scipy import stats
import numpy as np

localData = np.load("localdata.npy")
globalData = np.load("globaldata.npy")
print globalData[:,-1]
print globalData[:,1]

globalVarNames = ['lastScan','patientId','scanId','ptName','AAA','AGE','SEXE','IMC',\
                  'Psys','Pdias','HTA','nRxantiHTA','DLP','STATINES','volLum','voTH',\
                  'vTot','vTot_monthly','vTot_monthly_2percentthreshold','areaLum',\
                  'shapeLum','dmaxLum','d0Lum','davgLum','dmaxTH','dmaxTH_50mmthreshold',\
                  'd0TH','davgTH','volLum_monthly','areaLum_monthly','shapeLum_monthly'\
                  ,'dmaxLum_monthly','dmaxTH_monthly','dmaxTH_monthly5mmthreshold',\
                  'volTH_monthly','tortuosity','tortLum_monthly','curvature',\
                  'curvLum_monthly','Divergence_average_max','Gradients_average_max',\
                  'OSI_max','PatchArea_max','RRT_max','TAWSS_max','Thrombus_thickness_max',\
                  'ECAP_max','Divergence_average_min','Gradients_average_min',\
                  'OSI_min','PatchArea_min','RRT_min','TAWSS_min','Thrombus_thickness_min',\
                  'ECAP_min','Divergence_average_mean','Gradients_average_mean',\
                  'OSI_mean','PatchArea_mean','RRT_mean','TAWSS_mean','Thrombus_thickness_mean',\
                  'ECAP_mean','Divergence_average_std','Gradients_average_std','OSI_std',\
                  'PatchArea_std','RRT_std','TAWSS_std','Thrombus_thickness_std','ECAP_std',\
                  'localLumAreaVarMean','localLumAreaVarMax',\
                  ' localLumAreaVarMin','localThThVarMean','localThThVarMax','localThThVarMin',\
                  ' localDivAvgVarMean','localDivAvgVarMax','localDivAvgVarMin','localGradAvgVarMean',\
                  'localGradAvgVarMax','localGradAvgVarMin',' localOSIVarMean','localOSIVarMax',\
                  'localOSIVarMin','localRRTVarMean','localRRTVarMax','localRRTVarMin',\
                  'localTAWSSVarMean',' localTAWSSVarMax','localTAWSSVarMin','localECAPVarMean',\
                  'localECAPVarMax','localECAPVarMin','thrombusCoverage',\
                  'thrombusCoverageVar', 'dt','CummulativeRisk', 'dMaxGrowthRegression']


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


localDataAAA = localData[:-5400,:]
localDatanoAAA = localData[-5399:,:]


globalDataAAA = globalData[:-10,:]
globalDatanoAAA = globalData[-9:,:]

globalDataAAARisky = globalDataAAA[globalDataAAA[:,CummulativeRisk]=='True']
#globalDataAAARisky = globalDataAAARisky[globalDataAAARisky[:,idVolmax]=='True']
scanIdrisky = globalDataAAARisky[:,2]
globalDataAAAnotRisky = globalDataAAA
nrowremoved = 0
for i in xrange(globalDataAAA.shape[0]):
    if globalDataAAA[i,2] in scanIdrisky:
#        print 'risk', i, i-nrowremoved
        globalDataAAAnotRisky=np.delete(globalDataAAAnotRisky, i-nrowremoved ,0)
        nrowremoved += 1





if AAAvsnoAAA:
    for i in xrange(globalData.shape[1]):
        if i not in [0,3,4,18,33,6,98,25 ]:
            aaa= globalDataAAA[:, i].astype(float)
            noaaa=globalDatanoAAA[:, i].astype(float)
    #        print pdcolumnlist[i]
    #        print 'variance =', np.var(aaa), np.var(noaaa)
    #        print 'mean =', np.mean(aaa), np.mean(noaaa)
    #        print 'stdev =', np.std(aaa), np.std(noaaa)
            s,p = stats.ttest_ind(aaa, noaaa, equal_var=False, nan_policy='omit')
            if p<0.001:
                p = '<0.001'
            # print globalVarNames[i],',',np.nanmean(aaa),',',np.nanstd(aaa), ',',np.nanmean(noaaa),',',np.nanstd(noaaa),',',p
    print '\n\n'

    for i in xrange(localData.shape[1]):
        if i not in [28, 29, 30]:
            aaa= localDataAAA[:, i].astype(float)
            noaaa=localDatanoAAA[:, i].astype(float)
    #        print pdcolumnlist[i]
    #        print 'variance =', np.var(aaa), np.var(noaaa)
    #        print 'mean =', np.mean(aaa), np.mean(noaaa)
    #        print 'stdev =', np.std(aaa), np.std(noaaa)
            s,p = stats.ttest_ind(aaa, noaaa, equal_var=False, nan_policy='omit')
            if p<0.001:
                p = '<0.001'
            # print localVarNames[i],',',np.nanmean(aaa),',',np.nanstd(aaa), ',',np.nanmean(noaaa),',',np.nanstd(noaaa),',',p



if riskvsnorisk:
    for i in xrange(globalData.shape[1]):
        if i not in [0,3,4,18,33,6,98,25 ]:
#            print i
            risk= globalDataAAARisky[:, i].astype(float)
            norisk=globalDataAAAnotRisky[:, i].astype(float)
    #        print pdcolumnlist[i]
    #        print 'variance =', np.var(aaa), np.var(noaaa)
    #        print 'mean =', np.mean(aaa), np.mean(noaaa)
    #        print 'stdev =', np.std(aaa), np.std(noaaa)
            s,p = stats.ttest_ind(risk, norisk, equal_var=False, nan_policy='omit')
            if p<0.001:
                p = '<0.001'
            # print globalVarNames[i],',',np.nanmean(risk),',',np.nanstd(risk), ',',np.nanmean(norisk),',',np.nanstd(norisk),',',p
    print '\n\n'

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