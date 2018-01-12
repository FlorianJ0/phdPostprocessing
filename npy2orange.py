#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:18:37 2017

@author: florian
"""

import os
import numpy as np
pat = '/media/Windows/postProcess/WSS1/'

ptList = os.listdir(pat)
ptList.sort
ptList = filter(lambda a: a[:3] == 'AAA', ptList)

growthPtList = []
n = 0
growthPtList = np.empty([35,7])
for i in ptList:
    growthName = i+'_matDatafuncTime.npy'
    pathName = pat+i+'/'+growthName
    if os.path.exists(pathName):
#        print i, '\t\t', n
        g = np.load(pathName)
        avg = np.mean(g,axis=0)
        growth = (g[-1,:]-g[0,:])/g[-1,0]
        growthPtList[n] = np.insert(growth,0,n)

        n+=1
    else:
        print i
growthName = pat+'growth.txt'
#np.sort(growthPtList, axis = 2)
np.savetxt(growthName, growthPtList, delimiter='\t', header='id \t time \t vol_monthly\t area_monthly\t shape_monthly\t dmax_monthly\t dmaxTH_monthly')
print n

#b = np.load('/media/Windows/postProcess/WSS1/AAA_AB/AAA_AB_matDatafuncTime.npy')
#print b