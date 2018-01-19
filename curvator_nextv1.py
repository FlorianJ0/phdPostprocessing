from __future__ import division

# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:18:37 2017

@author: florian
Like a blind man at an orgy, I was going to have to feel things out.
"""
from vmtk import pypes
from vmtk import vmtkcontribscripts
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
import seaborn as sns
import pandas as pd
import shutil
from tqdm import tqdm
import csv
from scipy.optimize import curve_fit
import scipy
import numpy.polynomial.polynomial as poly

pat = '/media/Windows/postProcess/WSS1/'
plotsDir = '/home/florian/phd/post/plottes/'
ptList = os.listdir(pat)
ptList.sort
ptList.sort
ptList = filter(lambda a: a[:3] == 'AAA' or a[:3] == 'noA', ptList)
eps = 1E-7
growthPtList = []
nn = 0
pptList = []
computeCurv = True
scanIndex = 0  # patient id
lastScan = False  # check if last in series
k = 0
nexxtPatch = []
dt, vol, area, shape, dmax, dmaxTH = [], [], [], [], [], []
curvaturePT, tortPT, tort_monthly, curv_monthly, volth_monthly = [], [], [], [], []
tort_monthlytot, curv_monthlytot, volth_monthlytot = [], [], []
curvaturetot, torttot = [], []
thc, thctot = [], []
eth, ethtot = [], []
volthrombus, volthrombustot = [], []
toto = 0
# file struct: patientId,AGE,SEXE,IMC,Psys,Pdias,HTA,n(RxantiHTA),DLP,STATINES
dMaxSectons = np.array([39.6037911036, 42.5543824874, 45.0944839762, 42.3058183336, 44.6564335164, \
                        44.2043822915, 53.1236811747, 53.3096277186, 59.1616874243, 60.1541750511, \
                        28.4338809336, 30.8845742536, 31.2246723679, 44.9, 47.7135153082, 48.0, \
                        48.0548647198, 46.3823767679, 48.3545460221, 48.8422831112, 49.8191354173, \
                        54.9203277549, 37.9300111646, 57.898577019, 55.1, 58.6051546738, 60.2030817049, \
                        61.4, 63.6035953284, 82.9, 33.451725099, 50.1914778869, 59.640318022, \
                        41.3536031811, 43.5491223014, 48.1920081032, 58.1352427012, 59.3713740082, \
                        60.1, 59.4, 39.4513326043, 41.9759674126, 48.3397123418, 46.6733689738, \
                        49.8198622138, 51.6657851032, 57.1133597562, 28.1406162611, 30.5001206528, \
                        35.025062597, 44.4054700907, 42.7787202126, 41.3944358808, 45.6694751462, \
                        46.3051390043, 44.8927424885, 49.3097351406, 54.5249940876, 28.5075384653, \
                        29.5347104516, 29.0, 54.7, 57.0, 56.0, 58.0, 58.0, 49.0633351902, 49.6139755202, \
                        51.7279627337, 47.3239076365, 43.0, 46.6363221811, 52.5008169331, 63.3017119886, \
                        63.6271928319, 59.851219174, 47.859050117, 51.2540146065, 57.8723071288, 41.9718354646, \
                        51.0453670217, 51.4731200999, 55.4952106307, 49.059978308, 53.2586419299, \
                        52.9802114544, 51.7690829962, 53.1936045912, 58.1240543508, 61.0474807541, \
                        62.6059620658, 64.5895403365, 25.3, 26.3708217588, 34.428283987, 32.7029134802, \
                        55.5673941915, 55.0, 58.0, 57.4082387071, 57.4945017185, 58.9505905196, 60.5, \
                        54.2270828919, 55.9610242771, 54.8411875909, 58.7112747256, 61.6238677257, \
                        44.9500911182, 47.2353298369, 44.6041328071, 45.1397414311, 46.1029643649, \
                        47.6086561626, 47.4533991685, 48.9806273607, 52.1912135049, 51.2512462882, \
                        60.4828997718, 54.0530334316, 54.0529877928, 60.4829422641, 35.1565632246, \
                        38.901067447, 40.6027532392, 39.409289959, 49.7071506462, 50.3432367503, \
                        59.9323586018, 19.0524772549, 19.4620991125, 17.1281211881, 27.4466813052, \
                        19.0223888231, 15.1524665069, 18.287222266, 15.8338401954, 14.4888184585, 15.8905261632])
with open('/media/Windows/postProcess/WSS1/patient_clinical_properties.csv', 'rb') as f:
    reader = csv.reader(f)
    clinical = list(reader)
nexxt = '/media/Windows/postProcess/WSS1/AAA_AB/AAA_AB_20110126_simu_wss_patchData.npy'
nexxtPatchName = ''
nexxtPatchName_plot = ''
fail = []
tutu = 0
OSI_total = np.empty([600, 10])
scanIdex = 0
patientIndex = 0
OSI_total = []
RRT_total = []
TAWSS_total = []
ECAP_total = []
allPatches = np.empty([0, 28])
AAA = True
preums = True
varplot = []
nPatientScan = 0
nPatientScanL = []
scanIdLocal = []

#pour check avec reg a claude
dmaxTHGrowthReg = [] #list of dt,dmaxTh
dtTHGrowthReg = []
fastaclaude = 0
fastamoi = 0


# fig, ax = plt.subplots(5,8, sharex=False, sharey=True)
i = 0
last = 0
ptId = 0


def func(x, a, b, c, d):
    return a * np.exp(-c * (x - b)) + d


if computeCurv:
    fList = glob.glob(pat + '/*/*_simu_wss_patchData.npy')
    fList.sort()
    for fname in fList:
        outname = fname[:-23] + '_ctrl_registered_clipped_geom.vtp'
        curv = 'vmtkcenterlineresampling -ifile ' + fname + \
               '  -length 0.2 --pipe vmtkcenterlinesmoothing -factor 0.1 --pipe vmtkcenterlinegeometry -ofile ' + outname
        #        curv = pypes.PypeRun(curv)
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(outname)
        reader.Update()
        polyDataOutput = reader.GetOutput()

        curv = reader.GetOutput().GetPointData().GetArray(6)

        c = np.mean(vtk_to_numpy(curv))
        tortuosity = reader.GetOutput().GetCellData().GetArray(5)
        t = np.max(vtk_to_numpy(tortuosity))

        AAA = os.path.basename(fname)[0] == 'A'
        ptName = os.path.basename(fname)[5:9]
        patchFileName = fname[:-23] + '_simu_wss_patchData.npy'
        try:
            nexxtFileName = fList[fList.index(fname) + 1]
        except:
            lastScan = True

        initials = os.path.split(patchFileName)[1][:9]
        nexxtinitials = os.path.split(nexxtFileName)[1][:9]

        try:
            lastScan = initials != nexxtinitials
            preums = ptName != os.path.basename(fList[scanIndex - 1])[5:9]

        except:
            lastScan = True

        #        print preums
        #        print fname[-20:-37]
        #        print fname,
        #        print os.path.basename(fList[scanIndex - 1])[5:9], os.path.basename(fList[scanIndex + 1])[5:9]

        if not AAA:
            lastScan = False
            preums = False
        if patchFileName == '/media/Windows/postProcess/WSS1/noAAA_SAND/noAAA_SAND_20160215_simu_wss_patchData.npy':
            lastScan = True
        if patchFileName == '/media/Windows/postProcess/WSS1/noAAA_BALA/noAAA_BALA_20160213_simu_wss_patchData.npy':
            preums = True

        surf = fname[:-23] + '_simu_wss.vtp'
        clipped = fname[:-23] + '_simu_wss_clipped.vtp'
        thromb = fname[:-23] + '_MESH_rcc.vtp'

        matFileName = fname[:-23] + '_simu_wss_matData.npy'
        volFileName = fname[:-23] + '_thrombus_vol.npy'
        a = list(np.load(matFileName))
        volLum = a[0] / 1000  # convert to cm3
        areaLum = a[1] / 100  # convert to cm2
        shapeLum = a[2]
        dmaxLum = a[3]
        d0Lum = a[4]
        davgLum = a[5]
        dmaxTH = a[6]
        dmaxTH = dMaxSectons[scanIndex]
        deltadmaxTH = 0
        dmaxVarAnnual = np.NaN

        d0TH = a[7]
        davgTH = a[8]

        if preums and AAA:
            mappt = np.load(
                fname[:-32] + '_matDatafuncTimeTot.npy')  # dt ,vol, area, shape, dmax, dmaxTH, thVol, tort, curv

        if AAA and not lastScan:
            deltadmaxTH = dMaxSectons[scanIndex + 1] - dMaxSectons[scanIndex]
            dt = mappt[ptId + 1, 0] - mappt[ptId, 0]
            dmaxVarAnnual = 12 * deltadmaxTH / dt
            #            print 'dt', dt
            growth = list(12 * (mappt[ptId + 1, 1:] - mappt[ptId, 1:]) / dt)
            #            print mappt[ptId+1, -1]
            #            print ptId
            ptId += 1
        if lastScan or not AAA:
            dmaxVarAnnual = np.NaN
        # pour check avec reg a claude
        dmaxTHGrowthReg.append(dmaxTH)  # list of dt,dmaxTh
        dtTHGrowthReg = mappt[:, 0]
        slope = np.NaN
        if lastScan and AAA:
            slope, intercept, r_value, p_value, std_err = \
                scipy.stats.linregress(dtTHGrowthReg,dmaxTHGrowthReg)
            # print 'slope', slope, slope * 12, dtTHGrowthReg
            if slope*12>=5:fastaclaude+=1
            dmaxTHGrowthReg = []  # list of dt,dmaxTh
            dtTHGrowthReg = []

        if lastScan or not AAA:
            growth = list(np.ones(9) * np.nan)
            #            growth = list(growth)
            ptId = 0
        if not AAA:
            thVol = 0  # np.NAN
            growth = list(np.ones(9) * 0)
        # var in growth file: dt ,vol, area, shape, dmax, dmaxTH, thVol, tort, curv
        volLum_monthly = growth[0]
        areaLum_monthly = growth[1]
        shapeLum_monthly = growth[2]
        dmaxLum_monthly = growth[3]
        dmaxTH_monthly = growth[4]
        volTH_monthly = growth[5]
        tortLum_monthly = growth[6]
        curvLum_monthly = growth[7]
        thVol = mappt[ptId, -3] / 1000
        #        c, t = , mappt[ptId,-2], mappt[ptId,-1]
        vTot = thVol + volLum
        vTot_monthly = 0.001 * (volLum_monthly + volTH_monthly)
        vTot_monthly = vTot_monthly * 100 / vTot
        vTot_monthly_1dot5percentthreshold = vTot_monthly >= 2
        localLumAreaVarMean, localLumAreaVarMax = np.nan, np.nan
        localThThVarMean, localThThVarMax = np.nan, np.nan
        localDivAvgVarMean, localDivAvgVarMax = np.nan, np.nan
        localGradAvgVarMean, localGradAvgVarMax = np.nan, np.nan
        localOSIVarMean, localOSIVarMax = np.nan, np.nan
        localRRTVarMean, localRRTVarMax = np.nan, np.nan
        localTAWSSVarMean, localTAWSSVarMax = np.nan, np.nan
        localECAPVarMean, localECAPVarMax = np.nan, np.nan
        localDistToCtrlVarMean, localDistToCtrlVarMax = np.nan, np.nan

        thCoverage = 0
        dmGrowth1cm = 0
        # infos kept Divergence_average\tGradients_average\tOSI\tRRT\tTAWSS\tThrombus_thickness\tAbscissaMetric\nPatchArea
        if os.path.exists(patchFileName):
            nPatientScan += 1
            patch = np.load(patchFileName).astype('float32')
            nexxtPatch = np.load(nexxtFileName).astype('float32')
            DistToCtrl = patch[:, 4]
            Divergence_average = patch[:, 5]
            Gradients_average = patch[:, 6]
            OSI = patch[:, 9]
            OSI[OSI > 1] = np.nan
            PatchArea = patch[:, 10]
            RRT = patch[:, 11]
            TAWSS = patch[:, 15]
            TAWSS[TAWSS > 10] = np.nan

            if scanIndex in [12, 22] or patientIndex in [2, 28, 3, 22]:
                patch[:, 16][patch[:, -4] < 6] = 0
            if scanIndex in [] or patientIndex in [18]:
                patch[:, 16][patch[:, -4] >= 22] = 0
            th_thickness = patch[:, 16]
            th_thickness[th_thickness < 0] = 0
            if scanIndex == 8:
                th_thickness *= np.NaN

            #            patch[patch>1/eps] = np.nan
            ECAP = OSI / (TAWSS / 0.399)  # normaized by no AA avg()
            patchL = list(patch)
            patchMax = np.nanpercentile(patchL, 95, axis=0)
            patchMin = np.nanpercentile(patchL, 5, axis=0)
            patchMean = np.nanmean(patchL, axis=0)
            patchStd = np.nanstd(patchL, axis=0)
            thCoverage = 100 * len(patch[:, 16][patch[:, 16] > 1]) / len(patch[:, 16])
            thCoverageVar = np.NaN
            if not lastScan and AAA:
                thCoverageVar = np.NaN
                thCoverageNext = np.load(nexxtFileName[:-4] + 'thcoverage.npy')
                thCoverageVar = 12 * (thCoverageNext - thCoverage) / dt
            if not AAA:
                thCoverage, thCoverageVar = 0, 0

            OSI_total.append(OSI)
            ECAP_total.append(ECAP)
            RRT_total.append(RRT)
            TAWSS_total.append(TAWSS)
            thc.append(thCoverage)

            # local var
            if not lastScan or patchFileName == '/media/Windows/postProcess/WSS1/noAAA_SAND/noAAA_SAND_20160215_simu_wss_patchData.npy':
                nexxtECAP = nexxtPatch[:, 9] / nexxtPatch[:, 15]
                ECAP[ECAP > 30] = np.nan
                ECAP[ECAP > 8] = 8

                nexxtDistToCtrl = nexxtPatch[:, 4] + eps
                nexxtDivergence_average = nexxtPatch[:, 5] + eps
                nexxtGradients_average = nexxtPatch[:, 6] + eps
                nexxtOSI = nexxtPatch[:, 9] + eps
                nexxtPatchArea = nexxtPatch[:, 10] + eps
                nexxtRRT = nexxtPatch[:, 11] + eps
                nexxtTAWSS = nexxtPatch[:, 15] + eps
                if scanIndex in [11, 21] or patientIndex in [2, 28, 3, 22]:
                    #                    print 'iiiiiiiiiiii'
                    nexxtPatch[:, 16][nexxtPatch[:, -4] < 6] = 0
                if scanIndex in [] or patientIndex in [18]:
                    patch[:, 16][patch[:, -4] >= 22] = 0

                nexxtth_thickness = nexxtPatch[:, 16] + eps
                if scanIndex == 8:
                    nexxtth_thickness *= np.NaN

                if lastScan:
                    localLumAreaVarMax, localLumAreaVarSD, localLumAreaVarMin, localLumAreaVar, localLumAreaVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan
                    localThThVarMax, localThThVarSD, localThThVarMin, localThThVar, localThThVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan
                    localGradAvgVarMax, localGradAvgVarSD, localGradAvgVarMin, localGradAvgVar, localGradAvgVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan
                    localDivAvgVarMax, localDivAvgVarSD, localDivAvgVarMin, localDivAvgVar, localDivAvgVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan
                    localDistToCtrlVarMax, localDistToCtrlVarSD, localDistToCtrlVarMin, localDistToCtrlVar, localDistToCtrlVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan
                    localOSIVarMax, localOSIVarSD, localOSIVarMin, localOSIVar, localOSIVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan
                    localRRTVarMax, localRRTVarSD, localRRTVarMin, localRRTVar, localRRTVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan
                    localTAWSSVarMax, localTAWSSVarSD, localTAWSSVarMin, localTAWSSVar, localTAWSSVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan
                    localECAPVarMax, localECAPVarSD, localECAPVarMin, localECAPVar, localECAPVarMean = np.nan, np.nan, np.nan, np.ones(
                        [600, 1]) * np.nan, np.nan

                if not AAA:
                    localLumAreaVarMax, localLumAreaVarSD, localLumAreaVarMin, localLumAreaVar, localLumAreaVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0
                    localThThVarMax, localThThVarSD, localThThVarMin, localThThVar, localThThVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0
                    localGradAvgVarMax, localGradAvgVarSD, localGradAvgVarMin, localGradAvgVar, localGradAvgVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0
                    localDivAvgVarMax, localDivAvgVarSD, localDivAvgVarMin, localDivAvgVar, localDivAvgVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0
                    localDistToCtrlVarMax, localDistToCtrlVarSD, localDistToCtrlVarMin, localDistToCtrlVar, localDistToCtrlVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0
                    localOSIVarMax, localOSIVarSD, localOSIVarMin, localOSIVar, localOSIVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0
                    localRRTVarMax, localRRTVarSD, localRRTVarMin, localRRTVar, localRRTVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0
                    localTAWSSVarMax, localTAWSSVarSD, localTAWSSVarMin, localTAWSSVar, localTAWSSVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0
                    localECAPVarMax, localECAPVarMax, localECAPVarMin, localECAPVar, localECAPVarMean = 0, 0, 0, np.zeros(
                        [600, 1]), 0

                if AAA:

                    #                    print patchFileName
                    localLumAreaVar = 12 * (nexxtPatchArea - PatchArea) / dt
                    localLumAreaVarMean = np.nanmean(localLumAreaVar)
                    localLumAreaVarMax = np.nanpercentile(localLumAreaVar, 95)
                    localLumAreaVarMin = np.nanpercentile(localLumAreaVar, 5)
                    localLumAreaVarSD = np.std(localLumAreaVar)

                    th_thickness[th_thickness <= 0] = 0

                    localThThVar = 12 * (nexxtth_thickness - th_thickness) / dt
                    localThThVar[localThThVar >= 14] = 14
                    localThThVarMean = np.nanmean(localThThVar)
                    localThThVarMax = np.nanpercentile(localThThVar, 95)
                    localThThVarMin = np.nanpercentile(localThThVar, 5)
                    localThThVarSD = np.std(localThThVar)

                    #            if localeThVarMax > 1000:
                    #                localeThVarMax = np.nanNaN
                    localDivAvgVar = 12 * (nexxtDivergence_average - Divergence_average) / dt
                    localDivAvgVarMean = np.nanmean(localDivAvgVar)
                    if localDivAvgVarMean > 1E5:
                        localDivAvgVarMean = 1000
                    localDivAvgVarMax = np.nanpercentile(localDivAvgVar, 95)
                    localDivAvgVarMin = np.nanpercentile(localDivAvgVar, 5)

                    localGradAvgVar = 12 * (nexxtGradients_average - Gradients_average) / dt
                    localGradAvgVarMean = np.nanmean(localGradAvgVar)
                    localGradAvgVarMax = np.nanpercentile(localGradAvgVar, 95)
                    localGradAvgVarMin = np.nanpercentile(localGradAvgVar, 5)
                    localGradAvgVarSD = np.std(localGradAvgVar)

                    localDistToCtrlVar = 12 * (nexxtDistToCtrl - DistToCtrl) / dt
                    localDistToCtrlVarMean = np.nanmean(localDistToCtrlVar)
                    localDistToCtrlVarMax = np.nanpercentile(localDistToCtrlVar, 95)
                    localDistToCtrlVarMin = np.nanpercentile(localDistToCtrlVar, 5)
                    localDistToCtrlVarSD = np.std(localDistToCtrlVar)

                    localOSIVar = 12 * (nexxtOSI - OSI) / dt
                    localOSIVarMean = np.nanmean(localOSIVar)
                    localOSIVarMax = np.nanpercentile(localOSIVar, 95)
                    localOSIVarMin = np.nanpercentile(localOSIVar, 5)
                    localOSIVarSD = np.std(localOSIVar)

                    localRRTVar = 12 * (nexxtRRT - RRT) / dt
                    localRRTVarMean = np.nanmean(localRRTVar)
                    localRRTVarMax = np.nanpercentile(localRRTVar, 95)
                    localRRTVarMin = np.nanpercentile(localRRTVar, 5)
                    localRRTVarSD = np.std(localRRTVar)

                    localTAWSSVar = 12 * (nexxtTAWSS - TAWSS) / dt
                    localTAWSSVarMean = np.nanmean(localTAWSSVar)
                    localTAWSSVarMax = np.nanpercentile(localTAWSSVar, 95)
                    localTAWSSVarMin = np.nanpercentile(localTAWSSVar, 5)
                    localTAWSSVarSD = np.std(localTAWSSVar)

                    localECAPVar = 12 * (nexxtECAP - ECAP) / dt
                    localECAPVarMean = np.nanmean(localECAPVar)
                    localECAPVarMax = np.nanpercentile(localECAPVar, 95)
                    localECAPVarMin = np.nanpercentile(localECAPVar, 5)
                    localECAPVarSD = np.std(localECAPVar)

            growthVarList = [localLumAreaVar, localThThVar, localDivAvgVar, localGradAvgVar, localOSIVar, \
                             localRRTVar, localTAWSSVar, localECAPVar, ECAP, DistToCtrl, localDistToCtrlVar]
            tmpPatch = patch
            for gvar in growthVarList:
                gvar = np.reshape(gvar, (600, 1))
                tmpPatch = np.append(tmpPatch, gvar, axis=1)
            allPatches = np.append(allPatches, tmpPatch, axis=0)
            # print fname[-31:-23]
            # print os.path.split(fname)[1][:-23], dmaxTH, vTot
            if AAA and scanIndex < 1 and not patchFileName == '/media/Windows/postProcess/WSS1/AAA_BJGUY/AAA_BJGUY_20090213_simu_wss_patchData.npy':
                sns.set
                #                popt, pcov = curve_fit(func, x, y, [100,400,0.001,0])
                sns.set_style("dark")
                sns.set_palette("plasma", 5)
                coefs = poly.polyfit(TAWSS, th_thickness, 3)
                x_new = np.linspace(np.min(TAWSS), np.max(TAWSS), num=len(TAWSS) * 1)
                ffit = poly.polyval(x_new, coefs)
                #                plt.plot(x_new, ffit)
                #                plt.scatter(TAWSS, th_thickness)
                sns.jointplot(TAWSS, th_thickness, kind="reg", size=7, space=0)
            #                print a
            #                sns.regplot(TAWSS, th_thickness)
            #                 plt.show()

            varplot.append(localDistToCtrlVar)

            if lastScan:
                #                print 'nPatientScan ',nPatientScan
                nPatientScanL.append(nPatientScan)
                nPatientScan = 0
                #                thctot.append([listhttps://i.redd.it/100bd6hg8ua01.png(growthAr[:,0]),thc])
                #                ethtot.append([list(growthAr[:,0]),eth])
                #
                ##                print thc, 'thc', patchFileName
                #                thc = []
                #                eth= []
                ##                print 'last of serie', ptName
                ##                sns.set_style("dark")
                #                sns.set_palette("plasma",32)
                ###                toto.set(ylim=(0, 100))
                #                row = last // 8
                #                col = last% 8
                #                ax_curr = ax[row, col]
                #                ax_curr.tick_params(labelbottom='off')
                #                fig.subplots_adjust(hspace=.5)
                #                toto=sns.boxplot(data=varplot, ax=ax_curr,width=0.5,showfliers=False,linewidth=0.2)
                #                plt.plot(np.mean(ECAP), 'b-')
                #                if AAA:
                #                    toto.set_title(str(patientIndex),fontsize='xx-small')
                #                else:
                #                    toto.set_title('Healthy patients',fontsize='xx-small')

                ##                if patientIndex ==32:
                ##                    toto.set_title('Baseline aortas')
                ##                toto.set(ylim=(0, 100))
                #                plt.show()
                #                plt.setp([a.get_xticklabels() for a in axes[0   , :]], visible=False)
                varplot = []
                last += 1
        #                patientIndex +=1
        #            nexxtPatch = patch
        #            nexxtPatchName = patchFileName
        #            nexxtECAP = ECAP
        else:
            print patchFileName, 'patch file not exist'

        #        print patientIndex
        if AAA:
            patientId, AGE, SEXE, IMC, Psys, Pdias, HTA, nRxantiHTA, DLP, STATINES = clinical[patientIndex]
        else:
            patientId, AGE, SEXE, IMC, Psys, Pdias, HTA, nRxantiHTA, DLP, STATINES = list(np.ones(10) * np.nan)

        dmaxThreshold = False
        #        print SEXE
        if SEXE == 'F':
            if dmaxTH > 50:
                dmaxThreshold = True
        else:
            if dmaxTH > 55:
                dmaxThreshold = True
        dmaxTH_monthlyrisk = dmaxVarAnnual > 5 #dmaxTH_monthly * 2 > 5
        if dmaxVarAnnual > 10 : dmaxVarAnnual = 10
        if preums:
            risky = False

        if risky:
            risky = True
        elif (dmaxTH_monthlyrisk or dmaxThreshold):
            risky = True

        if not AAA:
            risky = False
        if  risky and lastScan:
            fastamoi += 1

        print 'dmax ',1 * dmaxTH_monthly * 2, dmaxVarAnnual,dmaxThreshold,risky, lastScan

        if lastScan:
            dt = np.nan
        l = [lastScan, patientIndex, scanIndex, os.path.basename(fname)[:-23], AAA, AGE, SEXE, IMC, Psys, Pdias, HTA,
             nRxantiHTA, DLP, STATINES, volLum, thVol, vTot, vTot_monthly, vTot_monthly_1dot5percentthreshold, areaLum, \
             shapeLum, dmaxLum * 2, d0Lum * 2, davgLum * 2, dmaxTH, dmaxThreshold, d0TH * 2, davgTH * 2,
             volLum_monthly / 1000, \
             areaLum_monthly / 100, shapeLum_monthly, dmaxLum_monthly * 2, dmaxVarAnnual, dmaxTH_monthlyrisk,
             volTH_monthly / 1000, \
             t, tortLum_monthly, c, curvLum_monthly, patchMax[5], patchMax[6], patchMax[9], patchMax[10], \
             patchMax[11], patchMax[15], patchMax[16], np.percentile(ECAP, 95), patchMin[5], patchMin[6], \
             patchMin[9], patchMin[10], patchMin[11], patchMin[15], patchMin[16], np.percentile(ECAP, 5), patchMean[5],
             patchMean[6], \
             patchMean[9], patchMean[10], patchMean[11], patchMean[15], patchMean[16], np.mean(ECAP), \
             patchStd[5], patchStd[6], patchStd[9], patchStd[10], patchStd[11], patchStd[15], \
             patchStd[16], np.std(ECAP), \
             localLumAreaVarMean, localLumAreaVarMax, localLumAreaVarMin, localThThVarMean, \
             localThThVarMax, localThThVarMin, localDivAvgVarMean, localDivAvgVarMax, \
             localDivAvgVarMin, localGradAvgVarMean, localGradAvgVarMax, localGradAvgVarMin, \
             localOSIVarMean, localOSIVarMax, localOSIVarMin, localRRTVarMean, localRRTVarMax, \
             localRRTVarMin, localTAWSSVarMean, localTAWSSVarMax, localTAWSSVarMin, localECAPVarMean, \
             localECAPVarMax, localECAPVarMin, thCoverage, thCoverageVar, dt, risky,slope]
        if np.inf in l:
            l = [np.nan if x in [np.inf, -np.inf] else x for x in l]
        if np.inf in l:
            print 'again'
        pptList.append(l)
        #        print 'prout'
        [np.nan if x == np.inf else x for x in pptList[-1][-50:]]

        scanIndex += 1
        if lastScan:
            patientIndex += 1
        scanIdLocal.append(np.ones(600) * scanIndex)
    # create target:  -1 var th thickness, -2 var stretching, var -3 dist to ctrl
    ththickThresh = np.reshape(np.where(allPatches[:, -10] > 0.42, 'True', 'False'), (allPatches.shape[0], 1))
    lumStrechThresh = np.reshape(np.where(allPatches[:, -11] > 5, 'True', 'False'), (allPatches.shape[0], 1))
    distToCtrlThresh = np.reshape(np.where(allPatches[:, -1] > 0.42, 'True', 'False'), (allPatches.shape[0], 1))

    allPatches = np.append(allPatches, distToCtrlThresh, axis=1)
    allPatches = np.append(allPatches, lumStrechThresh, axis=1)
    allPatches = np.append(allPatches, ththickThresh, axis=1)

    #    allPatches=np.append(allPatches,scanIdLocal,axis=1)
    #     print ptName

    length = len(sorted(pptList, key=len, reverse=True)[0])
    pptList = np.array([xi + [None] * (length - len(xi)) for xi in pptList])

    print 'fastaclaude =', fastaclaude, ' fastamoi = ', fastamoi




    outfile = pat + 'curvTortDB.npy'
    np.save(outfile, pptList)
    tabfile = pat + 'globalData.tab'

    np.savetxt("localData.tab", allPatches, delimiter='\t',
               header='AbscissaMetric\tAngularMetric\tBoundaryMetric\tClippingArray\tDistanceToCenterlines\tDivergence_average\tGradients_average\tGroupIds\tHarmonicMapping\tOSI\tPatchArea\tRRT\tSector\tSlab\tStretchedMapping\tTAWSS\tth_thickness\tlocalLumAreaVar\tlocalThThVar\tlocalDivAvgVar\tlocalGradAvgVar\tlocalOSIVar\tlocalRRTVar\tlocalTAWSSVar\tlocalECAPVar\tECAP\tlocalDistToCtrl\tlocalDistToCtrlVar\tthrombusGrowthThresh\tstretchGrowthThresh\tdisToCtrlGrowthThresh\tscanId',
               fmt='%5s')
    np.savetxt(tabfile, pptList, delimiter='\t',
               header='lastScan\tpatientId\tscanId\tptName\tAAA\tAGE\tSEXE\tIMC\tPsys\tPdias\tHTA\tnRxantiHTA\tDLP\tSTATINES\tvolLum\tvoTH\tvTot\tvTot_monthly\tvTot_monthly_2percentthreshold\tareaLum\tshapeLum\tdmaxLum\td0Lum\tdavgLum\tdmaxTH\tdmaxTH_50mmthreshold\td0TH\tdavgTH\tvolLum_monthly\tareaLum_monthly\tshapeLum_monthly\tdmaxLum_monthly\tdmaxTH_monthly\tdmaxTH_monthly5mmthreshold\tvolTH_monthly\ttortuosity\ttortLum_monthly\tcurvature\tcurvLum_monthly\tDivergence_average_max\tGradients_average_max\tOSI_max\tPatchArea_max\tRRT_max\tTAWSS_max\tThrombus_thickness_max\tECAP_max\tDivergence_average_min\tGradients_average_min\tOSI_min\tPatchArea_min\tRRT_min\tTAWSS_min\tThrombus_thickness_min\tECAP_min\tDivergence_average_mean\tGradients_average_mean\tOSI_mean\tPatchArea_mean\tRRT_mean\tTAWSS_mean\tThrombus_thickness_mean\tECAP_mean\tDivergence_average_std\tGradients_average_std\tOSI_std\tPatchArea_std\tRRT_std\tTAWSS_std\tThrombus_thickness_std\tECAP_std\tlocalLumAreaVarMean\tlocalLumAreaVarMax\tlocalLumAreaVarMin\tlocaleEThVarMean\tlocaleEThVarMax\tlocaleEThVarMin\tlocalDivAvgVarMean\tlocalDivAvgVarMax\tlocalDivAvgVarMin\tlocalGradAvgVarMean\tlocalGradAvgVarMax\tlocalGradAvgVarMin\tlocalOSIVarMean\tlocalOSIVarMax\tlocalOSIVarMin\tlocalRRTVarMean\tlocalRRTVarMax\tlocalRRTVarMin\tlocalTAWSSVarMean\tlocalTAWSSVarMax\tlocalTAWSSVarMin\tlocalECAPVarMean\tlocalECAPVarMax\tlocalECAPVarMin\tthrombusCoverage\tthrombusCoverageVar\tdt\tCummulativeRisk\tdMaxGrowthAVG',
               fmt='%5s')
    np.save("localdata.npy", allPatches)
    np.save("globaldata.npy", pptList)
    pdcolumnlist = ['lastScan', 'patientId', 'scanId', 'ptName', 'AAA', 'AGE', 'SEXE', 'IMC', 'Psys', 'Pdias', 'HTA',
                    'nRxantiHTA', 'DLP', 'STATINES', 'volLum', 'voTH', 'vTot', 'vTot_monthly',
                    'vTot_monthly_2percentthreshold', 'areaLum', 'shapeLum', 'dmaxLum', 'd0Lum', 'davgLum', 'dmaxTH',
                    'dmaxTH_50mmthreshold', 'd0TH', 'davgTH', 'volLum_monthly', 'areaLum_monthly', 'shapeLum_monthly',
                    'dmaxLum_monthly', 'dmaxVarAnnual', 'dmaxVarAnnual5mmthreshold', 'volTH_monthly', 'tortuosity',
                    'tortLum_monthly', 'curvature', 'curvLum_monthly', 'Divergence_average_max',
                    'Gradients_average_max', 'OSI_max', 'PatchArea_max', 'RRT_max', 'TAWSS_max',
                    'Thrombus_thickness_max', 'ECAP_max', 'Divergence_average_mean', 'Gradients_average_mean',
                    'OSI_mean', 'PatchArea_mean', 'RRT_mean', 'TAWSS_mean', 'Thrombus_thickness_mean', 'ECAP_mean',
                    'Divergence_average_std', 'Gradients_average_std', 'OSI_std', 'PatchArea_std', 'RRT_std',
                    'TAWSS_std', 'Thrombus_thickness_std', 'ECAP_std', 'localLumAreaVarMean', 'localLumAreaVarMax',
                    'localeEThVarMean', 'localeEThVarMax', 'localDivAvgVarMean', 'localDivAvgVarMax',
                    'localGradAvgVarMean', 'localGradAvgVarMax', 'localOSIVarMean', 'localOSIVarMax', 'localRRTVarMean',
                    'localRRTVarMax', 'localTAWSSVarMean', 'localTAWSSVarMax', 'localECAPVarMean', 'localECAPVarMax',
                    'thrombusCoverage', 'thrombusCoverageVar']

    # print toto
    #
    pairplt = 1
    #    if pairplt:
    #        #convert table to pd
    #        pptListPD = pd.DataFrame(pptList)
    #        pptListPD.columns = pdcolumnlist
    #        pptListPD['localECAPVarMax'] = pd.to_numeric(pptListPD['localECAPVarMax'], errors = 'coerce')
    #        for i in pdcolumnlist:
    #            pptListPD[i] = pd.to_numeric(pptListPD[i], errors = 'coerce')
    #            pptListPD = pptListPD[np.isfinite(pptListPD['localECAPVarMax'])]

    #    def hexbin(x, color, **kwargs):
    #        cmap = sns.light_palette(color, as_cmap=True)
    #        plt.hexbin(x, gridsize=15, cmap=cmap, **kwargs)

    #    sns.set()
    #    sns.pairplot(pptListPD[['localeEThVarMax', 'OSI_max', 'ECAP_max','Thrombus_thickness_max','dmaxTH_50mmthreshold','ECAP_mean']], hue= 'dmaxTH_50mmthreshold',palette="plasma")
    #    plt.show()
    #    g = sns.PairGrid(pptListPD[['OSI_max', 'ECAP_max','Thrombus_thickness_max','ECAP_mean']],dropna=True)
    #    g.map_upper(plt.scatter)
    #    g.map_lower(plt.hexbin, cmap="plasma")
    #    g.map_diag(plt.hist)

    print 'dun'
    sns.set()
    sns.set_style("white")
    #    sns.set_style("ticks")
    #    sns.despine(left=True)
    sns.set_context("paper")
    #    sns.axes_style()
    #    sns.despine()
    #    sns.set_palette("husl")
    #    plt.ylim(ymin = 0, ymax= 0.5)
    #    fig.set_axes
    #    plt.xlabel.
    #    plt.ylabel('Thrombus coverage (%)')
    #    plt.xlabel('Time after first acquisition (months)')
    #    plt.title('pouic')
    #    for i in xrange(len(thctot)-1):
    #        tutu = plt.plot(thctot[i][0], thctot[i][1],'.-')
    ##    t.ylim=(0, 70)
    #    plt.tight_layout()
    #    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    #    plt.savefig("thcover_allpatients.png",dpi=200)

    totalplot = True
#    plt.delaxes(ax[4][1]),plt.delaxes(ax[4][2]),plt.delaxes(ax[4][3]),plt.delaxes(ax[4][4]),plt.delaxes(ax[4][5]),plt.delaxes(ax[4][6]),plt.delaxes(ax[4][7])
#    plt.savefig("localDistToCtrlVar_allpatients.png",dpi=200)
#    plt.show()
#    if totalplot:
#        plt.rcParams["figure.figsize"] = (50,50)
#        frame1 = plt.gca()
#        frame1.axes.get_xaxis().set_visible(False)
#    #    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

#        plt.show()
import scipy.stats

allPatchesNum = np.empty([allPatches.shape[0], allPatches.shape[1]])
# nPatientScanL = [3, 3, 4, 3, 9, 3, 5, 3, 3, 4, 7, 3, 5, 3, 3, 5, 3, 3, 3, 4, 4, 3, 6, 4, 4, 3, 5, 3, 7, 4, 3, 4, 10][3, 3, 4, 3, 9, 3, 5, 3, 3, 4, 7, 3, 5, 3, 3, 5, 3, 3, 3, 4, 4, 3, 6, 4, 4, 3, 5, 3, 7, 4, 3, 4, 10]
for k in xrange(allPatches[0, :].shape[0]):
    # print k
    try:
        allPatchesNum[:, k] = [float(i) for i in allPatches[:, k]]
    except:
        break
        # print "iii"
a = 0
pval_ILTgrowth = np.empty([32, 5])
pval_lumenStretch = np.empty([32, 5])
pval_distToCenterlineGrowth = np.empty([32, 5])
# OSI,RRT,TAWSS, ILT, ECAP = column:  9 11 15 16 25
# iltgrowth, lumstretch, dst2ctrl = colum: 18 17 27
varList = [9, 11, 15, 16, 25]

'''
clf = LocalOutlierFactor(n_neighbors=50)
var=np.array([varx,vary]).T
y_pred = clf.fit_predict(var)
var_clean = var[y_pred>0]

clf = LocalOutlierFactor(n_neighbors=100)

pt = 0
vv=0
for n in nPatientScanL[:-1]:

    n -= 1 #last scan has no growth
    idStart = a *600
    idEnd = (a + n)* 600 - 1
#    print idStart, idEnd,n
    for v in varList:

        vary=allPatchesNum[idStart:idEnd,18]
        varx=allPatchesNum[idStart:idEnd,v]
#        g = sns.jointplot(varx,vary,kind="reg",dropna=True )
        mask = ~np.isnan(varx) & ~np.isnan(vary)
        var=np.array([varx[mask],vary[mask]]).T
        y_pred = clf.fit_predict(var)
        var_clean = var[y_pred>0]

        slope, intercept, r_value, p_value, std_err = \
        scipy.stats.linregress(var_clean[:,0],var_clean[:,1])
        pval_ILTgrowth[pt,vv] = r_value**2
#        print r_value**2,vv

        vary=allPatchesNum[idStart:idEnd,17]
        varx=allPatchesNum[idStart:idEnd,v]
#        g = sns.jointplot(varx,vary,kind="reg",dropna=True )
        mask = ~np.isnan(varx) & ~np.isnan(vary)
        var=np.array([varx[mask],vary[mask]]).T
        y_pred = clf.fit_predict(var)
        var_clean = var[y_pred>0]

        slope, intercept, r_value, p_value, std_err = \
        scipy.stats.linregress(var_clean[:,0],var_clean[:,1])
        pval_lumenStretch[pt,vv] = r_value**2


        vary=allPatchesNum[idStart:idEnd,27]
        varx=allPatchesNum[idStart:idEnd,v]
#        g = sns.jointplot(varx,vary,kind="reg",dropna=True )
        mask = ~np.isnan(varx) & ~np.isnan(vary)
        var=np.array([varx[mask],vary[mask]]).T
        y_pred = clf.fit_predict(var)
        var_clean = var[y_pred>0]

        slope, intercept, r_value, p_value, std_err = \
        scipy.stats.linregress(var_clean[:,0],var_clean[:,1])
        pval_distToCenterlineGrowth[pt,vv] = r_value**2
        vv+=1
    a+=n
    pt+= 1
    vv=0

print pval_ILTgrowth
pval_ILTgrowthPD = pd.DataFrame(np.append(pval_ILTgrowth,np.indices([32]).T,axis=1))
pval_ILTgrowthPD.columns = ['OSI','RRT','TAWSS', 'ILT', 'ECAP','patientIdex']

pval_lumenStretchPD = pd.DataFrame(np.append(pval_lumenStretch,np.indices([32]).T,axis=1))
pval_lumenStretchPD.columns = ['OSI','RRT','TAWSS', 'ILT', 'ECAP','patientIdex']

pval_distToCenterlineGrowthPD = pd.DataFrame(np.append(pval_distToCenterlineGrowth,np.indices([32]).T,axis=1))
pval_distToCenterlineGrowthPD.columns = ['OSI','RRT','TAWSS', 'ILT', 'ECAP','patientIdex']

sns.set()
sns.set(style="whitegrid")
cmap = sns.cubehelix_palette(as_cmap=True)
g = sns.PairGrid(pval_distToCenterlineGrowthPD,x_vars=pval_distToCenterlineGrowthPD.columns[:-1],y_vars=["patientIdex"],
                 size=10, aspect=.25)
plt.suptitle('oto')

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h", hue="husl",edgecolor="gray",linewidth=1)

# Use the same x axis limits on all columns and add better labels
g.set( xlabel="r**2", ylabel="")

# Use semantically meaningful titles for the columns
titles = ['OSI','RRT','TAWSS', 'ILT thickness', 'ECAP','patientIdex']


for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)
sns.jointplot
'''
