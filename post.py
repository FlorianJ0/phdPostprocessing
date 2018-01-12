import os
from subprocess import call
from shutil import copyfile
import numpy as np
__version__ = '1.0'
from vmtk import pypes
from vmtk import vmtkcontribscripts
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import collections as col
from scipy import stats
pat = '/media/Windows/postProcess/WSS1/'
import datetime
import glob
pat = '/home/florian/phd/post/run1/'

def export(f):
#    print f
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(f)
    print f, 'read'
    reader.Update()
    polyDataOutput = reader.GetOutput()
    gpd = polyDataOutput.GetCellData()
    nA = gpd.GetNumberOfArrays()
    if nA == 0:
        print f, 'na =0'
        return
    coords = np.empty([25, 24], dtype=int)
    for i in xrange(nA):
#        print gpd.GetArrayName(i)
        if gpd.GetArrayName(i) in ['Slab', 'Slab1', 'Slab2']:
            arr = gpd.GetArray(i)
            slab = vtk_to_numpy(arr)
            #idSlab = col.Counter(arr)
            # print idSlab
            # print idSlab[0]
#            print len(slab)
            # for i in len(idSlab):
            #

        if gpd.GetArrayName(i) in ['Sector', 'Sector1']:
            arr = gpd.GetArray(i)
            sector = vtk_to_numpy(arr)
            #idSector = col.Counter(arr)
            # print idSector
            # print idSector[0]
#            print len(sector)
#    patch = np.array([slab, sector])


    for n in xrange(slab.shape[0]):
        i = slab[n]
        j = sector[n]
        if n % 10 == 0:
#            print n
#            print int(i),int(j)
            coords[int(i), int(j)] = int(n)
#            print coords[i,j],'\n'


#    for i in xrange(25):
#        for j in xrange(24):
#            coords[i,j]=slab[i],sector[j]]

    ind = np.empty([coords.shape[0] * coords.shape[1]])
    for i in xrange(coords.shape[0]):
        for j in xrange(coords.shape[1]):
            ii = i * coords.shape[1] + j
            ind[ii] = coords[i, j]
            if ind[ii]>200000:
                ind[ii]=ind[ii-1]

    dataExport = np.empty([coords.shape[0] * coords.shape[1], nA])
#    print dataExport.shape
    fieldsList = []
#    fieldsList.append([])
#    fieldsList.append([])

    for n in xrange(nA):
        fieldsList.append([n,gpd.GetArrayName(n)])
#        fieldsList[1].append(gpd.GetArrayName(n))
    fieldsList =sorted(fieldsList,key=lambda l:l[1], reverse=False)
#    print fieldsList
    list2, my_list1 = map(list, zip(*fieldsList))
    writeId = 0
    for n in list2:
        arrName =  gpd.GetArrayName(n)
#        print arrName
        if arrName in ['Normals', 'CellEntityIds']:
            break
#        print arrName
        var = vtk_to_numpy(gpd.GetArray(n))
        for i in xrange(dataExport.shape[0]):
#            print i
            #            print var[int(ind[i])], i, n
            if arrName == 'Gradients_average':
                dataExport[i, writeId] = np.linalg.norm(var[int(ind[i])])
            else:
                try:
                    dataExport[i, writeId] = var[int(ind[i])]
                except:
#                    print f
                    return
        writeId +=1

    np.savetxt(f[:-21]+'_simu_wss_patchData.csv', dataExport)
    np.save(f[:-21]+'_simu_wss_patchData.npy', dataExport)
    print f[:-21]+'_simu_wss_patchData.npy exported'

    return dataExport


def listdir_fullpath(d):
    toto = []
    # torun = ['AAA_S', 'AAA_W', 'AAA_Y ']
    # torun = ['noAAA']#, 'AAA_B', 'AAA_C']
    torun = ['AAA', 'noA']
    l = [os.path.join(d, f) for f in os.listdir(d)]
    for k in l:
        kk = os.path.split(k)[1][:len(torun[0])]
        if kk in torun and os.path.isdir(k):
            toto.append(k)
    return toto


def patchator(diir, fiile, groupID, nz, ntheta):
    print diir, fiile
    i = diir+'/'+fiile
    print i
    # surfacator(fiile)
    print groupID
    print 'Centerlines extraction and branch splitting OK'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

#     myArguments_2 = 'vmtkbifurcationreferencesystems -ifile ' + diir + '/' + fiile[:-13] + '_ctrl_registered.vtp '\
#         '-radiusarray MaximumInscribedSphereRadius -blankingarray Blanking '\
#         '-groupidsarray GroupIds -ofile ' + diir + '/_cl_rs.vtp'
#     # myPype = pypes.PypeRun(myArguments_2)
# #
#     args = 'vmtkcenterlineattributes -ifile ' + diir + '/' + \
#         fiile[:-13] + '_ctrl_registered.vtp --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor -radiusarray MaximumInscribedSphereRadius -ofile ' + \
#         diir + '/' + fiile[:-13] + '_ctrl_registered.vtp'
    # myPype = pypes.PypeRun(args)

    print 'Bifurcation reference systems OK'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    clipped = i[:-4] + '_clipped.vtp'

    ctrl = diir + '/' + fiile[:-13] + '_ctrl_registered.vtp'
    thr = i[:-13] + '_MESH_rcc.vtp'
    myArguments_3 = 'vmtkbranchclipper -ifile ' + diir + '/' + fiile + ' -centerlinesfile ' + diir + '/' + fiile[:-13] + '_ctrl_registered.vtp' + ' '\
        '-groupidsarray GroupIds -groupids ' + str(groupID) + ' -radiusarray MaximumInscribedSphereRadius '\
        '-blankingarray Blanking --pipe vmtksurfaceconnectivity -ofile ' + clipped

    # ppt = np.array([vol, area, shape])
    # np.save(fname, ppt)

#    if not os.path.isfile(i[:-4] + '_clipped.vtp'):
    print i[:-4] + '_clipped.vtp doesnt exist. creating \n\n\n\n'
    myPype = pypes.PypeRun(myArguments_3)
    sub = 'vmtksurfacetriangle -ifile ' + clipped + ' -ofile ' + clipped
    mysub = pypes.PypeRun(sub)

    sub = 'vmtksurfacesubdivision -method butterfly -ifile ' + clipped + ' -ofile ' + clipped + ' -subdivisions 1'
#        mysub = pypes.PypeRun(sub)
    print 'Surface splitting OK'
    cp = 'vmtksurfacecapper -ifile ' + clipped + ' -ofile ' + clipped
    a = 'vmtksurfaceviewer -ifile ' + clipped
    myPype = pypes.PypeRun(a)
    close = raw_input('\n\nclose holes ?\n\n')
    if close == 'y':
        myPype = pypes.PypeRun(cp)



    remesh = False
    remesh = raw_input('subdivision?\n')
    if remesh == 'y':
        sub = 'vmtksurfacesubdivision -method butterfly -ifile ' + clipped + ' -ofile ' + clipped + ' -subdivisions 1'
        mysub = pypes.PypeRun(sub)
        print 'Surface splitting OK'

#    a='vmtksurfaceremeshing -ifile ' +clipped + ' -edgelength 0.001 -ofile ' + clipped
#    remesh = pypes.PypeRun(a)

    cp = 'vmtksurfacecapper -ifile ' + clipped + ' -ofile ' + \
        clipped[:-4] + '_closed.vtp -interactive 0'
    myPype = pypes.PypeRun(cp)
    ppt = 'vmtksurfacemassproperties -ifile ' + clipped[:-4] + '_closed.vtp'
    myppt = pypes.PypeRun(ppt)
    ptt = myppt.GetScriptObject('vmtksurfacemassproperties', '0')
    vol = ptt.Volume
    area = ptt.SurfaceArea
    shape = ptt.ShapeIndex
    print vol, area, shape


    dst = 'vmtkdistancetocenterlines -ifile ' + clipped + \
        ' -centerlinesfile ' + ctrl + ' -ofile ' + clipped
    myPype = pypes.PypeRun(dst)

    dst0 = 'vmtksurfacedistance -signeddistancearray th_thickness -ifile ' + \
        clipped + ' -rfile  ' + thr + ' -ofile ' + clipped
    myPype = pypes.PypeRun(dst0)

    thctrl = 'vmtkcenterlines -ifile ' + thr + ' -ofile ' + \
        thr[:-4] + '_ctrl.vtp -seedselector openprofiles'
    if not os.path.isfile(thr[:-4] + '_ctrl.vtp'):
        myPype = pypes.PypeRun(thctrl)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(thr[:-4] + '_ctrl.vtp')
    reader.Update()
    polyDataOutput = reader.GetOutput()
    print('\n\n\n\n\n')
    uu = reader.GetOutput().GetPointData().GetArray(0)
    print reader.GetOutput().GetPointData().GetArray(0)
    # print uu
    U = vtk_to_numpy(uu)
    [dmaxTH, d0TH, davgTH] = np.max(U), U[0], np.mean(U)
    print dmaxTH, d0TH, davgTH

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    # LUMEN
    a = 'vmtkcenterlines   -ifile ' + clipped +'  -seedselector openprofiles --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor -ofile ' + \
        diir + '/' + fiile[:-13] + '_ctrl_registered_clipped.vtp'
    aa = 'vmtkbifurcationreferencesystems -ifile ' + diir + '/' + \
        fiile[:-13] + '_ctrl_registered_clipped.vtp -radiusarray MaximumInscribedSphereRadius -blankingarray Blanking -groupidsarray GroupIds -ofile ' + \
        diir + '/' + fiile[:-13] + '_ctrl_registered_clippedRS.vtp'
    b = 'vmtkbranchclipper -ifile ' + i[:-4] + '_clipped.vtp -centerlinesfile ' + diir + '/' + fiile[:-13] + \
        '_ctrl_registered_clipped.vtp -groupidsarray GroupIds -radiusarray MaximumInscribedSphereRadius -blankingarray Blanking -ofile ' + \
        i[:-4] + '_clipped.vtp'
    c = 'vmtkbranchmetrics -ifile ' + i[:-4] + '_clipped.vtp -centerlinesfile ' + diir + '/' + fiile[:-13] + \
        '_ctrl_registered_clipped.vtp  -abscissasarray Abscissas -normalsarray ParallelTransportNormals -groupidsarray GroupIds -centerlineidsarray CenterlineIds -tractidsarray TractIds -blankingarray Blanking -radiusarray MaximumInscribedSphereRadius -ofile ' + \
        i[:-4] + '_clipped.vtp'

    # d = 'vmtkbranchmapping -ifile ' + i[:-4] + '_clipped.vtp -centerlinesfile ' + diir + '/' + fiile[:-13] + '_ctrl_registered_clipped.vtp -referencesystemsfile ' + diir + '/' + fiile[:-13] + \
    #     '_ctrl_registered_clippedRS.vtp -normalsarray ParallelTransportNormals -abscissasarray Abscissas -groupidsarray GroupIds -centerlineidsarray CenterlineIds -tractidsarray TractIds -referencesystemsnormalarray Normal -radiusarray MaximumInscribedSphereRadius -blankingarray Blanking -angularmetricarray AngularMetric -abscissametricarray AbscissaMetric -ofile ' + \
    #     i[:-4] + '_clipped.vtp'
    # e = 'vmtkbranchpatching -ifile ' + i + '/' + \
    #     fiile[:-4] + '_clipped.vtp -groupidsarray GroupIds -longitudinalmappingarray StretchedMapping -circularmappingarray AngularMetric -longitudinalpatchsize '+dz+' -circularpatches 24 -ofile ' + i[:-4] + '_clipped.vtp'
    if not os.path.isfile(diir + '/' + fiile[:-13] + '_ctrl_registered_clipped.vtp'):
        myPype = pypes.PypeRun(a)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(
        diir + '/' + fiile[:-13] + '_ctrl_registered_clipped.vtp')
    reader.Update()
    polyDataOutput = reader.GetOutput()
    print('\n\n\n\n\n')
    uu = reader.GetOutput().GetPointData().GetArray(0)
    print reader.GetOutput().GetPointData().GetArray(0)
    # print uu
    U = vtk_to_numpy(uu)
    [dmax, d0, davg] = np.max(U), U[0], np.mean(U)
    print dmax, d0, davg

    ppt = np.array([vol, area, shape, dmax, d0, davg, dmaxTH, d0TH, davgTH])

    myPype = pypes.PypeRun(aa)
    myPype = pypes.PypeRun(b)
    myPype = pypes.PypeRun(c)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(i[:-4] + '_clipped.vtp')
    # reader.ReadAllScalarsOn()
    # reader.ReadAllVectorsOn()
    reader.Update()
    polyDataOutput = reader.GetOutput()
    print('\n\n\n\n\n')
    if close == 'y':
        uu = reader.GetOutput().GetPointData().GetArray(11)
    else:
        uu = reader.GetOutput().GetPointData().GetArray(10)


    print reader.GetOutput().GetPointData()
    # print uu
    U = vtk_to_numpy(uu)
    [AbscissasMin, AbscissasMax] = np.min(U), np.max(U)
    dz = (AbscissasMax - AbscissasMin) / 24.
    print'Dz = ', dz
    raw_input('prout')
    if dz<1 or dz>10:
        dz = raw_input('dz ?\n')

    d = 'vmtkbranchmapping -ifile ' + clipped +' -centerlinesfile ' + diir + '/' + fiile[:-13] + '_ctrl_registered_clipped.vtp -referencesystemsfile ' + diir + '/' + fiile[:-13] + \
        '_ctrl_registered_clippedRS.vtp -normalsarray ParallelTransportNormals -abscissasarray Abscissas -groupidsarray GroupIds -centerlineidsarray CenterlineIds -tractidsarray TractIds -referencesystemsnormalarray Normal -radiusarray MaximumInscribedSphereRadius -blankingarray Blanking -angularmetricarray AngularMetric -abscissametricarray AbscissaMetric -ofile ' + \
        i[:-4] + '_clipped.vtp'

    e = 'vmtkbranchpatching -ifile ' + clipped+' -groupidsarray GroupIds -longitudinalmappingarray StretchedMapping -circularmappingarray AngularMetric -longitudinalpatchsize ' + \
        str(dz) + ' -circularpatches 24 -ofile ' + clipped

    mapipe = pypes.PypeRun(d)
    myPype = pypes.PypeRun(e)

    # THROMBUS
    myArguments_3 = 'vmtkbranchclipper -ifile ' + i[:-13] + '_MESH_rcc.vtp -centerlinesfile ' + diir + '/' + fiile[:-13] + '_ctrl_registered.vtp' + ' '\
        '-groupidsarray GroupIds -groupids ' + str(groupID) + ' -radiusarray MaximumInscribedSphereRadius '\
        '-blankingarray Blanking --pipe vmtksurfaceconnectivity -ofile ' + \
        i[:-13] + '_MESH_rcc.vtp'
    # myPype = pypes.PypeRun(myArguments_3)

    print 'Patching of surface mesh and attributes OK'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
#    np.save(fname, ppt)
    dual_view = 'vmtksurfacereader -ifile ' + clipped
    mypipev = pypes.PypeRun(dual_view)

    exp = export(clipped)
    return ppt, exp


if __name__ == '__main__':
    failList = glob.glob('/home/florian/phd/post/run0/AAA_CJMAN/*_simu_wss_clipped.vtp')
    print failList
#    failList = ['AAA_RL_20130925_simu_wss_clipped.vtp', 'AAA_RL_20140325_simu_wss_clipped.vtp', 'AAA_CLET_20070704_simu_wss_clipped.vtp', 'AAA_CLET_20071206_simu_wss_clipped.vtp', 'AAA_CLET_20110720_simu_wss_clipped.vtp', 'AAA_CJMAR_20100118_simu_wss_clipped.vtp', 'AAA_GFLO_20041124_simu_wss_clipped.vtp', 'AAA_LFLO_20080225_simu_wss_clipped.vtp', 'AAA_LFLO_20080421_simu_wss_clipped.vtp', 'AAA_LFLO_20071206_simu_wss_clipped.vtp', 'AAA_LFLO_20080616_simu_wss_clipped.vtp', 'AAA_LFLO_20080520_simu_wss_clipped.vtp', 'AAA_LG_20080811_simu_wss_clipped.vtp', 'AAA_RM_20061010_simu_wss_clipped.vtp', 'AAA_RM_20050531_simu_wss_clipped.vtp', 'AAA_SM_20120402_simu_wss_clipped.vtp', 'AAA_SM_20130212_simu_wss_clipped.vtp', 'AAA_WSEE_20080814_simu_wss_clipped.vtp', 'AAA_WSEE_20070614_simu_wss_clipped.vtp', 'AAA_YP_20131105_simu_wss_clipped.vtp']
    k=0
    for f in failList:
        print f, 'sent'
        export(f)
        print k

        k+=1
    regis = 0
    ctrl = 0
    print('dun')
#    coords = export(
#        '/home/florian/phd/post/AAA_CLA/AAA_CLa_20130226_simu_wss_clipped.vtp')
    print 'pejlewjnfwenfnwfnwfe'

    a = listdir_fullpath(pat)
    a.sort()
#    a=['/media/Windows/postProcess/WSS1/AAA_CSA/AAA_CSA_20140123_simu_wss_clipped.vtp]
    for i in a:
        print('\n' + os.path.split(i)[1])
        th = []
        nSeg = 0
        seg = os.path.join(i, 'segmentations')
        for s in os.listdir(seg):
            if os.path.splitext(s)[1] == '.vtk':
                nSeg += 1
                th.append(s)
        # print(nSeg, th)
        th.sort()
#        print th
        # quit()

        nt = 0 #nb of acquisitions
        ni = 0 #acquisition being evaluated
        dtPrev = 0.
        if ctrl:
            aq = [] #aquisition list
            for k in os.listdir(i):
                # print k
#                print k
                if k[-4:]=='simu':
                    nt +=1
                    aq.append(k)

            ptData = np.empty([nt, 6]) # patient data matix: date, val, area, shape, damx, dmanx

            print '\n\n number of total acquisition: ', nt
#            for k in os.listdir(i):
#                if k[-7:] == 'wss.vtp':
            aq.sort()
            print k,'roro'
#            break86
            for k in aq:
                k+='_wss.vtp'
                ni +=1
                print ni,nt
                date = k[-21:-13]
                date = datetime.datetime.strptime(date, '%Y%m%d')

                if ni == 1:
                    prevDate = date

                dt = date - prevDate
                dt = dt.days/(365/12.) + dtPrev
                prevDate = date
                dtPrev = dt
                print 'Date difference', prevDate - date
                tu = 'vmtkcenterlineviewer -cellarray GroupIds  -ifile ' + \
                    i + '/' + k[:-13] + '_ctrl_registered.vtp'
                print tu
                my = pypes.PypeRun(tu)
                var = raw_input('id  ?\n')
                [matData, patchData] = patchator(i, k, var, 24, 24)
                fname = i + '/' + k[:-4]+'_patchData.csv'
                np.savetxt(fname, patchData,header='Divergence_average, Gradients_average, OSI, RRT, TAWSS, ClippingArray, GroupIds, DistanceToCenterlines, th_thickness, AngularMetric, AbscissaMetric, BoundaryMetric, HarmonicMapping, StretchedMapping, Slab, Sector, PatchArea\n', comments='', newline=' ')
                np.save(fname[:-4],patchData)
                fname = i + '/' + k[:-4]+'_matData.csv'
                np.savetxt(fname, matData,header='vol, area, shape, dmax, d0, davg, dmaxTH, d0TH, davgTH\n', comments='', newline=' ')
                np.save(fname[:-4], matData)


                #[vol, area, shape, dmax, d0, davg, dmaxTH, d0TH, davgTH]
#                print [float(dt), matData[0],matData[1], matData[2], matData[3], matData[6] ]
                ptData[ni-1,:] = [dt, matData[0],matData[1], matData[2], matData[3], matData[6] ]
                if nt - ni == 0:
                        fname = i + '/' + k[:-22]+'_matDatafuncTime.csv'
                        np.savetxt(fname, ptData, header='dt ,vol, area, shape, dmax, dmaxTH', comments='')
                        np.save(fname[:-4], ptData)
                        avg = np.mean(ptData,axis=0)
                        avg[0] = dt/nt
                        fname = i + '/' + k[:-22]+'_matDataavgTime.csv'
                        np.savetxt(fname, avg, header='dt_avg ,vol_avg, area_avg, shape_avg, dmax_avg, dmaxTH_avg\n', comments='', newline=' ')
                        np.save(fname[:-4], avg)
                        growth = (ptData[-1,:]-ptData[0,:])/dt
                        fname = i + '/' + k[:-22]+'_matDatagrowthTime.csv'
                        np.savetxt(fname, growth[1:], header='vol_monthly, area_monthly, shape_monthly, dmax_monthly, dmaxTH_monthly\n', comments='', newline=' ')
                        np.save(fname[:-4], growth[1])
                        lin = growth[1:]
                        for t in xrange(5):
                            slope, intercept, r_value, p_value, std_err = stats.linregress(ptData[:,0],ptData[:,t+1])
                            print ptData[:,0], ptData[:,t+1]
                            print '\n\n\n'
                            print slope, intercept, r_value, p_value, std_err
                            print r_value*r_value
                            lin[t] = r_value*r_value
                            print lin
#                            quit()
                        fname = i + '/' + k[:-22]+'_matrSquare.csv'
                        np.savetxt(fname, lin, header='vol_r2, area_r2, shape_r2, dmax_r2, dmaxTH_r2\n', comments='', newline=' ')
                        np.save(fname[:-4], lin)

#
        if regis:
            cpp = 'vmtksurfacewriter -ifile ' + seg + '/' + \
                th[0] + ' -ofile ' + i + '/' + th[0][:-4] + '_rcc.vtp'
            print cpp
            mycpp = pypes.PypeRun(cpp)
            tutu = 'vmtksurfacetransform -ifile ' + i + '/' + \
                th[0][:-4] + '_rcc.vtp -ofile ' + i + '/' + \
                th[0][:-4] + '_rcc.vtp -scaling 1000 1000 1000'
            mycpp = pypes.PypeRun(tutu)
            tutu = 'vmtksurfacetransform -ifile ' + i + '/' + \
                th[0][:-4] + '_rcc.vtp -ofile ' + i + '/' + \
                th[0][:-4] + '_rcc.vtp -rotation 0 0 180'
            mycpp = pypes.PypeRun(tutu)

            for ii in th:
                # make wss file readable by vmtk
                copyfile('/home/florian/phd/vtu2vtp.py', i +
                         '/' + ii[:-8] + 'simu/vtu2vtp.py')
                print('/home/florian/phd/vtu2vtp.py', i +
                      '/' + ii[:-8] + 'simu/vtu2vtp.py')

                fname = i + '/' + ii[:-8] + 'simu/' + ii[:-8] + 'simu_wss.vtk'
                fname = fname.replace('/', '\/')
                a = "sed -i 's/filename/" + fname + "/g'  " + \
                    i + '/' + ii[:-8] + "simu/vtu2vtp.py"
                call(a, shell=True)
                a = '/home/florian/ParaView-5.4.0-263-g737e178-Qt5-MPI-Linux-64bit/bin/paraview  --script=' + \
                    i + '/' + ii[:-8] + "simu/vtu2vtp.py"
                call(a, shell=True)
                tutu = 'vmtkmeshtosurface -ifile ' + i + '/' + \
                    ii[:-8] + 'simu/' + ii[:-8] + 'simu_wss.vtu -ofile ' + \
                    i + '/' + ii[:-8] + 'simu_wss.vtp'
                mycpp = pypes.PypeRun(tutu)
                tutu = 'vmtksurfacetransform -ifile ' + i + '/' + \
                    ii[:-8] + 'simu_wss.vtp  -ofile ' + i + '/' + \
                    ii[:-8] + 'simu_wss.vtp -scaling 1000 1000 1000'
                mycpp = pypes.PypeRun(tutu)

                # register
                mat = i + '/' + ii[:-8] + 'split.stl_transformation-matrix.dat'
                ref = i + '/' + ii[:-8] + 'split_registered.stl'
                if os.path.isfile(mat):
                    trans = np.empty([4, 4])
                    with open(mat) as f:
                        matt = f.readlines()
                        matt = [x.strip() for x in matt]
                        # matt =  [y for x in matt for y in x]
                    matt = ((str(matt).replace(',', '').replace(
                        '[', '').replace(']', '').replace('\'', '')))
                    matt = matt.split()
                    # print matt
                    # print matt[:3]
                    trans[0] = matt[:4]
                    trans[1] = matt[4:8]
                    trans[2] = matt[8:12]
                    trans[3] = matt[12:16]
                    roro = trans[:3, :3]
                    thname = i + '/segmentations/' + ii
                    thnamerc = 'totorc.vtp'
                    # thnamercc = thname[:-4]+'_rcc.vtp'
                    thnamercc = i + '/' + ii[:-4] + '_rcc.vtp'

                    a = 'vmtksurfacetransform -ifile ' + thname + \
                        ' -ofile ' + thnamerc + ' -scaling 1000 1000 1000'
                    call(a, shell=True)
                    print a
                    a = 'vmtksurfacetransform -ifile ' + thnamerc + \
                        ' -ofile ' + thnamerc + ' -rotation 0 0 180'
                    call(a, shell=True)
                    a = 'vmtkicpregistration -landmarks 5000 -ifile ' + \
                        ref[:-15] + '.stl  -rfile  ' + \
                        ref + '   -ofile tutu.vtp'
                    mypipe = pypes.PypeRun(a)
                    transmat = mypipe.GetScriptObject(
                        'vmtkicpregistration', '0')
                    mat0 = transmat.MatrixCoefficients
                    npmat0 = np.reshape(np.array(mat0), [4, 4])

                    dual_view = 'vmtksurfacereader -ifile ' + ref + \
                        ' --pipe vmtkrenderer --pipe vmtksurfaceviewer -display 0  --pipe vmtksurfaceviewer -ifile  tutu.vtp -color 1 0 0 -display 1'
                    mypipev = pypes.PypeRun(dual_view)
                    var = raw_input(
                        'Manual correction  ? [y]/[n]/[r]otation\n')
                    if var == 'y':
                        print 'entering manual correction '

                        a = 'vmtksurfacetransforminteractive -ifile  tutu.vtp -ofile tata.vtp -rfile ' + ref
                        mypipe = pypes.PypeRun(a)
                        transmat = mypipe.GetScriptObject(
                            'vmtksurfacetransforminteractive', '0')
                        mat = transmat.MatrixCoefficients
                        npmanuel = np.reshape(np.array(mat), [4, 4])
                        a = 'vmtkicpregistration -landmarks 5000 -ifile tata.vtp  -rfile  ' + \
                            ref + '   -ofile tutu.vtp'
                        mypipe = pypes.PypeRun(a)

                        transmat = mypipe.GetScriptObject(
                            'vmtkicpregistration', '0')
                        maticp = transmat.MatrixCoefficients
                        perf = transmat.MaximumMeanDistance
                        print('~~PERF = ', perf)
                        npmaticp = np.reshape(np.array(maticp), [4, 4])
                        print(mat)
                        newmat = np.dot(npmaticp, npmat0)
                        newmat = np.dot(npmanuel, newmat)
                        dual_view = 'vmtksurfacereader -ifile ' + ref + \
                            ' --pipe vmtkrenderer --pipe vmtksurfaceviewer -display 0  --pipe vmtksurfaceviewer -ifile  tutu.vtp -color 1 0 0 -display 1'
                        mypipev = pypes.PypeRun(dual_view)

                        a = 'vmtksurfacetransform -ifile ' + thnamerc + \
                            ' -ofile ' + thnamerc + '  -rotation 0 0 180'
                        # mypipesd = pypes.PypeRun(a)
                        a = 'vmtksurfacetransform -ifile ' + thnamerc + ' -ofile ' + thnamercc + ' -matrix ' + \
                            str(((str(newmat).replace(',', '').replace(
                                '[', '').replace(']', '').replace('\'', ''))))
                        mypipe = pypes.PypeRun(a)

                    else:
                        print 'no entering manual correction'
                        a = 'vmtksurfacetransform -ifile ' + thnamerc + ' -ofile ' + thnamercc + ' -matrix ' + \
                            str(((str(mat0).replace(',', '').replace(
                                '[', '').replace(']', '').replace('\'', ''))))
                        mypipe = pypes.PypeRun(a)

                    dual_view = 'vmtksurfacereader -ifile ' + ref + \
                        ' --pipe vmtkrenderer --pipe vmtksurfaceviewer -display 0  --pipe vmtksurfaceviewer -ifile ' + \
                        thnamercc + ' -color 1 0 0 -display 1'
                    mypipe = pypes.PypeRun(dual_view)
