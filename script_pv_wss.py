import os
from subprocess import call
# __version__ = '1.0'

# pat = '/RQexec/jolyflor/'
# #pat = '/home/p0054421/Downloads/test_import/'
# def listdir_fullpath(d):
#     toto = []
#     torun = ['AAA_S', 'AAA_W','AAA_Y', 'noAAA']
#     #torun = ['noAAA']#, 'AAA_B', 'AAA_C'] 
#     torun = ['AAA','noA']
#     l = [os.path.join(d, f) for f in os.listdir(d)]
#     for k in l:
#         kk =os.path.split(k)[1][:len(torun[0])]
#         if kk in torun and os.path.isdir(k):
#             j = [os.path.join(k, f) for f in os.listdir(k)]
#             for jj in j:
#                 if os.path.split(jj)[1][-5:] == '_simu' and os.path.isdir(jj):
#                     toto.append(jj)
#     return toto


# if __name__ == '__main__':

#     a = listdir_fullpath(pat)
#     a.sort()
#     #a = a[:30]
#     k = 0
#     for pat in a:
#         patt = os.path.split(pat)[1]
#         print pat
#         exit()
#         # fout = pat+'/'+patt+'.pbs'
#         # pbs = "#!/bin/bash\n\





#         #### disable automatic camera reset on 'Show'
# paraview.simple._DisableFirstRenderCameraReset()



# # load state
# LoadState('/media/Windows/postProcess/WSS1/tawss_osi_rrt.pvsm', LoadStateDataFileOptions='Choose File Names',
#     DataDirectory='/media/Windows/postProcess/WSS1',
#     AAA_LM_20130423_simufoamFileName='/media/Windows/postProcess/WSS1/noAAA_PFAB/noAAA_PFAB_20160214_simu/noAAA_PFAB_20160214_simu.foam')


# mergeBlocks1 = FindSource('MergeBlocks1')

# SetActiveSource(mergeBlocks1)


# SaveData('/media/Windows/postProcess/WSS1/335.vtk', proxy=mergeBlocks1)
# quit()




# import os
# from subprocess import call
# __version__ = '1.0'

pat = '/media/Windows/postProcess/WSS1/'
#pat = '/home/p0054421/Downloads/test_import/'
def listdir_fullpath(d):
    toto = []
    torun = ['AAA_S', 'AAA_W','AAA_Y', 'noAAA'] 
    #torun = ['noAAA']#, 'AAA_B', 'AAA_C'] 
    torun = ['AAA','noA']
    l = [os.path.join(d, f) for f in os.listdir(d)]
    for k in l:
        kk =os.path.split(k)[1][:len(torun[0])]
        if kk in torun and os.path.isdir(k):
            j = [os.path.join(k, f) for f in os.listdir(k)]
            for jj in j:
                if os.path.split(jj)[1][-5:] == '_simu' and os.path.isdir(jj):
                    toto.append(jj)
    return toto


if __name__ == '__main__':

    a = listdir_fullpath(pat)
    a.sort()
    #a = a[:30]
    k = 0
    for pat in a:
        patt = os.path.split(pat)[1]
        fout = pat+'/'+patt+'_wssrrtosi.py'
        pbs = "from paraview.simple import *\n\
paraview.simple._DisableFirstRenderCameraReset()\n\
LoadState('/media/Windows/postProcess/WSS1/tawss_osi_rrt.pvsm', LoadStateDataFileOptions='Choose File Names',DataDirectory='/media/Windows/postProcess/WSS1',    AAA_LM_20130423_simufoamFileName='"+fout+".foam')\n\
mergeBlocks1 = FindSource('MergeBlocks1')\n\
SetActiveSource(mergeBlocks1)\n\
SaveData('"+pat+'/'+patt+'_wss.vtk'+"', proxy=mergeBlocks1)\n\
quit()"
        
        if not (os.path.isdir(pat+'/0')):
            #print('reconstruct script')
            #with open(fout,'w') as f:
            #    f.write(pbs)
            #a = 'qsub ' + fout
            #print(a,'\n')
            #call(a, shell = True)
            # print('no 0, nexr')
            #print(path)
            with open(fout,'w') as f:
                print(fout)
                f.write(pbs)


            a = "~/ParaView-5.4.0-RC3-61-g4c7f251-Qt5-OpenGL2-MPI-Linux-64bit/bin/paraview --script=" + fout
            print(a,'\n')
            call(a, shell = True)

        else:
            print('already reconstructed')
            a = "mv " + pat+"/0 " + pat+"/zero"
            # call(a,shell = True)


        #a = "cp /RQexec/jolyflor/runScripts/changeDictionaryDict "+pat+"/system/"
        # a = "~/ParaView-5.4.0-RC3-61-g4c7f251-Qt5-OpenGL2-MPI-Linux-64bit/bin/paraview --script=" + fout
        # print(a)
        # call(a, shell = True)


# with open(fout,'w') as f:
#     print(fout)
    # f.write(pbs)
# a = 'cp -r /RQexec/jolyflor/test_conv/refCase/* ' + pat
# call(a, shell = True)
a = "~/ParaView-5.4.0-RC3-61-g4c7f251-Qt5-OpenGL2-MPI-Linux-64bit/bin/paraview --script=" + fout
print(a,'\n')
call(a, shell = True)

