#### import the simple module from the paraview
#from paraview.simple import *
#### disable automatic camera reset on 'Show'
from vmtk import pypes
from vmtk import vmtkcontribscripts
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from vmtk import vmtkscripts

l=[
  '/media/Windows/postProcess/WSS1/AAA_AB/AAA_AB_20110126_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_AB/AAA_AB_20130218_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_AB/AAA_AB_20140714_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_AL/AAA_AL_20130423_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_AL/AAA_AL_20131129_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_AL/AAA_AL_20140516_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_BJGUY/AAA_BJGUY_20070928_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_BJGUY/AAA_BJGUY_20071003_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_BJGUY/AAA_BJGUY_20090213_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_BJGUY/AAA_BJGUY_20090421_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CF/AAA_CF_20110823_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CF/AAA_CF_20130814_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CF/AAA_CF_20130816_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20040107_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20050113_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20061130_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20070502_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20071227_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20080626_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20090810_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20101208_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAN/AAA_CJMAN_20120320_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAR/AAA_CJMAR_20050506_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAR/AAA_CJMAR_20100118_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CJMAR/AAA_CJMAR_20100817_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CLa/AAA_CLa_20031117_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CLa/AAA_CLa_20100705_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CLa/AAA_CLa_20130226_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CLET/AAA_CLET_20061215_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CLET/AAA_CLET_20070222_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CLET/AAA_CLET_20070704_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CLET/AAA_CLET_20071206_MESH_rcc.vtp',
#  '/media/Windows/postProcess/WSS1/AAA_CLET/AAA_CLET_20110720_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CSA/AAA_CSA_20120328_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CSA/AAA_CSA_20130225_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CSA/AAA_CSA_20140123_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CSL/AAA_CSL_20121207_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CSL/AAA_CSL_20130502_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CSL/AAA_CSL_20140103_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_CSL/AAA_CSL_20140715_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FL/AAA_FL_20040630_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FL/AAA_FL_20050601_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FL/AAA_FL_20060928_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FL/AAA_FL_20070301_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FL/AAA_FL_20070905_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FL/AAA_FL_20080403_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FL/AAA_FL_20081020_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FR/AAA_FR_20090730_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FR/AAA_FR_20110125_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_FR/AAA_FR_20121017_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GFLO/AAA_GFLO_20030306_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GFLO/AAA_GFLO_20030505_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GFLO/AAA_GFLO_20030731_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GFLO/AAA_GFLO_20041124_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GFLO/AAA_GFLO_20060602_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GJGUY/AAA_GJGUY_20070416_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GJGUY/AAA_GJGUY_20080815_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GJGUY/AAA_GJGUY_20100413_MESH_rcc.vtp',
#  '/media/Windows/postProcess/WSS1/AAA_GJGUY/AAA_GJGUY_20111228_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GV/AAA_GV_20110419_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GV/AAA_GV_20130417_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_GV/AAA_GV_20140527_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LFLO/AAA_LFLO_20071206_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LFLO/AAA_LFLO_20080225_MESH_rcc.vtp',
#  '/media/Windows/postProcess/WSS1/AAA_LFLO/AAA_LFLO_20080421_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LFLO/AAA_LFLO_20080520_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LFLO/AAA_LFLO_20080616_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LG/AAA_LG_20080811_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LG/AAA_LG_20090203_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LG/AAA_LG_20090803_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LJAC/AAA_LJAC_20040609_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LJAC/AAA_LJAC_20050622_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LJAC/AAA_LJAC_20070711_MESH_rcc.vtp',
#  '/media/Windows/postProcess/WSS1/AAA_LJAC/AAA_LJAC_20080703_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LM/AAA_LM_20090826_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LM/AAA_LM_20130423_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_LM/AAA_LM_20130524_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_MB/AAA_MB_20061227_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_MB/AAA_MB_20100908_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_MB/AAA_MB_20110505_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_MB/AAA_MB_20120503_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_MSER/AAA_MSER_20110629_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_MSER/AAA_MSER_20120515_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_MSER/AAA_MSER_20130212_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_MSER/AAA_MSER_20131223_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PBEN/AAA_PBEN_20101220_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PBEN/AAA_PBEN_20120727_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PBEN/AAA_PBEN_20120828_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PJAC/AAA_PJAC_20080604_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PJAC/AAA_PJAC_20081106_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PJAC/AAA_PJAC_20091111_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PJAC/AAA_PJAC_20100712_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PJAC/AAA_PJAC_20101115_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_PJAC/AAA_PJAC_20101206_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RFRA/AAA_RFRA_20080218_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RFRA/AAA_RFRA_20090730_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RFRA/AAA_RFRA_20110125_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RFRA/AAA_RFRA_20121017_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RL/AAA_RL_20130208_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RL/AAA_RL_20130925_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RL/AAA_RL_20140325_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RL/AAA_RL_20140917_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RM/AAA_RM_20031120_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RM/AAA_RM_20050531_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RM/AAA_RM_20061010_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RS/AAA_RS_20100505_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RS/AAA_RS_20101115_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RS/AAA_RS_20111128_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RS/AAA_RS_20130111_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_RS/AAA_RS_20131217_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SK/AAA_SK_20111019_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SK/AAA_SK_20130930_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SK/AAA_SK_20140114_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SM/AAA_SM_20111012_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SM/AAA_SM_20120111_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SM/AAA_SM_20120402_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SM/AAA_SM_20120515_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SM/AAA_SM_20120703_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SM/AAA_SM_20121213_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_SM/AAA_SM_20130212_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_WSEE/AAA_WSEE_20050311_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_WSEE/AAA_WSEE_20060519_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_WSEE/AAA_WSEE_20070614_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_WSEE/AAA_WSEE_20080814_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_YL/AAA_YL_20090529_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_YL/AAA_YL_20100507_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_YL/AAA_YL_20121217_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_YP/AAA_YP_20100507_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_YP/AAA_YP_20120604_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_YP/AAA_YP_20121213_MESH_rcc.vtp',
  '/media/Windows/postProcess/WSS1/AAA_YP/AAA_YP_20131105_MESH_rcc.vtp',
#  '/media/Windows/postProcess/WSS1/noAAA_BCLA/noAAA_BCLA_20160215_simu_wss_clipped.vtp',
#  '/media/Windows/postProcess/WSS1/noAAA_PFAB/noAAA_PFAB_20160214_simu_wss_clipped.vtp',
#  '/media/Windows/postProcess/WSS1/noAAA_BFRA/noAAA_BFRA_20160215_simu_wss_clipped.vtp',
#  '/media/Windows/postProcess/WSS1/noAAA_PMON/noAAA_PMON_20160215_simu_wss_clipped.vtp',
#  '/media/Windows/postProcess/WSS1/noAAA_CANT/noAAA_CANT_20150911_simu_wss_clipped.vtp',
#  '/media/Windows/postProcess/WSS1/noAAA_SAND/noAAA_SAND_20160215_simu_wss_clipped.vtp',
#  '/media/Windows/postProcess/WSS1/noAAA_CSER/noAAA_CSER_20160214_simu_wss_clipped.vtp',

  ]
#l.sort
print(l.sort)
dMaxList = []
ptList=[]
i=0
for fname in l[:]:
    print i
    i+=1
# fname =   '/media/Windows/postProcess/WSS1/AAA_SM/AAA_SM_20120515_simu_wss.vtp'
    outname = fname[:-12]+'thrombus.vtp'
    capped = fname[:-12]+'thrombuscappedz.vtp'
    lum = fname[:-12]+'simu_wss_clipped.vtp'
    ctrl = fname[:-12]+'MESH_rcc_ctrl.vtp'
    th = fname[:-12]+'MESH_rcc.vtp'
    ctrlSM = fname[:-12]+'MESH_rcc_ctrlSM.vtp'
    patchFileNameFix = fname[:-12]+'simu_wss_patchData.npy'

    sections = fname[:-12]+'TH_sections.vtp'
    a = 'vmtkcenterlinesections -ifile '+th + ' -centerlinesfile '+ ctrlSM + ' -ofile ' + sections

#    aa = pypes.PypeRun(a)
    print a

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(sections)
    reader.Update()
    polyDataOutput = reader.GetOutput()
    gpd = polyDataOutput.GetCellData()
    nA = gpd.GetNumberOfArrays()

    dmax = reader.GetOutput().GetCellData().GetArray(2)
    t = np.max(vtk_to_numpy(dmax))
    print patchFileNameFix, ' Dmax ',t
    dMaxList.append(t)
    ptList.append(patchFileNameFix)

    a= 'vmtkcenterlineresampling -ifile ' + ctrl + \
            '  -length 0.2 --pipe vmtkcenterlinesmoothing -factor 0.1 -ofile ' + ctrlSM

#    aa = pypes.PypeRun(a)
    a = 'vmtksurfacereader -ifile '+outname+'  --pipe vmtkcenterlines -seedselector openprofiles \
     -capdisplacement 0.5  --pipe vmtkendpointextractor -numberofendpointspheres 1 --pipe vmtkbranchclipper\
      --pipe vmtksurfaceconnectivity -cleanoutput 1 --pipe vmtksurfacecapper -interactive 0\
       --pipe vmtksurfacewriter -ofile ' + capped
    # aa = pypes.PypeRun(a)

    dual_view = 'vmtksurfacereader -ifile ' + lum + \
     ' --pipe vmtkrenderer --pipe vmtksurfaceviewer -display 0  --pipe vmtksurfaceviewer -ifile  '+capped+' -color 1 0 0 -display 1'
    # mypipev = pypes.PypeRun(dual_view)

    triangulation = 'vmtksurfacetriangle  -ifile ' +lum+ ' -ofile ' + fname[:-12]+'lumen.vtp'
    # mypipev = pypes.PypeRun(triangulation)

    bobol = 'vmtksurfacebooleanoperation -i2file ' +fname[:-12]+'lumen.vtp   -ifile ' +capped+ '  -ofile '+ fname[:-12]+'thrombus_vol.vtp -operation difference'
#    pypes.PypeRun(bobol)

    # view = 'vmtksurfaceviewer -ifile ' + fname[:-12]+'thrombus_vol.vtp'
    # pypes.PypeRun(view)

    ppt = 'vmtksurfacemassproperties -ifile '+ fname[:-12]+'thrombus_vol.vtp'
#    pypes.PypeRun(ppt)

    ppt1 = 'vmtksurfaceconnectivity -ifile '+ fname[:-12]+'thrombus_vol.vtp  --pipe vmtksurfacemassproperties '
#    pypes.PypeRun(ppt1)

    dual_view = 'vmtksurfacereader -ifile ' + fname[:-12]+'thrombus_vol.vtp' + \
     ' --pipe vmtksurfaceconnectivity --pipe vmtkrenderer --pipe vmtksurfaceviewer -color 1 0 0 -display 0  --pipe vmtksurfaceviewer -ifile  '+fname[:-12]+'thrombus_vol.vtp  -display 1'
#    mypipev = pypes.PypeRun(dual_view)

    ppt = 'vmtksurfacemassproperties -ifile ' + fname[:-12]+'thrombus_vol.vtp'
#    myppt = pypes.PypeRun(ppt)
#    ptt = myppt.GetScriptObject('vmtksurfacemassproperties', '0')
#    vol = ptt.Volume
#    print vol, '\n\n\n\n\n\n\n\n\n\n'
#    np.save(fname[:-12]+'thrombus_vol.npy',np.float(vol))

'''
    paraview.simple._DisableFirstRenderCameraReset()

    # create a new 'XML PolyData Reader'
    aAA_LG_20080811_MESH_rccvtp = XMLPolyDataReader(FileName=[fname])
    aAA_LG_20080811_MESH_rccvtp.PointArrayStatus = ['scalars']

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1298, 803]

    # get color transfer function/color map for 'scalars'
    scalarsLUT = GetColorTransferFunction('scalars')

    # show data in view
    aAA_LG_20080811_MESH_rccvtpDisplay = Show(aAA_LG_20080811_MESH_rccvtp, renderView1)
    # trace defaults for the display properties.
    aAA_LG_20080811_MESH_rccvtpDisplay.Representation = 'Surface'
    aAA_LG_20080811_MESH_rccvtpDisplay.ColorArrayName = ['POINTS', 'scalars']
    aAA_LG_20080811_MESH_rccvtpDisplay.LookupTable = scalarsLUT
    aAA_LG_20080811_MESH_rccvtpDisplay.OSPRayScaleArray = 'scalars'
    aAA_LG_20080811_MESH_rccvtpDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    aAA_LG_20080811_MESH_rccvtpDisplay.SelectOrientationVectors = 'None'
    aAA_LG_20080811_MESH_rccvtpDisplay.ScaleFactor = 12.453399658203125
    aAA_LG_20080811_MESH_rccvtpDisplay.SelectScaleArray = 'scalars'
    aAA_LG_20080811_MESH_rccvtpDisplay.GlyphType = 'Arrow'
    aAA_LG_20080811_MESH_rccvtpDisplay.GlyphTableIndexArray = 'scalars'
    aAA_LG_20080811_MESH_rccvtpDisplay.DataAxesGrid = 'GridAxesRepresentation'
    aAA_LG_20080811_MESH_rccvtpDisplay.PolarAxes = 'PolarAxesRepresentation'
    aAA_LG_20080811_MESH_rccvtpDisplay.GaussianRadius = 6.2266998291015625
    aAA_LG_20080811_MESH_rccvtpDisplay.SetScaleArray = ['POINTS', 'scalars_0']
    aAA_LG_20080811_MESH_rccvtpDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    aAA_LG_20080811_MESH_rccvtpDisplay.OpacityArray = ['POINTS', 'scalars_0']
    aAA_LG_20080811_MESH_rccvtpDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    aAA_LG_20080811_MESH_rccvtpDisplay.InputVectors = [None, '']

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    aAA_LG_20080811_MESH_rccvtpDisplay.OSPRayScaleFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    aAA_LG_20080811_MESH_rccvtpDisplay.ScaleTransferFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    aAA_LG_20080811_MESH_rccvtpDisplay.OpacityTransferFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # reset view to fit data
    renderView1.ResetCamera()

    # show color bar/color legend
    aAA_LG_20080811_MESH_rccvtpDisplay.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Extract Surface'
    extractSurface1 = ExtractSurface(Input=aAA_LG_20080811_MESH_rccvtp)

    # show data in view
    extractSurface1Display = Show(extractSurface1, renderView1)
    # trace defaults for the display properties.
    extractSurface1Display.Representation = 'Surface'
    extractSurface1Display.ColorArrayName = ['POINTS', 'scalars']
    extractSurface1Display.LookupTable = scalarsLUT
    extractSurface1Display.OSPRayScaleArray = 'scalars'
    extractSurface1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    extractSurface1Display.SelectOrientationVectors = 'None'
    extractSurface1Display.ScaleFactor = 12.453399658203125
    extractSurface1Display.SelectScaleArray = 'scalars'
    extractSurface1Display.GlyphType = 'Arrow'
    extractSurface1Display.GlyphTableIndexArray = 'scalars'
    extractSurface1Display.DataAxesGrid = 'GridAxesRepresentation'
    extractSurface1Display.PolarAxes = 'PolarAxesRepresentation'
    extractSurface1Display.GaussianRadius = 6.2266998291015625
    extractSurface1Display.SetScaleArray = ['POINTS', 'scalars_0']
    extractSurface1Display.ScaleTransferFunction = 'PiecewiseFunction'
    extractSurface1Display.OpacityArray = ['POINTS', 'scalars_0']
    extractSurface1Display.OpacityTransferFunction = 'PiecewiseFunction'
    extractSurface1Display.InputVectors = [None, '']

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    extractSurface1Display.OSPRayScaleFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    extractSurface1Display.ScaleTransferFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    extractSurface1Display.OpacityTransferFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(aAA_LG_20080811_MESH_rccvtp, renderView1)

    # show color bar/color legend
    extractSurface1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # create a new 'Smooth'
    smooth1 = Smooth(Input=extractSurface1)

    # show data in view
    smooth1Display = Show(smooth1, renderView1)
    # trace defaults for the display properties.
    smooth1Display.Representation = 'Surface'
    smooth1Display.ColorArrayName = ['POINTS', 'scalars']
    smooth1Display.LookupTable = scalarsLUT
    smooth1Display.OSPRayScaleArray = 'scalars'
    smooth1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    smooth1Display.SelectOrientationVectors = 'None'
    smooth1Display.ScaleFactor = 12.453399658203125
    smooth1Display.SelectScaleArray = 'scalars'
    smooth1Display.GlyphType = 'Arrow'
    smooth1Display.GlyphTableIndexArray = 'scalars'
    smooth1Display.DataAxesGrid = 'GridAxesRepresentation'
    smooth1Display.PolarAxes = 'PolarAxesRepresentation'
    smooth1Display.GaussianRadius = 6.2266998291015625
    smooth1Display.SetScaleArray = ['POINTS', 'scalars_0']
    smooth1Display.ScaleTransferFunction = 'PiecewiseFunction'
    smooth1Display.OpacityArray = ['POINTS', 'scalars_0']
    smooth1Display.OpacityTransferFunction = 'PiecewiseFunction'
    smooth1Display.InputVectors = [None, '']

    # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
    smooth1Display.OSPRayScaleFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    smooth1Display.ScaleTransferFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    smooth1Display.OpacityTransferFunction.Points = [0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

    # hide data in view
    Hide(extractSurface1, renderView1)

    # show color bar/color legend
    smooth1Display.SetScalarBarVisibility(renderView1, True)

    # update the view to ensure updated data information
    renderView1.Update()

    # set active source
    SetActiveSource(aAA_LG_20080811_MESH_rccvtp)

    # set active source
    SetActiveSource(smooth1)

    # save data
    SaveData(outname, proxy=smooth1)

    # set active source
    SetActiveSource(extractSurface1)

    # hide data in view
    Hide(smooth1, renderView1)

    # show data in view
    extractSurface1Display = Show(extractSurface1, renderView1)

    # show color bar/color legend
    extractSurface1Display.SetScalarBarVisibility(renderView1, True)

    # destroy smooth1
    Delete(smooth1)
    del smooth1

    # set active source
    SetActiveSource(aAA_LG_20080811_MESH_rccvtp)

    # hide data in view
    Hide(extractSurface1, renderView1)

    # show data in view
    aAA_LG_20080811_MESH_rccvtpDisplay = Show(aAA_LG_20080811_MESH_rccvtp, renderView1)

    # show color bar/color legend
    aAA_LG_20080811_MESH_rccvtpDisplay.SetScalarBarVisibility(renderView1, True)

    # destroy extractSurface1
    Delete(extractSurface1)
    del extractSurface1

    # destroy aAA_LG_20080811_MESH_rccvtp
    Delete(aAA_LG_20080811_MESH_rccvtp)
    del aAA_LG_20080811_MESH_rccvtp
'''