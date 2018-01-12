# import the simple module from the paraview
from paraview.simple import *
# disable automatic camera reset on 'Show'
fname = 'filename'
outname = fname[:-1]+'u'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
aAA_LM_20130524_simu_wssvtk = LegacyVTKReader(FileNames=[fname])

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1300, 949]

# show data in view
tataDisplay = Show(
    aAA_LM_20130524_simu_wssvtk, renderView1)
# trace defaults for the display properties.
tataDisplay.Representation = 'Surface'
tataDisplay.ColorArrayName = [None, '']
tataDisplay.OSPRayScaleArray = 'Divergence_average'
tataDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
tataDisplay.SelectOrientationVectors = 'Divergence_average'
tataDisplay.ScaleFactor = 0.02827631235122681
tataDisplay.SelectScaleArray = 'Divergence_average'
tataDisplay.GlyphType = 'Arrow'
tataDisplay.GlyphTableIndexArray = 'Divergence_average'
tataDisplay.DataAxesGrid = 'GridAxesRepresentation'
tataDisplay.PolarAxes = 'PolarAxesRepresentation'
tataDisplay.ScalarOpacityUnitDistance = 0.009500978122699271
tataDisplay.GaussianRadius = 0.014138156175613405
tataDisplay.SetScaleArray = [
    'POINTS', 'Divergence_average']
tataDisplay.ScaleTransferFunction = 'PiecewiseFunction'
tataDisplay.OpacityArray = [
    'POINTS', 'Divergence_average']
tataDisplay.OpacityTransferFunction = 'PiecewiseFunction'
tataDisplay.InputVectors = [None, '']

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
tataDisplay.OSPRayScaleFunction.Points = [
    0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
tataDisplay.ScaleTransferFunction.Points = [
    0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
tataDisplay.OpacityTransferFunction.Points = [
    0.03772579878568649, 0.0, 0.5, 0.0, 9.976140022277832, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()

# update the view to ensure updated data information
renderView1.Update()

# save data
SaveData(outname,
         proxy=aAA_LM_20130524_simu_wssvtk)
exit()