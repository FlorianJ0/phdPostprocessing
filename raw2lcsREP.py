#! /usr/bin/env python
# coding: utf-8
import os
import sys
sys.path.append(
    "/RQexec/jolyflor/visit2_12_3.linux-x86_64/2.12.3/linux-x86_64/lib/site-packages")
sys.path.append(
    "/RQexec/jolyflor/visit2_12_3.linux-x86_64/2.12.3/linux-x86_64/lib")
sys.path.append(
    '/home/apps/Logiciels/Python/numpy/python2.7/1.8.1/lib/python2.7/site-packages')
sys.path.append(
    '/home/apps/Logiciels/Python/scipy/python2.7/0.15.1/lib/python2.7/site-packages')
sys.path.append(
    '/exec5/GROUP/kauffman/jolyflor/virtualpython/lib/python2.7/site-packages')

import scipy.interpolate as interpolate
from vtk import vtkRectilinearGridReader
from vtk.util import numpy_support as VN
import numpy as np
import dicom as dcm
import vtk
from os.path import exists
from pyevtk.hl import imageToVTK

ND = 3
diir = os.getcwd()
rawfilename = diir + '/Output/rep'
fname = diir + '/VTK_recti/U_resamp_0000.vtk'
print '~~~~~~~~~~~~~~~'
print 'WORKING DIR IS ' + diir
print '~~~~~~~~~~~~~~~\n'

if exists(rawfilename) == True:
    filename = rawfilename
    fid = open(filename)
    Compute_Type = np.fromfile(fid, '<i5', 1)
    fid.close()
elif exists(rawfilename) == False:
    filename = rawfilename + '0000.raw'
    if exists(filename) == True:
        Compute_Type = 0
    else:
        print "Could not open %s" % filename

# print filename
fid = open(filename, 'rb')
Compute_Type = np.fromfile(fid, '<i4', 1)
Time_Origin = np.fromfile(fid, '<i4', 6)
FrameTime = np.fromfile(fid, '<d', 1)
Output_TRes = np.fromfile(fid, '<i4', 1)
# print 'Output_TRes'
# print Output_TRes
Atmos_Set = np.fromfile(fid, '<i4', 1)
Atmos_Radius = np.fromfile(fid, '<d', 1)
Slide_Number = np.fromfile(fid, '<i4', 1)
Track_Storm = np.fromfile(fid, '<i4', 1)
ftlemin = np.fromfile(fid, '<d', 3)
ftlemax = np.fromfile(fid, '<d', 3)
FTLE_Res = np.fromfile(fid, '<i4', 3)
LCS_NumFields = np.fromfile(fid, '<i4', 1)
fid.close()

FTLE_BlockSize = np.prod(FTLE_Res)
XYblock = np.prod(FTLE_Res[:2])

if np.size(FTLE_Res) > 2:
    Zblock = np.prod(FTLE_Res[2:])
else:
    Zblock = 1

F = np.zeros((FTLE_Res[1], FTLE_Res[0], Zblock,
              LCS_NumFields[0], Output_TRes[0]))
# print 'F size'
# print F.shape
ss = 1

while 1:

    filename = rawfilename + '%04d.raw' % (ss - 1)
    if exists(filename) == False:
        break
    print filename
    fid = open(filename, 'rb')
    Compute_Type = np.fromfile(fid, '<i4', 1)
    Time_Origin = np.fromfile(fid, '<i4', 6)
    FrameTime = np.fromfile(fid, '<d', 1)
    Output_TRes = np.fromfile(fid, '<i4', 1)
    # print 'Output_TRes'
    # print Output_TRes
    Atmos_Set = np.fromfile(fid, '<i4', 1)
    Atmos_Radius = np.fromfile(fid, '<d', 1)
    Slide_Number = np.fromfile(fid, '<i4', 1)
    Track_Storm = np.fromfile(fid, '<i4', 1)
    ftlemin = np.fromfile(fid, '<d', 3)
    ftlemax = np.fromfile(fid, '<d', 3)
    FTLE_Res = np.fromfile(fid, '<i4', 3)
    LCS_NumFields = np.fromfile(fid, '<i4', 1)
    for nf in range(LCS_NumFields):
        for nb in range(Zblock):
            fdata = np.fromfile(fid, '<d', XYblock).reshape(
                (int(FTLE_Res[0]), int(FTLE_Res[1])), order='F')
            F[:, :, nb, nf, ss - 1] = fdata.T
    ss = ss + 1
    print ss


fid.close()

print F.shape
if ND == 2:
    F = np.squeeze(np.reshape(F, (int(FTLE_Res[1]), int(FTLE_Res[
                   0]), LCS_NumFields, Output_TRes), order='F'))
else:
    F = np.squeeze(np.reshape(F, (int(FTLE_Res[1]), int(FTLE_Res[
                   0]), int(FTLE_Res[2:]), int(LCS_NumFields), int(Output_TRes))))


X = np.linspace(ftlemin[0], ftlemax[0], FTLE_Res[0])
Y = np.linspace(ftlemin[1], ftlemax[1], FTLE_Res[1])
Z = np.linspace(ftlemin[2], ftlemax[2], FTLE_Res[2])


# first time step
nwF = F[:, :, :, :]
FTLE = nwF[:, :, :, 0]  # .T.swapaxes(1, 2)
Omega = nwF[:, :, :, 1]  # .T.swapaxes(1, 2)
Eval = nwF[:, :, :, 2]  # .T.swapaxes(1, 2)
smFTLE = nwF[:, :, :, 3]  # .T.swapaxes(1, 2)


reader = vtkRectilinearGridReader()
reader.SetFileName(fname)
reader.ReadAllScalarsOn()
reader.ReadAllVectorsOn()
reader.Update()
data = reader.GetOutput()
dim = data.GetDimensions()
bounds = data.GetBounds()
arrowGlyph = data.GetPointData().GetArray('U')
vectU = VN.vtk_to_numpy(arrowGlyph)
vectU[vectU < -99] = -1
vectXYZ = vectU
vectX = vectXYZ[:, 0]
# msk = np.reshape(vectX, [dim[0], dim[1], dim[2]], order='F')
# msk[msk > -1] = 1
# msk[msk < 1] = 0

# msk = msk.T
nx, ny, nz = dim
# sample = 2
# Mx, My, Mz = Nx * sample, Ny * sample, Nz * sample

# Interpolation 3D
xmin = bounds[0]
xmax = bounds[1]
ymin = bounds[2]
ymax = bounds[3]
zmin = bounds[4]
zmax = bounds[5]


x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
z = np.linspace(ymin, ymax, nz)

print bounds
print dim
print x.shape
ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)
# pressure = np.random.rand(ncells).reshape( (nx, ny, nz), order = 'C')
# print pressure.shape
print FTLE.shape
# FTLE = Omega[:156,:178,:389]+0.1
print FTLE.shape
# temp = np.random.rand(npoints).reshape((nx + 1, ny + 1, nz + 1))
print np.max(FTLE)

aFTLE = (FTLE/np.max(FTLE)) * (1+Omega/np.max(Omega))

FTLE = FTLE/np.max(FTLE)
FTLE *= 10000
FTLE.astype(int)
FTLE = np.ascontiguousarray(FTLE, dtype=np.int32)
fileName = os.path.split(os.path.splitext(os.getcwd())[0])[1][:-5] + '_ftleREP'
imageToVTK(fileName, origin=(xmin, ymin, zmin), spacing=(
    0.0075, 0.0075, 0.0075), pointData={"FTLE": FTLE})

Omega =	Omega/np.max(Omega)
Omega *= 10000
Omega.astype(int)
Omega = np.ascontiguousarray(Omega, dtype=np.int32)
fileName = os.path.split(os.path.splitext(os.getcwd())[0])[1][:-5] + '_omegaREP'
imageToVTK(fileName, origin=(xmin, ymin, zmin), spacing=(
    0.0075, 0.0075, 0.0075), pointData={"Omega": Omega})

aFTLE *= 10000
aFTLE.astype(int)
aFTLE = np.ascontiguousarray(aFTLE, dtype=np.int32)
fileName = os.path.split(os.path.splitext(os.getcwd())[0])[1][:-5] + '_aFTLEREP'
imageToVTK(fileName, origin=(xmin, ymin, zmin), spacing=(
    0.0075, 0.0075, 0.0075), pointData={"aFTLE": aFTLE})
