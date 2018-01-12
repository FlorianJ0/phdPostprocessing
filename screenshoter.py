#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:56:32 2017

@author: florian
"""

#!/usr/bin/env python

import sys
from vmtk import pypes
from vmtk import vmtkscripts
import glob
import pypes

surfList = glob.glob('/media/Windows/postProcess/WSS1/AAA*/*_simu_wss.vtp')
thrombusList = []
for i in surfList:
    thromb = i[:-12]+'MESH_rcc.vtp'
    thrombusList.append(thromb)

a = 'vmtkrenderer --pipe vmtksurfaceviewer -ifile '+surfList[0]+' -opacity 0.5 -color 1.0 0.8 0.8 \
--pipe vmtksurfaceviewer -ifile '+thrombusList[0]

print a

#my = pypes.PypeRun(a)

dual_view = 'vmtksurfacereader -ifile ' +surfList[0]+ \
    ' --pipe vmtkrenderer -background  1 1 1 --pipe vmtksurfaceviewer -display 0  --pipe vmtksurfaceviewer -ifile  '+thrombusList[0]+' -color 1 0 0 -display 1'
my = pypes.PypeRun(dual_view)
