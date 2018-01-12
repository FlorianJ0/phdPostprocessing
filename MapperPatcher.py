#!/usr/bin/env python 
import sys 
import math 
import string 

import vtkvmtk 

from vmtk import pypes 
from vmtk import vmtkscripts 
from vmtk import pypescript
from pypescript import pypeScript

class MapperPatcher(object):

	def __init__(self):
		self.SourcePath = " "
		self.OutputFolderPath = " " 
		self.VmtkCommand = " "
	
	def inputPath(self, obj):
		StringuPIZDIIMATII = raw_input("Enter the path and filename for the "+obj+" file:\n")
		copy2 = StringuPIZDIIMATII
		while copy2[-1:] != "\\":
			copy2 = copy2[:-1]
		self.OutputFolderPath = copy2
		return StringuPIZDIIMATII
	
		
	def mapperAndPatcher(self):
		#WHAT IS THE INPUT OBJECT - TO GET ITS PATH
		object = "surface"
		#ENTERING THE PATH
		self.SourcePath = MapperPatcher.inputPath(self, object)
		#FIRST STEP - GENERATING THE CENTERLINES WITH SOME ATTRIBUTES
		#THE VMTK COMMAND
		self.VmtkCommand = "vmtkcenterlines -ifile "+self.SourcePath+" --pipe vmtkcenterlineattributes --pipe vmtkbranchextractor -ofile "+self.OutputFolderPath+"Centerlines.vtp --pipe vmtksurfacewriter -i @vmtkbranchextractor.o -ofile "+self.OutputFolderPath+"Centerlines.dat"
		#RUNNING THE VMTK COMMAND
		myPype = pypes.PypeRun(self.VmtkCommand)
		#SECOND STEP - GENERATING THE REFERENCE SYSTEMS ALONG THE CENTERLINE NETWORK 
		#THE VMTK COMMAND
		self.VmtkCommand = "vmtkbifurcationreferencesystems -ifile "+self.OutputFolderPath+"Centerlines.vtp -radiusarray MaximumInscribedSphereRadius -blankingarray Blanking -groupidsarray GroupIds -ofile "+self.OutputFolderPath+"BifurcationReferenceSystemMapperPatcher.vtp --pipe vmtksurfacewriter -i @vmtkbifurcationreferencesystems.o -ofile "+self.OutputFolderPath+"BifurcationReferenceSystemMapperPatcher.dat"
		#RUNNING THE VMTK COMMAND
		myPype = pypes.PypeRun(self.VmtkCommand)
		#THIRD STEP - SUBDIVIDING THE SURFACE INTO ITS BRANCHES TO MAPP AND PATCH THE INFO DIRECT ON EACH BRANCH
		#THE VMTK COMMAND
		self.VmtkCommand = "vmtkbranchclipper -ifile "+self.SourcePath+"  -centerlinesfile "+self.OutputFolderPath+"Centerlines.vtp -groupidsarray GroupIds -radiusarray MaximumInscribedSphereRadius -blankingarray Blanking -ofile "+self.OutputFolderPath+"SubdividedSurface.vtp --pipe vmtksurfacewriter -i @vmtkbranchclipper.o -ofile "+self.OutputFolderPath+"SubdividedSurface.dat"
		#RUNNING THE VMTK COMMAND
		myPype = pypes.PypeRun(self.VmtkCommand)
		#FOURTH STEP - GENERATING THE LONGITUDINAL AND CIRCUMFERENTIAL METRICS
		#THE VMTK COMMAND
		self.VmtkCommand = "vmtkbranchmetrics -ifile "+self.OutputFolderPath+"SubdividedSurface.vtp -centerlinesfile "+self.OutputFolderPath+"Centerlines.vtp -abscissasarray Abscissas -normalsarray ParallelTransportNormals -groupidsarray GroupIds -centerlineidsarray CenterlineIds -tractidsarray TractIds -blankingarray Blanking -radiusarray MaximumInscribedSphereRadius -ofile "+self.OutputFolderPath+"SurfaceSubdividedMetrics.vtp --pipe vmtksurfacewriter -i @vmtkbranchmetrics.o -ofile "+self.OutputFolderPath+"SurfaceSubdividedMetrics.dat"
		#RUNNING THE VMTK COMMAND
		myPype = pypes.PypeRun(self.VmtkCommand)
		#FIFTH STEP - MAPPING OF THE METRICS ON THE BRANCHES
		#THE VMTK COMMAND
		self.VmtkCommand = "vmtkbranchmapping -ifile "+self.OutputFolderPath+"SurfaceSubdividedMetrics.vtp -centerlinesfile "+self.OutputFolderPath+"Centerlines.vtp -referencesystemsfile "+self.OutputFolderPath+"BifurcationReferenceSystemMapperPatcher.vtp -normalsarray ParallelTransportNormals -abscissasarray Abscissas -groupidsarray GroupIds -centerlineidsarray CenterlineIds -tractidsarray TractIds -referencesystemsnormalarray Normal -radiusarray MaximumInscribedSphereRadius -blankingarray Blanking -angularmetricarray AngularMetric -abscissametricarray AbscissaMetric -ofile "+self.OutputFolderPath+"Mapping.vtp --pipe vmtksurfacewriter -i @vmtkbranchmapping.o -ofile "+self.OutputFolderPath+"Mapping.dat"
		#RUNNING THE VMTK COMMAND
		myPype = pypes.PypeRun(self.VmtkCommand)
		#ENTERING TWO PARAMETERS FOR THE SIXTH STEP - THE SIZE IN MILIMETERS OF EACH PATCH AND THE NUMBER OF SECTORS BETWEEN (-PI,+PI)
		longitudinalPatchSize = raw_input("Please enter the size of the longitudinal patch(mm): \n")
		numberOfCircularPatches = raw_input("Please enter how many patches you wish to have on: \n")
		#SIXTH STEP - PATCHING THE INFORMATION ON THE SURFACE
		#THE VMTK COMMAND
		self.VmtkCommand = "vmtkbranchpatching -ifile "+self.OutputFolderPath+"Mapping.vtp -groupidsarray GroupIds -longitudinalmappingarray StretchedMapping -circularmappingarray AngularMetric -longitudinalpatchsize "+longitudinalPatchSize+" -circularpatches "+numberOfCircularPatches+" -ofile "+self.OutputFolderPath+"Patching.vtp --pipe vmtksurfacewriter -i @vmtkbranchpatching.o -ofile "+self.OutputFolderPath+"Patching.dat"
		#RUNNING THE VMTK COMMAND
		myPype = pypes.PypeRun(self.VmtkCommand)

def main():
	
	endProgram = " "
	while (endProgram != "0") :
		Option = ""
		while Option != "1" and Option != "0": 
			Option = raw_input("1-Mapping and Patching\n0-Exit\n")
		
		if Option == "1":
			ga = MapperPatcher()
			MapperPatcher.mapperAndPatcher(ga)
		if Option =="0":
			endProgram = "0"
			
if __name__  =='__main__':
	main()
		