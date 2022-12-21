from __future__ import with_statement, print_function

import sys
 
from ij import IJ
from ij import WindowManager
 
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import DogDetectorFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter

# We have to do the following to avoid errors with UTF8 chars generated in 
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding('utf-8')
 

day1 = parent + "130616/chiaraM130616/B75B88VH_DocumentFiles/"
filename = "B75B88VH_F00000004.tif"
# Get currently selected image
# imp = WindowManager.getCurrentImage()

# imp = IJ.getImage("foo.tif");
# imp = IJ.selectWindow("foo.tif");
# original = IJ.openImage(day1 + filename)
# original = WindowManager.getCurrentImage()
original = WindowManager.getImage(filename)
original.show()
title = original.getShortTitle()

imp = original.duplicate().crop("1-10")
imp.setTitle("testimage")
imp.show()
# imp = original
# cleanup
# original.close()

width, height, nChannels, nSlices, nFrames = imp.getDimensions()
# nSlices is number of frames in stack
# swap slices and frames
imp.setDimensions(1, 1, nSlices)


#----------------------------
# Create the model object now
#----------------------------
 
# Some of the parameters we configure below need to have
# a reference to the model at creation. So we create an
# empty model now.
 
model = Model()
 
# Send all messages to ImageJ log window.
model.setLogger(Logger.IJ_LOGGER)
 
 
 
#------------------------
# Prepare settings object
#------------------------
 
settings = Settings(imp)
 
# Configure detector - We use the Strings for the keys
settings.detectorFactory = DogDetectorFactory()
settings.detectorSettings = {
    'DO_SUBPIXEL_LOCALIZATION' : True,
    'RADIUS' : 5.0,
    'TARGET_CHANNEL' : 1,
    'THRESHOLD' : 0.,
    'DO_MEDIAN_FILTERING' : True,
}  
 
# Configure spot filters - Classical filter on quality
filter1 = FeatureFilter('QUALITY', 30, True)
# settings.addSpotFilter(filter1)
 
# Configure tracker - We want to allow merges and fusions
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap() # almost good enough
settings.trackerSettings['LINKING_MAX_DISTANCE'] = 3.0
settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 5.0
settings.trackerSettings['MAX_FRAME_GAP'] = 3
# settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = True
# settings.trackerSettings['ALLOW_TRACK_MERGING'] = True
 
# Add ALL the feature analyzers known to TrackMate. They will 
# yield numerical features for the results, such as speed, mean intensity etc.
settings.addAllAnalyzers()
 
# Configure track filters - We want to get rid of the two immobile spots at
# the bottom right of the image. Track displacement must be above 10 pixels.
 
filter2 = FeatureFilter('TRACK_DISPLACEMENT', 5, True)
# settings.addTrackFilter(filter2)
 
settings.initialSpotFilterValue = 0.5
print(str(settings))
 
#-------------------
# Instantiate plugin
#-------------------
 
trackmate = TrackMate(model, settings)
 
#--------
# Process
#--------
 
ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
 
# ok = trackmate.process()  # this does the whole process, detection + tracking
ok = trackmate.execDetection()  # just basic detection
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

# spot filtering    
ok = trackmate.execInitialSpotFiltering()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
 	
ok = trackmate.execSpotFiltering(False)
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))
 
#----------------
# Display results
#----------------

# Echo results with the logger we set at start:
logger = model.getLogger()
logger.log('Found ' + str(model.getTrackModel().nTracks(True)) + ' tracks')

   
# A selection.
selectionModel = SelectionModel( model )
 
# Read the default display settings.
ds = DisplaySettingsIO.readUserDefault()
 
displayer =  HyperStackDisplayer( model, selectionModel, imp, ds )
displayer.render()
displayer.refresh()

# feature model
fm = model.getFeatureModel()

# spots ------------
spots = model.getSpots()
# spots is a fiji.plugin.trackmate.SpotCollection
logger.log(str(spots))
logger.log(str(spots.firstKey()) + " " + str(spots.lastKey()))

logger.log("saving data now ...")

with open(day1 + title + ".csv", "w") as f:
	f.write("spotID,x,y,t\n")
	for frame in range(nSlices):
		for spot in spots.iterable(frame):
			x = spot.getFeature("POSITION_X")
			y = spot.getFeature("POSITION_Y")
			t = spot.getFeature("FRAME")
			sid = spot.ID()
			f.write(",".join([str(sid), str(int(x)), str(int(y)), str(int(t))]) + "\n")



