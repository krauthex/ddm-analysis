# from __future__ import with_statement, print_function

import sys
import os

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


reload(sys)
sys.setdefaultencoding('utf-8')

def get_features(spot):
	"""Returns a list of features: ID, X, Y, Frame, Radius, Quality as strings."""
	
	feature_keys = ["POSITION_X", "POSITION_Y", "FRAME", "RADIUS", "QUALITY"]
	features = []
	for key in feature_keys:
		feature = spot.getFeature(key)
		if key == "FRAME":
			feature = int(feature)
		features.append(str(feature))
		
	sid = str(spot.ID())
	
	return [sid] + features 	
	


def process_single_batch(im, slices, storepath):
	
	# image setup 
	imp = im.duplicate().crop(slices)
	imp.setTitle("batch_" + slices)
	imp.show()
	
	_, _, _, _, nFrames = imp.getDimensions()
	
	# model setup
	model = Model()
	model.setLogger(Logger.IJ_LOGGER)
	
	# settings
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
	
	settings.trackerFactory = SparseLAPTrackerFactory()
	settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap() # almost good enough
	settings.trackerSettings['LINKING_MAX_DISTANCE'] = 3.0
	settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 5.0
	settings.trackerSettings['MAX_FRAME_GAP'] = 3
	
	settings.addAllAnalyzers()
	settings.initialSpotFilterValue = 0.5
	
	# plugin
	trackmate = TrackMate(model, settings)
	
	# Process
	 
	ok = trackmate.checkInput()
	if not ok:
	    sys.exit(str(trackmate.getErrorMessage()))
	 
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
	
	with open(storepath + "part_" + slices + ".csv", "w") as f:
		f.write("ID,x,y,t,r,quality\n")
		for spot in spots.iterable(False):
			sid, x, y, t, r, q = get_features(spot)
			# x = spot.getFeature("POSITION_X")
			# y = spot.getFeature("POSITION_Y")
			# t = spot.getFeature("FRAME")
			# sid = spot.ID()
			f.write(",".join([sid, x, y, t, r, q]) + "\n")
	
	# cleanup
	imp.close()
	del model, fm, trackmate, displayer, selectionModel, logger
	print("finished batch " + slices)
	return True


day1 = parent + "130616/chiaraM130616/B75B88VH_DocumentFiles/"

files = [
	"B75B88VH_F00000004.tif",
	"B75B88VH_F00000006.tif",
	"B75B88VH_F00000008.tif",
	"B75B88VH_F00000010.tif",
	"B75B88VH_F00000012.tif",
	"B75B88VH_F00000014.tif",
	"B75B88VH_F00000016.tif",
	"B75B88VH_F00000018.tif",
	"B75B88VH_F00000020.tif",
	"B75B88VH_F00000022.tif"
]
filename = files[0]

original = IJ.openImage(day1 + filename)
original.show()
title = original.getShortTitle()

width, height, nChannels, nSlices, nFrames = original.getDimensions()
# nSlices is number of frames in stack
# swap slices and frames
original.setDimensions(1, 1, nSlices)

# setup of batches for processing
batches = 3
batchsize = nSlices // batches

# crop slices 
crops = []
for i in range(batches):
	crops.append(str(i*batchsize + 1) + "-" + str(batchsize*(i+1)))

store = day1 + filename.replace(".tif", "-analysis/positions/")
if not os.path.exists(store):
	os.mkdir(store)

for crop in crops:
	process_single_batch(original, crop, store)

