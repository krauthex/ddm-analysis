// data path
var savepath = "/home/fkra/Desktop/"

// image name
var image_name = "base.tif";
var n2v_image_name = image_name + "-n2v"

// open image
open(savepath + image_name);
var img_title = File.nameWithoutExtension;

// noise 2 void
run("N2V predict", 
	"modelfile=[" + data_path + "/N2V_model/n2v-7051892842688668933.bioimage.io.zip]"
	+ " input=" + image_name
	+ " axes=XY"
	+ " batchsize=10"
	+ " numtiles=1"
	+ " showprogressdialog=true"
	+ " convertoutputtoinputformat=true"
);

// select output image and rename it
selectWindow("output");
rename(n2v_image_name);

// upscale image 
// we can leave the dimensions fixed since they are all the same for all MCF-10A images
run("Scale...", "x=4 y=4 width=2688 height=2048 interpolation=Bicubic create");
var upscaled_image = n2v_image_name + "-1"

// run stardist
var probThresh = 0.479071  // default
var nmsThresh = 0.3  // default
run("Command From Macro", 
	"command=[de.csbdresden.stardist.StarDist2D],"
	+ "args=['input':'" + upscaled_image + "', "
		+ "'modelChoice':'Versatile (fluorescent nuclei)', "
		+ "'normalizeInput':'true', "
		+ "'percentileBottom':'1.0', "
		+ "'percentileTop':'99.8', "
		+ "'probThresh':'" + probThresh + "', "
		+ "'nmsThresh':'" + nmsThresh + "', "
		+ "'outputType':'Both', "
		+ "'nTiles':'1', "
		+ "'excludeBoundary':'2', "
		+ "'roiPosition':'Automatic', "
		+ "'verbose':'false', "
		+ "'showCsbdeepProgress':'false', "
		+ "'showProbAndDist':'false'], "
	+ "process=[false]"
);


// open results as list and save it as a file
roiManager("List");
saveAs("Results", savepath + "results-" + img_title + ".csv");

// move initial image to front and close everything else
selectWindow(image_name);
close("\\Others");  // close all other images
close("ROI Manager");  // close the ROI Manager
close("results-" + img_title + ".csv");  // Close the results window
