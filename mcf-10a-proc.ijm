// info
getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
print("It is now " + hour + ":" + minute + ":" + second + ".");

// data path
// var savepath = "/home/fabian/Desktop/";
var data_path = "/media/fabian/Data/fabian/EXP MCF-10A ctrl H2B GFP 72h 1minTIMEFRAME/";

// image path & name
//var image_path = data_path + "130616/chiaraM130616/B75B88VH_DocumentFiles/";
// var image_path = data_path + "140616day2/B75D488Y_DocumentFiles/";
var image_path = data_path + "150616 day3/B75EZ8L6_DocumentFiles/";

//var image_name = "base.tif";
//var image_name = "B75B88VH_F00000004.tif"
//var image_name =  "B75D488Y_F00000004.tif"
var image_name = "B75EZ8L6_F00000004.tif"
var n2v_image_name = image_name + "-n2v"

// setup
setBatchMode("hide");
var start = 1;

// open image
// open(savepath + image_name);
open(image_path + image_name)
var img_title = File.nameWithoutExtension;

// multipage tiff info
getDimensions(width, height, channels, slices, frames);
print("The multipage tiff has " + slices + " frames.");

var slice_title = "";

// setup savedir
var savepath = image_path + img_title + "-analysis/stardist/";
File.makeDirectory(savepath);

for (i = start; i <= slices; i++) {
	selectWindow(image_name);
	setSlice(i);
	slice_title = img_title + "_" + IJ.pad(i, lengthOf("" + slices));
	run("Duplicate...", "title=" + slice_title);

	// noise 2 void
	run("N2V predict", 
		"modelfile=[" + data_path + "/N2V_model/n2v-7051892842688668933.bioimage.io.zip]"
		+ " input=" + slice_title // image_name
		+ " axes=XY"
		+ " batchsize=10"
		+ " numtiles=1"
		+ " showprogressdialog=false"
		+ " convertoutputtoinputformat=true"
	);

	// ------
	/*
	list = getList("image.titles");
	for(j=0; j<list.length; j++) {
		print("  " + list[j]);
	}
	list = getList("window.titles");
	for(j=0; j<list.length; j++) {
		print("  " + list[j]);
	}
	*/
	// -------
	
	// select output image and rename it
	selectWindow("output");
	
	// rename(n2v_image_name);
	n2v_image_name = "n2v-" + slice_title;
	rename(n2v_image_name);
	
	// upscale image 
	// we can leave the dimensions fixed since they are all the same for all MCF-10A images
	run("Scale...", "x=4 y=4 width=2688 height=2048 interpolation=Bicubic create");
	var upscaled_image = n2v_image_name + "-1";

	// run stardist
	var probThresh = 0.479071;  // default
	var nmsThresh = 0.3;  // default
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

	// select label image 
	var label_img_name = "labels-" + slice_title;
	selectWindow("Label Image");
	rename(label_img_name);
	run("8-bit");
	saveAs("Tiff", savepath + label_img_name);
	close(label_img_name);

	// open results as list and save it as a file
	roiManager("List");
	saveAs("Results", savepath + "results-" + slice_title + ".csv");

	// cleanup
	close("results-" + slice_title + ".csv");  // Close the results window
	close(label_img_name);
	close(upscaled_image);
	close(n2v_image_name);
	close(slice_title);
	run("Collect Garbage");
}

getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
print("Finished at " + hour + ":" + minute + ":" + second + ".");
/*
// move initial image to front and close everything else
selectWindow(image_name);
close("\\Others");  // close all other images
close("ROI Manager");  // close the ROI Manager
close("results-" + img_title + ".csv");  // Close the results window
*/
