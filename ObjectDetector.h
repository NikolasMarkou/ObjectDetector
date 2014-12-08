#ifndef EXTRACT_OBJECTS_H
#define EXTRACT_OBJECTS_H

#include <list>
#include <argp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits.h>
#include <vector>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#define MAX_INPUT_DETECTORS   128
#define MAX_INPUT_DIRECTORIES 128
#define MAX_INPUT_FILES       128
#define DEFAULT_DETECTOR_PATH "cascades/haarcascade_profileface.xml"

#define REPORT(condition,message){if(condition) {std::cout<<message;} }
#define VERBOSE(message){if(arguments.verbose){std::cout<<message;} }

/*
 * Supported image files
 *
 * Windows bitmaps - *.bmp, *.dib (always supported)
 * JPEG files - *.jpeg, *.jpg, *.jpe (see the Notes section)
 * JPEG 2000 files - *.jp2 (see the Notes section)
 * Portable Network Graphics - *.png (see the Notes section)
 * Portable image format - *.pbm, *.pgm, *.ppm (always supported)
 * Sun rasters - *.sr, *.ras (always supported)
 * TIFF files - *.tiff, *.tif (see the Notes section)
 */

static std::string imageExtensions[] = {
    "bmp", "dib", "jpeg", "jpg", "jpe", "jp2", "png", "pdm", "pgm", "ppm", "sr", "ras", "tiff", "tif", ""
};

/*
 * Supported video files
 *
 * AVI files - *.avi
 */

static std::string videoExtensions[] = {
    "avi", "mp4", ""
};

enum FILE_TYPE { VIDEO, IMAGE, OTHER };

bool fileExists (const std::string& path); 
FILE_TYPE identifyFileType(const std::string& path);
std::string removeExtension(const std::string& filename);
template <typename T> std::string numberToString ( T Number );
std::vector<cv::Rect> detectAndExtract(cv::Mat imageSrc, cv::CascadeClassifier detector);

/*
 * Return the file type of the given path
 */
FILE_TYPE identifyFileType(const std::string& path)
{
    if (path.find_last_of(".") != std::string::npos)
    {
	int iterator = 0;
	std::string extension = path.substr(path.find_last_of(".") + 1);
	
	while (imageExtensions[iterator].length() > 0)
	{
	    if (imageExtensions[iterator].compare(extension) == 0)
	    {
		return IMAGE; 
	    }   
	    iterator++;
	}		

	iterator = 0;

	while (videoExtensions[iterator].length() > 0)
	{
	    if (videoExtensions[iterator].compare(extension) == 0)
	    {
		return VIDEO;
	    }
	    iterator++;   
	}
	
    }

    return OTHER;
}

/*
 * Remove filename extension
 */
std::string removeExtension(const std::string& filename)
{
    std::string tmpFilename(filename);
    int lastDotCharacter = tmpFilename.find_last_of(".");

    if (lastDotCharacter == std::string::npos)
    {
	return tmpFilename;
    }

    std::string cleanTmpFilename = tmpFilename.substr(0, lastDotCharacter);

    return cleanTmpFilename;
}

/*
 * Returns true if the filename with the 
 * given path exists and false otherwise
 */
bool fileExists (const std::string& path) 
{
    struct stat buffer = {0};  
    return (stat(path.c_str(), &buffer) == 0); 
}

/*
 * Detect object in the input image and return a list of rectangle detections
 */ 
std::vector<cv::Rect> detectAndExtract(cv::Mat imageSrc, cv::CascadeClassifier detector)
{
    cv::Mat imageSrcGray;
    std::vector<cv::Rect> detections;
    /* 
     * Convert to grayscale and equalize it using
     * histogram equalization
     */  
    cv::cvtColor(imageSrc, imageSrcGray, CV_BGR2GRAY);
    cv::equalizeHist(imageSrcGray, imageSrcGray);

    detector.detectMultiScale(imageSrcGray, detections, 1.1, 3, CV_HAAR_SCALE_IMAGE, cv::Size(50, 50));
    
    return detections;
}


/*
 * Iterate through detections, extract cropped images
 * and save them
 */
bool extractAndSaveDetections(cv::Mat imageSrc, std::vector<cv::Rect> detections, std::string& directory, std::string& prefix, std::string& filename, std::string& extension, int frame = 0)
{
    for (int iter = 0 ; iter < detections.size() ; iter++)
    {
	cv::Mat imageSrcCrop = imageSrc(detections[iter]);
	std::string stringIter = numberToString(iter);
	std::string stringFrame = numberToString(frame);
	std::string filenameCrop = directory;
	if (prefix.length() > 0)
	{
	    filenameCrop += prefix + "_";
	}
	filenameCrop += filename + "_" + stringFrame + "_" + stringIter + extension;
	bool success = cv::imwrite(filenameCrop, imageSrcCrop);
	
	if (success == false)
	{
	    std::cerr << "Failed to write " << filenameCrop << std::endl;
	}
    }

    return true;
}

/* Convert any kind of number to string */
template <typename T> std::string numberToString ( T Number )
{
    std::ostringstream ss;
    ss << Number;
    return ss.str();
}

#endif
