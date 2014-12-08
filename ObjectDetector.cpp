#include <list>
#include <argp.h>
#include <string.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <dirent.h>
#include <limits.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ObjectDetector.h"

using namespace cv;

const char *argp_program_version = "extract-objects 0.01";

const char *argp_program_bug_address = "<nikolasmarkou@gmail.com>";

/* A description of the arguments we accept. */
static char args_doc[] = "ARG1 ARG2";

static char doc[] = "object-detector is utility for detecting extracting objects from images and videos";

/* Used by main to communicate with parse_opt. */
struct arguments
{
    int output;
    int verbose;
    int recursive;
    int noFiles; 
    int noDetectors;
    int minDetections;
    int noDirectories;
    char prefix[PATH_MAX]; 
    char detector[PATH_MAX];
    char extension[PATH_MAX];
    char outputDirectory[PATH_MAX];
    char files[MAX_INPUT_FILES][PATH_MAX]; 
    char directories[MAX_INPUT_DIRECTORIES][PATH_MAX];
};

/* The options we understand. */
static struct argp_option options[] = 
{
    {"max-detections",   'm', "DETECTIONS",0, "Min detections before reporting" },
    {"output"   ,        'o', 0,           0, "Write output files" },        
    {"verbose"  ,        'v', 0,           0, "Produce verbose output" },
    {"recursive",        'r', 0,           0, "Recursively browse the directories" },    
    {"detector" ,        'x', "DETECTOR",  0, "Object extractor file" },
    {"file"     ,        'f', "FILE",      0, "Single input file"},
    {"directory",        'd', "DIRECTORY", 0, "Single input directory"},
    {"output-file",      'y', "FILE",      0, "File to write the detection output"},
    {"output-directory", 'u', "DIRECTORY", 0, "Directory to output files"},
    {"prefix",           'p', "PREFIX",    0, "Prefix to add to output files"},
    {"extension",        'e', "EXTENSION", 0, "Extension of the detection output files"},
    { 0 }
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
    /* Get the input argument from argp_parse, which we
     * know is a pointer to our arguments structure. 
     */
    struct arguments *arguments = (struct arguments*)state->input;

    switch (key)
    {
	case 'o':
	{
	    arguments->output = 1;
	    break;
	}
	case 'm':
	{
	    int tmp = 0;
	    tmp = atoi(arg);
	    if (tmp > 0)
	    {
		arguments->minDetections = tmp;
	    }
	    else
	    {
		argp_usage (state);
	    }
	    break;
	}
	case 'v': 
	{
	    arguments->verbose = 1;
	    break;
	}
	case 'r':
	{
	    arguments->recursive = 1;
	    break;
	}
	case 'u':
	{
	    strncpy(arguments->outputDirectory, arg, PATH_MAX);    
	    break;
	}
	case 'e':
	{
	    strncpy(arguments->extension, arg, PATH_MAX);
	    break;
	}
	case 'x':
	{
	    strncpy(arguments->detector, arg, PATH_MAX);
	    break;
	}
	case 'p':
	{
	    strncpy(arguments->prefix, arg, PATH_MAX);
	    break;
	}
	case 'f':
	{
	    if (arguments->noFiles >= MAX_INPUT_FILES)
	    {
		/* Too many arguments. */
		argp_usage (state);
	    }
	    strncpy(arguments->files[arguments->noFiles], arg, PATH_MAX);
	    arguments->noFiles++;
	    break;
	}
	case 'd':
	{
	    if (arguments->noDirectories >= MAX_INPUT_DIRECTORIES)
	    {
		/* Too many arguments. */
		argp_usage (state);
	    }
	    strncpy(arguments->directories[arguments->noDirectories], arg, PATH_MAX);
	    arguments->noDirectories++;
	    break;
	}
	default:
	{
	    return ARGP_ERR_UNKNOWN;
	}
    }

    return 0;
}

/* Argument parser */
static struct argp argp = { options, parse_opt, args_doc, doc };

int main(int argc, char** argv )
{
    int iterator = 0; 
    struct arguments arguments = {0};
    CascadeClassifier detector;
    
    /* fill in default arguments */
    arguments.output = 0;
    arguments.verbose = 0;
    strncpy(arguments.extension,".jpg", PATH_MAX);
    if (getcwd(arguments.outputDirectory, PATH_MAX) == NULL)
    {
	strncpy(arguments.outputDirectory, ".", PATH_MAX);
    }
    argp_parse (&argp, argc, argv, 0, 0, &arguments);
    
    /* check if output directory exists */
    std::string outputDirectory(arguments.outputDirectory);
    outputDirectory += "/";
    if (!fileExists(outputDirectory))
    {
	std::cerr << "[" << outputDirectory << "]" <<":Output directory does not exist" << std::endl;
	exit(-1);
    }

    /* prefix to add to all output files */
    std::string prefix(arguments.prefix);
    
    /* extension of all the detection files */
    std::string extension(arguments.extension);
 
    /* if no detectors are entered add default face detector */
    if (strlen(arguments.detector) == 0)
    {
	strncpy(arguments.detector, DEFAULT_DETECTOR_PATH, PATH_MAX);
	VERBOSE("No custom detector added, using default detector " << DEFAULT_DETECTOR_PATH << std::endl)
    }

    std::cout << "Verbose : " << arguments.verbose << std::endl;	
    std::cout << "Recursive : " << arguments.recursive << std::endl;	
    std::cout << "Number of files : " << arguments.noFiles << std::endl;	
    std::cout << "Number of directories : " << arguments.noDirectories << std::endl;	
    std::cout << "Detector : " << arguments.detector << std::endl; 
    std::cout << "Prefix of output files : "<< arguments.prefix << std::endl;
    
    /* Iterate through detectors and initialize detectors */
    std::string detectorFilename(arguments.detector);

    /* check if detector file exists and can be correctly read */
    if (fileExists(detectorFilename) == false) 
    {
	std::cerr << "[" << detectorFilename << "]:" << "Detector file does not exist" <<  std::endl;  
	exit(-1);
    } 
	
    if (!detector.load(detectorFilename))
    {
	std::cerr << "[" << detectorFilename << "]:" << "Cannot load detector file" << std::endl;  
	exit(-1);
    }
 
    VERBOSE("[" << detectorFilename << "]:" << "Correctly loaded detector" << std::endl)
    
    /* Iterate through input files and extract items */
    for (iterator = 0; iterator < arguments.noFiles ; iterator++)
    {
	char *baseFilename = NULL;
	baseFilename = dirname(arguments.files[iterator]);
	std::string baseFilenameStr(baseFilename);
	std::string filename(arguments.files[iterator]);	
	FILE_TYPE filetype = identifyFileType(filename);

	if ( filetype == IMAGE )
	{
	    Mat matSrc = imread(filename.c_str());
	
	    if (!matSrc.data)
	    {
		std::cerr << "[" << filename << "]:" << "cannot load image file " << std::endl;
		continue;
	    }
	    
	    VERBOSE("[" << filename << "]:" << "processing image file "  << std::endl)
	    
	    std::vector<Rect> detections = detectAndExtract(matSrc, detector);
	    
	    if (detections.size() > 0)
	    {
		VERBOSE("[" << filename << "]:" << detections.size() << " detection(s)" << std::endl)
		if (arguments.output)
		{
		    extractAndSaveDetections(matSrc, detections, prefix, outputDirectory, baseFilenameStr, extension);
		}
	    }
	}
	else if ( filetype == VIDEO )
	{
	    int frame = 0;
	    VideoCapture video;
	    bool readSuccessful = true;

	    if (!video.open(filename))
	    {
		std::cerr << "[" << filename << "]:" << "cannot open video file " << std::endl;
		continue;
	    }

	    VERBOSE("[" << filename << "]:" << "processing video file" << std::endl)
	    
	    /* read each frame and extract objects */
	    while (readSuccessful)
	    {
		cv::Mat matSrc;
		readSuccessful = video.read(matSrc);
		
		if (!readSuccessful || !matSrc.data)
		{
		    VERBOSE("[" << filename << "]:" << "done processing video file" << std::endl)		    
		    continue;   
		}
		
		std::vector<cv::Rect> detections = detectAndExtract(matSrc, detector);
		if (detections.size() > 0)
		{
	    
		    VERBOSE("[" << filename << "]:" << "["<< numberToString(frame) <<"]:" << detections.size() << " detection(s)" <<std::endl)
		    
		    if (arguments.output)
		    {
			extractAndSaveDetections(matSrc, detections, outputDirectory, prefix, filename, extension, frame);
		    }
		}	
		
		frame += 1;	
		
	    }
	}
    }
 
    /* Iterate through directories and for each file in the extract items */
    for (iterator = 0; iterator < arguments.noDirectories ; iterator++)
    {

    }    
      
    exit(0);
}
