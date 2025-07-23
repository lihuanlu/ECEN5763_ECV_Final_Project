/***************************************************************************************
 * @file    cam_plate.cpp
 * @brief   Process pre-recorded video to detect and extract license plate. And then 
 *          use Tesseract OCR engine for character recognition.
 *
 * @author        <Li-Huan Lu>
 * @date          <07/22/2025>
 * @note          Combine cannycam.cpp with sobelcam.cpp and add new features.
 * @reference     OpenCV: Automatic License/Number Plate Recognition (ANPR) with Python
 *                by Adrian Rosebrock
 *                https://tesseract-ocr.github.io/tessdoc/Examples_C++.html 
 ***************************************************************************************/
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
//#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <algorithm>  // for std::sort
#include <vector>
#include <syslog.h>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace std;
using namespace cv;

#define SYSTEM_ERROR (-1)
#define ESCAPE_KEY   (27)
#define FONT         FONT_HERSHEY_SIMPLEX

/**********************************************************************************
 * Public variable     
 **********************************************************************************/
Mat src, src_gray, src_contour;
Mat blackhat, grad_thresh, light;
Mat grad_x, abs_grad_x, norm_grad_x;

vector<vector<Point>> contours;

const char* window_name1 = "Source";
const char* window_name2 = "Cropped";
const char* window_name3 = "Contours";

string outText; // OCR result string

char plate_file[32];
char contour_file[32];
unsigned int frame_count = 0;

/**********************************************************************************
 * Translated from Python to C++ by ChatGPT
 * https://tesseract-ocr.github.io/tessdoc/Examples_C++.html      
 **********************************************************************************/
/**********************************************************************************
 * @name       pruneLicensePlateCandidates()
 *
 * @brief      { License plate candidate selection among contours. }
 *
 * @param[in]  gray           { Grayscale of the source image }
 * @param[in]  contours       { Contours found from previous process }
 * @param[in]  candidates     { To store the potential candidates }
 * @param[in]  minAR          { Minimum aspect ratio for selection }
 * @param[in]  maxAR          { Maximum aspect ratio for selection }
 * 
 * @return     None           
 **********************************************************************************/
void pruneLicensePlateCandidates(const Mat& gray, const vector<vector<Point>>& contours,
                                 vector<RotatedRect>& candidates, float minAR = 2.0f, float maxAR = 6.0f) 
{
    for (const auto& c : contours){
        // Fit a rotated bounding box to the contour
        RotatedRect rr = minAreaRect(c);

        // Extract the width and height
        float w = rr.size.width;
        float h = rr.size.height;

        // Make sure width is the larger side
        float aspectRatio = w > h ? w / h : h / w;

        // Check if aspect ratio is within the range of license plate
        if (aspectRatio >= minAR && aspectRatio <= maxAR){
            // Get rotated bounding box points
            Point2f boxPoints[4];
            rr.points(boxPoints);

            // Crop out the potential plate candidate from the gray image
            Mat M, rotated, cropped;
            float angle = rr.angle;
            Size rect_size = rr.size;

            if (rr.angle < -45.0f){
                angle += 90.0f;
                swap(rect_size.width, rect_size.height);
            }

            // Rotation matrix and affine warp
            M = getRotationMatrix2D(rr.center, angle, 1.0);
            warpAffine(gray, rotated, M, gray.size(), INTER_CUBIC);
            getRectSubPix(rotated, rect_size, rr.center, cropped);

            // Threshold the cropped region to get binary image
            Mat thresh;
            threshold(cropped, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
            imshow(window_name2, thresh);
			
			sprintf(plate_file, "plate_result%04d.jpg", frame_count);
			imwrite(plate_file, thresh);
			
            // Store the rotated rect (candidate)
            candidates.push_back(rr);

            // Optionally: you can store the cropped region as well
            // candidates_images.push_back(cropped);
        }
    }
}

/**********************************************************************************
 * Copied from
 * https://tesseract-ocr.github.io/tessdoc/Examples_C++.html      
 **********************************************************************************/
/**********************************************************************************
 * @name       ocr_process()
 *
 * @brief      { Character recognition using Tesseract OCR engine. }
 *
 * @param[in]  str          { Pointer to cropped license plate file. }
 * 
 * @return     None           
 **********************************************************************************/
string ocr_process(const char* str)
{
	
	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init(NULL, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    // Open input image with leptonica library
    Pix *image = pixRead(str);
    api->SetImage(image);
	api->SetSourceResolution(300); // To remove the DPI warning
    // Get OCR result
    char* rawText = api->GetUTF8Text();
    //printf("OCR output:\n%s", outText);

    // Destroy used object and release memory
    api->End();
    delete api;
	string result(rawText);
    delete [] rawText;	
    pixDestroy(&image);	
	
	return result;
}

/**********************************************************************************
 * @name       preprocess()
 *
 * @brief      { Pre-processing pipeline. }
 *
 * @param[in]  None
 * 
 * @return     None           
 **********************************************************************************/
void preprocess()
{
	double minVal, maxVal;
	
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	
	Mat rectKern = getStructuringElement(MORPH_RECT, Size(13, 5));
	//Mat rectKern = getStructuringElement(MORPH_RECT, Size(20, 5));
	
	// Highlights the license plate numbers against the rest of the photo
	morphologyEx(src_gray, blackhat, MORPH_BLACKHAT, rectKern);
	
	// Find regions in the image that are light
	Mat squareKern = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(src_gray, light, MORPH_CLOSE, squareKern);
	threshold(light, light, 0, 255, THRESH_BINARY | THRESH_OTSU);
	
	
	// Scharr operator (Sobel with ksize = -1)
	// Scharr( src_gray, grad_x, ddepth, 1, 0, -1, delta, BORDER_DEFAULT );
	Sobel(blackhat, grad_x, CV_32F, 1, 0, -1);
	abs_grad_x = abs(grad_x);                 // Take absolute value
	minMaxLoc(abs_grad_x, &minVal, &maxVal);  // Normalize
    // Scale to 0-255
    abs_grad_x.convertTo(norm_grad_x, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // Blur the gradient representation, applying a closing
    GaussianBlur(norm_grad_x, norm_grad_x, Size(5, 5), 0, 0);
	morphologyEx(norm_grad_x, norm_grad_x, MORPH_CLOSE, squareKern);
	threshold(norm_grad_x, grad_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
	
    
	// Remove noise
	// perform a series of erosions and dilations to clean up the thresholded image
	// erode(src, dst, kernel, anchor, interations, borderType, borderValue)
    erode(grad_thresh, grad_thresh, squareKern, Point(-1, -1), 2);
	dilate(grad_thresh, grad_thresh, squareKern, Point(-1, -1), 2);
	
	// take the bitwise AND between the threshold result and the light regions of the image
	bitwise_and(grad_thresh, grad_thresh, grad_thresh, light);
	dilate(grad_thresh, grad_thresh, squareKern, Point(-1, -1), 8);
	//erode(grad_thresh, grad_thresh, squareKern, Point(-1, -1), 1);
	
	
	// Contour
	
	vector<Vec4i> hierarchy;
	// findContours(image, contours, hierarchy, mode, method, offset=Point())
	// retrieves only the extreme outer contours
	// compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    findContours(grad_thresh.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Sort contours by area, descending
    sort(contours.begin(), contours.end(),
    [](const vector<Point>& c1, const vector<Point>& c2) {
        return contourArea(c1, false) > contourArea(c2, false);
    });

    // Keep only the largest N
    int keep = 5;
    if (contours.size() > keep)
        contours.resize(keep);
	
	// Draw the N contours on original image
	src_contour = src.clone();
	for(int i = 0; i < contours.size(); i++ ){
        drawContours(src_contour, contours, i, Scalar(0,255,0), 2, LINE_8, hierarchy, 0);
    }
		
}

/**********************************************************************************
 * Main function        
 **********************************************************************************/
int main( int argc, char** argv )
{
    const char* default_file = "plate_vid.mp4";
	const char* filename;
	VideoCapture cap;
	string new_ocr;
	string prev_ocr;
	unsigned int ocr_cnt = 0;
	Point pt1(10, 40);
	char frame_text[128] = "OCR: ";
	
	if (argc == 2){
        filename = argv[1];
		cap.open(filename);
    }
	else{
	    filename = default_file;
		cap.open(filename);
	}
	
    // Check if file is loaded fine
    if(!cap.isOpened()){
        printf(" Error opening file or device\n");
        printf("Usage: ./cam_plate (using default video)\n");
	    printf("Usage: ./cam_plate ""video_name.mp4""\n");
        exit(SYSTEM_ERROR);
    }
	
    while(1)
    {
		cap.read(src);
		
		if (src.empty()){
			printf("Video ended.\n");
			break; // check if at end
        }
		
		frame_count++;
		
	    // Pre-processing pipeline	
        preprocess();
	
	    // Contour selection
	    vector<RotatedRect> candidatePlates;
	    pruneLicensePlateCandidates(src_gray, contours, candidatePlates);
	    
		// Send the cropped image to Tesseract OCR
	    new_ocr = ocr_process(plate_file);
		
		//if (strcmp(new_ocr, prev_ocr) == 0)
		if (new_ocr == prev_ocr)
		    ocr_cnt++;		
		
		if (ocr_cnt == 1){
			//printf("OCR output:\n%s", new_ocr.c_str());
			syslog(LOG_INFO, "OCR output: %s", new_ocr.c_str());
			sprintf(frame_text, "OCR: %s", new_ocr.c_str());
		    putText(src_contour, frame_text, pt1, FONT, 1, Scalar(0,0,255), 2, LINE_8, false);
			ocr_cnt = 0;
		}
		else{
			putText(src_contour, "OCR:", pt1, FONT, 1, Scalar(0,0,255), 2, LINE_8, false);
		}
		
		prev_ocr = new_ocr;		
		
		sprintf(contour_file, "contour_result%04d.ppm", frame_count);
	    imwrite(contour_file, src_contour);
	
	    imshow(window_name1, src);
		imshow(window_name3, src_contour);
		
	    char ch;
        if (ch = waitKey(1) == ESCAPE_KEY) break;
    }
	
    destroyAllWindows();
	
    return 0;
}
