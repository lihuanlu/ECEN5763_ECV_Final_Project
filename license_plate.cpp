/***************************************************************************************
 * @file    license_plate.cpp
 * @brief   Process a single image to detect and extract license plate. And then uses
 *          Tesseract OCR engine for character recognition.
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
#include <opencv2/opencv.hpp>
#include <algorithm>  // for std::sort
#include <vector>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace std;
using namespace cv;

#define SYSTEM_ERROR (-1)
#define ESCAPE_KEY   (27)

/**********************************************************************************
 * Public variable     
 **********************************************************************************/
Mat src, src_gray;
Mat blackhat, grad_thresh, light;
Mat grad_x, abs_grad_x, norm_grad_x;
Mat plate_cropped, plate_thresh;
//Testing 
Mat blackhat1, blackhat2;

vector<vector<Point>> contours;

bool plate_detected = false;

const char* window_name1 = "Source";
const char* window_name2 = "Result";
const char* filename;

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
		printf("width = %f, height = %f\n", w, h);
		printf("AR = %f\n", aspectRatio);
		
        // Get bounding box points 
	    Point2f boxPoints[4];
        rr.points(boxPoints);
            
	    for (int i = 0; i < 4; ++i){
            line(src, boxPoints[i], boxPoints[(i + 1) % 4], Scalar(0, 0, 255), 2);
		    printf("points (%f, %f) to (%f, %f)\n",
                    boxPoints[i].x, boxPoints[i].y,
                    boxPoints[(i + 1) % 4].x, boxPoints[(i + 1) % 4].y);
	    }	
		
        // Check if aspect ratio is within the range of license plate
        if (aspectRatio >= minAR && aspectRatio <= maxAR){      
            // Crop out the potential plate candidate from the gray image
            Mat M, rotated, cropped;
            float angle = rr.angle;
            Size rect_size = rr.size;
            printf("Angle = %f\n", angle);
            if (rr.angle < -45.0f){
                angle += 90.0f;
                swap(rect_size.width, rect_size.height);
            }
            
			// Check if the center is around the center of the original image
			if (rr.center.x > src.cols / 3 && rr.center.x < src.cols * 2 / 3){
                plate_detected = true;
				
				// Rotation matrix and affine warp
                M = getRotationMatrix2D(rr.center, angle, 1.0);
                warpAffine(gray, rotated, M, gray.size(), INTER_CUBIC);
                getRectSubPix(rotated, rect_size, rr.center, plate_cropped);
/*
                // Threshold the cropped region to get binary image
                Mat thresh;
                threshold(cropped, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
                imshow(window_name2, thresh);
			    imwrite("plate_result.jpg", thresh);*/
				break;
			}
			else{
				printf("Not a candidate.\n");
				plate_detected = false;
			}
            // Store the rotated rect (candidate)
            candidates.push_back(rr);
        }

		printf("\n");
    }
}

void LicensePlate_postprocess()
{
	// Threshold the cropped region to get binary image
    
    threshold(plate_cropped, plate_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
    imshow(window_name2, plate_thresh);
	imwrite("plate_result.jpg", plate_thresh);
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
void ocr_process(const char* str)
{
    char *outText;
	
	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    // Initialize tesseract-ocr with English, without specifying tessdata path
    if (api->Init(NULL, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }

    // Open input image with leptonica library
    Pix *image = pixRead(str);
    api->SetImage(image);
    // Get OCR result
    outText = api->GetUTF8Text();
    printf("OCR output:\n%s", outText);

    // Destroy used object and release memory
    api->End();
    delete api;
    delete [] outText;
    pixDestroy(&image);	
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
	
	//Mat rectKern = getStructuringElement(MORPH_RECT, Size(13, 5));
	Mat rectKern = getStructuringElement(MORPH_RECT, Size(20, 10));
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
	//Sobel(light, grad_x, CV_32F, 1, 0, -1);
	abs_grad_x = abs(grad_x);                 // Take absolute value
	minMaxLoc(abs_grad_x, &minVal, &maxVal);  // Normalize
    // Scale to 0-255
    abs_grad_x.convertTo(norm_grad_x, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    // Blur the gradient representation, applying a closing
    GaussianBlur(norm_grad_x, norm_grad_x, Size(5, 5), 0, 0);
	morphologyEx(norm_grad_x, norm_grad_x, MORPH_CLOSE, squareKern);
	
	//test
	//Mat crossKern = getStructuringElement(MORPH_CROSS, Size(3, 3));
	//dilate(norm_grad_x, norm_grad_x, crossKern, Point(-1, -1), 2);
	//
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
	for(int i = 0; i < contours.size(); i++ ){
        drawContours(src, contours, i, Scalar(0,255,0), 2, LINE_8, hierarchy, 0);
    }
	
}

/**********************************************************************************
 * Main function        
 **********************************************************************************/
int main( int argc, char** argv )
{
	
	if(argc < 2)
    {
       printf("Usage: %s <Input image>\n", argv[0]);
       return SYSTEM_ERROR;
    }
    else{
		filename = argv[1];
	}
	
	src = imread(filename, IMREAD_COLOR); // Load an image
	
    if(src.empty()){
        printf("Could not open or find the image!\n");
        return SYSTEM_ERROR;
    }
    
    // Pre-processing pipeline	
    preprocess();
	
	// Contour selection
	vector<RotatedRect> candidatePlates;
	pruneLicensePlateCandidates(src_gray, contours, candidatePlates, 1.6, 4.0);
	
	// Store some results during the pre-processing for inspection
	imwrite("blackhat.jpg", blackhat);
	imwrite("light_region.jpg", light);
	imwrite("norm_grad_x.jpg", norm_grad_x);
	imwrite("grad_thresh.jpg", grad_thresh);
    
	// Send the cropped image to Tesseract OCR
	if (plate_detected){
		LicensePlate_postprocess();
	    ocr_process("plate_result.jpg");
    }
	
	// Temporary result display
    while(1){
        imshow(window_name1, src);
		imshow("BLACK HAT", blackhat);
		//imshow("BLACK HAT1", blackhat1);
		//imshow("BLACK HAT2", blackhat2);
		//imshow("Light Regions", light);
		imshow("Scharr", norm_grad_x);
		//imshow("Grad Erode/Dilate", grad_thresh);
		imshow("Final", grad_thresh);
		
	    char ch;
        if (ch = waitKey(10) == ESCAPE_KEY) break;
    }
  
    destroyAllWindows();
    return 0;
}
