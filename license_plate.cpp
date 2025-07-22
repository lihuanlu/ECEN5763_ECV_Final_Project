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

Mat src, src_gray, temp_result;
Mat blackhat, grad_thresh, light;
Mat grad_x, abs_grad_x, norm_grad_x;

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
 * @brief      { Apply Canny operation. }
 *
 * @param[in]  userdata          { Pointer to FrameData structure }
 * @param[in]  userdata          { Pointer to FrameData structure }
 * @param[in]  userdata          { Pointer to FrameData structure }
 * @param[in]  userdata          { Pointer to FrameData structure }
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
			imwrite("plate_result.jpg", thresh);
			
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
 * Main function        
 **********************************************************************************/
int main( int argc, char** argv )
{
	
	double minVal, maxVal;
	
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
	vector<vector<Point>> contours;
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
	
	for(int i = 0; i < contours.size(); i++ ){
        drawContours(src, contours, i, Scalar(0,255,0), 2, LINE_8, hierarchy, 0);
    }
	
	vector<RotatedRect> candidatePlates;
	pruneLicensePlateCandidates(src_gray, contours, candidatePlates);
	
	//grad_thresh.copyTo(temp_result);
	//imwrite("temp_result.jpg", temp_result);
	imwrite("blackhat.jpg", blackhat);
	imwrite("light_region.jpg", light);
	imwrite("norm_grad_x.jpg", norm_grad_x);
	imwrite("grad_thresh.jpg", grad_thresh);
    
	ocr_process("plate_result.jpg");

    while(1){
        imshow(window_name1, src);
		//imshow("BLACK HAT", blackhat);
		imshow("Light Regions", light);
		//imshow("Scharr", norm_grad_x);
		//imshow("Grad Erode/Dilate", grad_thresh);
		imshow("Final", grad_thresh);
		
	    char ch;
        if (ch = waitKey(10) == ESCAPE_KEY) break;
    }
  
    destroyAllWindows();
    return 0;
}
