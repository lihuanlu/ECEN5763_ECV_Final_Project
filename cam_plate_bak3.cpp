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
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <algorithm>  // for std::sort
#include <vector>

#include <pthread.h>
#include <syslog.h>
#include <time.h>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace std;
using namespace cv;

#define SYSTEM_ERROR (-1)
#define ESCAPE_KEY   (27)
#define FONT         FONT_HERSHEY_SIMPLEX

#define MY_CLOCK_TYPE CLOCK_MONOTONIC

/**********************************************************************************
 * Public variable     
 **********************************************************************************/
Mat src, src_gray, src_contour;
Mat blackhat, grad_thresh, light;
Mat grad_x, abs_grad_x, norm_grad_x;
Mat plate_cropped, plate_thresh;

vector<vector<Point>> contours;

const char* window_name1 = "Source";
const char* window_name2 = "Cropped";
const char* window_name3 = "Contours";

string outText; // OCR result string
bool plate_detected = false;

char plate_file[48];
char contour_file[48];
unsigned int frame_count = 0;

double frame_start = 0.0, frame_end = 0.0;
double start_time = 0.0, end_time = 0.0;
double fps = 0.0, avg_fps = 0.0, total_fps = 0.0;
double new_fps = 0.0, one_sec_fps = 0.0;
struct timespec current_time;

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
            
			// Draw all contours that fits AR in yellow
			for (int i = 0; i < 4; ++i){
                line(src_contour, boxPoints[i], boxPoints[(i + 1) % 4], Scalar(33, 222, 255), 2);	
	        }
				
            // Crop out the potential plate candidate from the gray image
			Mat M, rotated;
            float angle = rr.angle;
            Size rect_size = rr.size;

            if (rr.angle < -45.0f){
                angle += 90.0f;
                swap(rect_size.width, rect_size.height);
            }
            
			// Skip those contours that are too big
			if (rr.size.width > src.cols / 4 || rr.size.height > src.rows / 4){
				plate_detected = false;
				continue;				
			}
			
			// Check if the center is around the center of the original image
			if (rr.center.y < src.rows / 3 || rr.center.y > src.rows * 2 / 3){
			    plate_detected = false;
				continue;				
			}
			
			// Check if the center is around the center of the original image
			if (rr.center.x > src.cols / 3 && rr.center.x < src.cols * 2 / 3){
                plate_detected = true;
				//printf("Candidate found.\n");
				
			    for (int i = 0; i < 4; ++i){
                    line(src_contour, boxPoints[i], boxPoints[(i + 1) % 4], Scalar(0, 0, 255), 2);	
	            }
                
				M = getRotationMatrix2D(rr.center, angle, 1.0);
                warpAffine(gray, rotated, M, gray.size(), INTER_CUBIC);
                getRectSubPix(rotated, rect_size, rr.center, plate_cropped);
				// Store the rotated rect (candidate)
                //candidates.push_back(rr);
			
				break;
		    }
			else{
				plate_detected = false;
			}
            
        } // end if aspectRatio
    } // end for

}

void LicensePlate_postprocess()
{
	// Threshold the cropped region to get binary image
    threshold(plate_cropped, plate_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
	sprintf(plate_file, "plate_result/plate_result%04d.jpg", frame_count);
	imwrite(plate_file, plate_thresh);
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
	api->SetSourceResolution(300); // To avoid the DPI warning
    // Get OCR result
    char* rawText = api->GetUTF8Text();

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
	unsigned int one_sec_frame_count = 0;
	double one_sec_counter = 0.0;
	Point ocr_pt(10, 40);
	Point fps_pt(10, 440);
	char frame_text[128] = "OCR: ";
	char frame_text2[16] = "FPS: ";
	
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
	
	cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
	
	clock_gettime(MY_CLOCK_TYPE, &current_time);
    start_time = (double)current_time.tv_sec + ((double)current_time.tv_nsec/1000000000.0);
	printf("Start license plate detection\n");	
	
    while(1)
    {
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        frame_start = (double)current_time.tv_sec + ((double)current_time.tv_nsec/1000000000.0);
		
		cap.read(src);
		
		if (src.empty()){
			clock_gettime(MY_CLOCK_TYPE, &current_time);
            end_time = (double)current_time.tv_sec + ((double)current_time.tv_nsec/1000000000.0);
			//avg_fps = (double)frame_count / (end_time - start_time);
			avg_fps = total_fps / (double)frame_count;
			printf("Video ended at %lf.\n", end_time - start_time);
			printf("Average frame rate: %.1lf fps\n", avg_fps);
			printf("Total frames = %d\n", frame_count);
			
			syslog(LOG_CRIT, "Average frame rate: %.1lf fps. Total frames = %d", fps, frame_count);
			break; // check if at end
        }
		
		frame_count++;
		one_sec_frame_count++;
		
	    // Pre-processing pipeline	
        preprocess();
	
	    // Contour selection
	    vector<RotatedRect> candidatePlates;
	    pruneLicensePlateCandidates(src_gray, contours, candidatePlates, 1.5, 6.0);
	    
		if (plate_detected){
			// Post-processing
		    LicensePlate_postprocess();
	        // Send the cropped image to Tesseract OCR
	        new_ocr = ocr_process(plate_file);
			//new_ocr.erase(std::remove(new_ocr.begin(), new_ocr.end(), '\n'), new_ocr.end());
			if (!new_ocr.empty() && new_ocr.back() == '\n')
                new_ocr.pop_back();
			syslog(LOG_CRIT, "OCR output: %s at frame %d", new_ocr.c_str(), frame_count);
			
			// Count as correct OCR if 2 consecutive results are the same
		    if (new_ocr == prev_ocr) 
		        ocr_cnt++;
			
			prev_ocr = new_ocr;	
			plate_detected = false;
        }		
		
		if (ocr_cnt == 1 && new_ocr.length() > 3){
			printf("OCR output:\n%s", new_ocr.c_str());
			sprintf(frame_text, "OCR: %s", new_ocr.c_str());
		    putText(src_contour, frame_text, ocr_pt, FONT, 0.8, Scalar(0,0,255), 2, LINE_8, false);
			ocr_cnt = 0;
		}
		else{
			putText(src_contour, "OCR:", ocr_pt, FONT, 0.8, Scalar(0,0,255), 2, LINE_8, false);
		}
        
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        frame_end = (double)current_time.tv_sec + ((double)current_time.tv_nsec/1000000000.0);
		fps = 1.0 / (frame_end - frame_start);
		total_fps += fps;
		one_sec_fps += fps;
		syslog(LOG_CRIT, "Frame %d FPS = %0.1lf", frame_count, fps);
		
		// Update fps to show on frame every 1 second
	    one_sec_counter += (frame_end - frame_start);
	    if (one_sec_counter >= 1.0){
	        new_fps = one_sec_fps / (double)one_sec_frame_count;
            sprintf(frame_text2, "FPS: %.01lf", new_fps);	    
			one_sec_counter = 0;
			one_sec_fps = 0;
	    }
		
		putText(src_contour, frame_text2, fps_pt, FONT, 0.8, Scalar(0,0,255), 2, LINE_8, false);
		
		sprintf(contour_file, "contour_result/contour_result%04d.ppm", frame_count);
	    imwrite(contour_file, src_contour);
	
		//imshow(window_name3, src_contour);
		
	    //char ch;
        //if (ch = waitKey(1) == ESCAPE_KEY) break;
    }
		
	cap.release();
    destroyAllWindows();
	
    return 0;
}
