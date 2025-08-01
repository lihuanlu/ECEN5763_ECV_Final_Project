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
#include <semaphore.h>
#include <unistd.h>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

using namespace std;
using namespace cv;

#define NUM_THREADS (7)
#define NUM_CPUS    (4)

#define SYSTEM_ERROR (-1)
#define ESCAPE_KEY   (27)
#define FONT         FONT_HERSHEY_SIMPLEX
#define BUFFER_SIZE  (30)

#define MY_CLOCK_TYPE CLOCK_MONOTONIC

/**********************************************************************************
 * Public variable     
 **********************************************************************************/
VideoCapture cap;
bool video_ended = false;
bool debug_mode = false;
bool abort_procedure = false;

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

typedef struct
{
    unsigned int frame_num;
	double frame_fps;
	Mat frame;	
} frame_buffer_t;

frame_buffer_t frame_buffer[BUFFER_SIZE];
int frame_wr_idx = 0;
int frame_rd_idx = 0;

frame_buffer_t plate_buffer[BUFFER_SIZE];
int plate_wr_idx = 0;
int plate_rd_idx = 0;

string new_ocr;
string prev_ocr;
unsigned int ocr_cnt = 0;
unsigned int one_sec_frame_count = 0;
double one_sec_counter = 0.0;
Point ocr_pt(10, 40);
Point fps_pt(10, 440);
char frame_text[128] = "OCR: ";
char frame_text2[16] = "FPS: ";

/**********************************************************************************
 * Thread variable     
 **********************************************************************************/
typedef struct
{
    int threadIdx;
	unsigned long long serviceCnt;
	double service_wcet;
	//const char* file_name;
	float minAR;
	float maxAR;
} threadParams_t;


pthread_t threads[NUM_THREADS];
threadParams_t threadParams[NUM_THREADS];

struct sched_param rt_param[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];

struct sched_param main_param;
pthread_attr_t main_attr;

sem_t semS1, semS2, semS3, semS4, semS5, semS6, semS7;

/**********************************************************************************
 * @name       S1_frame_read()
 *
 * @brief      { Read frame and convert to grayscale. }
 *
 * @param[in]  threadp          { Pointer to thread parameter structure. }
 * 
 * @return     None           
 **********************************************************************************/
void *S1_frame_read(void *threadp)
{
	double execution_time = 0.0;
	double service_start = 0.0;
	double service_end = 0.0;
	threadParams_t *threadParams = (threadParams_t *)threadp;

	threadParams->service_wcet = 0.0;
	threadParams->serviceCnt = 0;
	
	while (!video_ended){
		sem_wait(&semS1);	
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_start = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		frame_start = service_start;
		
		cap.read(src);
		
        if (src.empty()){ // end of file
            video_ended = true;
            abort_procedure = true;			
            sem_post(&semS2); sem_post(&semS3); 
			sem_post(&semS4); sem_post(&semS5);		
            break;			
		}
		
		frame_count++;
		one_sec_frame_count++;
		threadParams->serviceCnt++;

		sem_post(&semS2);
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		execution_time = service_end - service_start;
		
		if (execution_time > threadParams->service_wcet) threadParams->service_wcet = execution_time;
		
		// Log
		syslog(LOG_CRIT, "S1_frame_read on core %d for release %llu starts at %lf msec\n", sched_getcpu(), threadParams->serviceCnt, service_start);
		syslog(LOG_CRIT, "S1_frame_read finishes at %lf msec, C1 = %lf msec\n", service_end, execution_time);
    }
	
	syslog(LOG_CRIT, "S1_frame_read WCET = %lf msec\n", threadParams->service_wcet);
    pthread_exit((void *)0);	
} // end S1_frame_read


/**********************************************************************************
 * @name       S2_preprocess()
 *
 * @brief      { Pre-processing pipeline. }
 *
 * @param[in]  threadp          { Pointer to thread parameter structure. }
 * 
 * @return     None           
 **********************************************************************************/
void *S2_preprocess(void *threadp)
{
	double execution_time = 0.0;
	double service_start = 0.0;
	double service_end = 0.0;
	double minVal, maxVal;
	threadParams_t *threadParams = (threadParams_t *)threadp;

	threadParams->service_wcet = 0.0;	
	threadParams->serviceCnt = 0;
	
	while (!video_ended){
		sem_wait(&semS2);	
		
		if (abort_procedure) break;
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_start = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		
		threadParams->serviceCnt++;
		
	    // To grayscale
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
	
	    src_contour = src.clone();
		if (debug_mode){
	        // Draw the N contours on original frame
	        for(int i = 0; i < contours.size(); i++ ){
                drawContours(src_contour, contours, i, Scalar(0,255,0), 2, LINE_8, hierarchy, 0);
            }
		}
		
		sem_post(&semS3);
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		execution_time = service_end - service_start;
		
		if (execution_time > threadParams->service_wcet) threadParams->service_wcet = execution_time;
		
		// Log
		syslog(LOG_CRIT, "S2_preprocess on core %d for release %llu starts at %lf msec\n", sched_getcpu(), threadParams->serviceCnt, service_start);
		syslog(LOG_CRIT, "S2_preprocess finishes at %lf msec, C2 = %lf msec\n", service_end, execution_time);
	}
    
	syslog(LOG_CRIT, "S2_preprocess WCET = %lf msec\n", threadParams->service_wcet);
    pthread_exit((void *)0);
} // end of S2_preprocess

/**********************************************************************************
 * Translated from Python to C++ by ChatGPT
 * https://tesseract-ocr.github.io/tessdoc/Examples_C++.html      
 **********************************************************************************/
/**********************************************************************************
 * @name       S3_contour_selection()
 *
 * @brief      { License plate candidate selection among contours. }
 *
 * @param[in]  threadp        { Pointer to thread parameter structure. }
 * @param[in]  contours       { Contours found from previous process }
 * @param[in]  candidates     { To store the potential candidates }
 * @param[in]  minAR          { Minimum aspect ratio for selection }
 * @param[in]  maxAR          { Maximum aspect ratio for selection }
 * 
 * @return     None           
 **********************************************************************************/
//void pruneLicensePlateCandidates(const Mat& gray, const vector<vector<Point>>& contours,
//                                 vector<RotatedRect>& candidates, float minAR = 2.0f, float maxAR = 6.0f) 
void *S3_contour_selection(void *threadp)
{
	double execution_time = 0.0;
	double service_start = 0.0;
	double service_end = 0.0;
	threadParams_t *threadParams = (threadParams_t *)threadp;
    
	threadParams->service_wcet = 0.0;
	threadParams->serviceCnt = 0;
	frame_wr_idx = 0;
	
	while (!video_ended){
		sem_wait(&semS3);
		
		if (abort_procedure) break;
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_start = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		
		threadParams->serviceCnt++;
		
	    for (const auto& c : contours){
            // Fit a rotated bounding box to the contour
            RotatedRect rr = minAreaRect(c);
        
            // Extract the width and height
            float w = rr.size.width;
            float h = rr.size.height;

            // Make sure width is the larger side
            float aspectRatio = w > h ? w / h : h / w;

            // Check if aspect ratio is within the range of license plate
            if (aspectRatio >= threadParams->minAR && aspectRatio <= threadParams->maxAR){
                // Get rotated bounding box points
                Point2f boxPoints[4];
                rr.points(boxPoints);
                
				if (debug_mode){
			        // Draw all contours that fits AR in yellow
			        for (int i = 0; i < 4; ++i){
                        line(src_contour, boxPoints[i], boxPoints[(i + 1) % 4], Scalar(33, 222, 255), 2);	
				    }	
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
				
			        for (int i = 0; i < 4; ++i){
                        line(src_contour, boxPoints[i], boxPoints[(i + 1) % 4], Scalar(0, 0, 255), 2);	
	                }
                
				    M = getRotationMatrix2D(rr.center, angle, 1.0);
                    warpAffine(src_gray, rotated, M, src_gray.size(), INTER_CUBIC);
                    getRectSubPix(rotated, rect_size, rr.center, plate_cropped);
			        
				    break;
		        }
			    else{
				    plate_detected = false;
			    }
            } // end if aspectRatio
        } // end for
		
		if (plate_detected){			
			// This frame continues in the pipeline
			// Start S4 if there is a license plate
			sem_post(&semS4); 
		}
		else{
			// No license plate, this frame is done.
			// Print FPS and pass the frame to S6.
			// Start S1 for the next frame.
			frame_buffer[frame_wr_idx].frame = src_contour.clone();
			frame_buffer[frame_wr_idx].frame_num = frame_count;
		    
			clock_gettime(MY_CLOCK_TYPE, &current_time);
            frame_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		    fps = 1000.0 / (frame_end - frame_start);
		    total_fps += fps;
		    one_sec_fps += fps;
		    //syslog(LOG_CRIT, "Frame %d FPS = %0.1lf", frame_count, fps);
		
		    // Update fps to show on frame every 1 second
	        one_sec_counter += (frame_end - frame_start);
	        if (one_sec_counter >= 1.0){
	            frame_buffer[frame_wr_idx].frame_fps = one_sec_fps / (double)one_sec_frame_count;
                sprintf(frame_text2, "FPS: %.01lf", frame_buffer[frame_wr_idx].frame_fps);	    
			    one_sec_counter = 0;
			    one_sec_fps = 0;
	        }
		    frame_wr_idx = (frame_wr_idx + 1) % BUFFER_SIZE;
			sem_post(&semS1); 	
		}
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		execution_time = service_end - service_start;
		
		if (execution_time > threadParams->service_wcet) threadParams->service_wcet = execution_time;
		
		// Log
		syslog(LOG_CRIT, "S3_contour_selection on core %d for release %llu starts at %lf msec\n", sched_getcpu(), threadParams->serviceCnt, service_start);
		syslog(LOG_CRIT, "S3_contour_selection finishes at %lf msec, C3 = %lf msec\n", service_end, execution_time);
		syslog(LOG_CRIT, "Frame %d FPS = %0.1lf", frame_count, fps);
	} // end while
	
	syslog(LOG_CRIT, "S3_contour_selection WCET = %lf msec\n", threadParams->service_wcet);
    pthread_exit((void *)0);
} // end of S3_contour_selection

/**********************************************************************************
 * @name       S4_post_process()
 *
 * @brief      { License plate candidate post processing. }
 *
 * @param[in]  threadp          { Pointer to thread parameter structure. }
 * 
 * @return     None           
 **********************************************************************************/
void *S4_post_process(void *threadp)
{
	double execution_time = 0.0;
	double service_start = 0.0;
	double service_end = 0.0;
	threadParams_t *threadParams = (threadParams_t *)threadp;

	threadParams->service_wcet = 0.0;
	threadParams->serviceCnt = 0;
	
	while (!video_ended){
		sem_wait(&semS4);
		
		if (abort_procedure) break;
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_start = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		
		threadParams->serviceCnt++;
		
	    // Resize
	    resize(plate_cropped, plate_cropped, Size(plate_cropped.cols*2, plate_cropped.rows*2));
	    // Threshold the cropped region to get binary image
        threshold(plate_cropped, plate_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
		
		plate_buffer[plate_wr_idx].frame = plate_thresh.clone();
	    plate_buffer[plate_wr_idx].frame_num = frame_count;
		plate_wr_idx = (plate_wr_idx + 1) % BUFFER_SIZE;
		
		sem_post(&semS5);
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		execution_time = service_end - service_start;
		
		if (execution_time > threadParams->service_wcet) threadParams->service_wcet = execution_time;
		
		// Log
		syslog(LOG_CRIT, "S4_post_process on core %d for release %llu starts at %lf msec\n", sched_getcpu(), threadParams->serviceCnt, service_start);
		syslog(LOG_CRIT, "S4_post_process finishes at %lf msec, C4 = %lf msec\n", service_end, execution_time);
	} // end while
	
	syslog(LOG_CRIT, "S4_post_process WCET = %lf msec\n", threadParams->service_wcet);
    pthread_exit((void *)0);
} // end of S4_post_process

/**********************************************************************************
 * Copied from below link and adapted.
 * https://tesseract-ocr.github.io/tessdoc/Examples_C++.html      
 **********************************************************************************/
/**********************************************************************************
 * @name       S5_ocr_process()
 *
 * @brief      { Character recognition using Tesseract OCR engine. }
 *
 * @param[in]  threadp          { Pointer to thread parameter structure. }
 * 
 * @return     None           
 **********************************************************************************/
void *S5_ocr_process(void *threadp)
{
	double execution_time = 0.0;
	double service_start = 0.0;
	double service_end = 0.0;
	threadParams_t *threadParams = (threadParams_t *)threadp;

	threadParams->service_wcet = 0.0;
	threadParams->serviceCnt = 0;
	prev_ocr = outText;	
	
	while (!video_ended){
		sem_wait(&semS5);
		
		if (abort_procedure) break;
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_start = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		threadParams->serviceCnt++;
		
	    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
        // Initialize tesseract-ocr with English, without specifying tessdata path
        if (api->Init(NULL, "eng")) {
            fprintf(stderr, "Could not initialize tesseract.\n");
            exit(1);
        }

        // Open input image with leptonica library
        // api->SetImage(const unsigned char* imagedata, int width, int height, int bytes_per_pixel, int bytes_per_line);
	    api->SetImage(plate_thresh.data,  // pointer to the raw pixel data of the OpenCV Mat
		              plate_thresh.cols, 
					  plate_thresh.rows, 
					  1,                  // bytes per pixel (1 for grayscale)
					  plate_thresh.step); // number of bytes per row of the image (stride)
	    api->SetSourceResolution(300); // To avoid the DPI warning
        
		// Get OCR result
        char* rawText = api->GetUTF8Text();
        outText = rawText;
	    
		//outText.erase(std::remove(outText.begin(), outText.end(), '\n'), outText.end());
		if (!outText.empty() && outText.back() == '\n')
            outText.pop_back();
	    syslog(LOG_CRIT, "OCR output: %s at frame %d", outText.c_str(), frame_count);
			
		// Count as correct OCR if 2 consecutive results are the same
		if (outText == prev_ocr){ 
		    ocr_cnt++;		    
			//plate_detected = false;
        }		
		
		//if (ocr_cnt == 1 && outText.length() > 3){
		if (outText.length() > 3){	
			printf("OCR output: %s\n", outText.c_str());
			sprintf(frame_text, "OCR: %s", outText.c_str());
		    putText(src_contour, frame_text, ocr_pt, FONT, 0.8, Scalar(0,0,255), 2, LINE_8, false);
			ocr_cnt = 0;
		}
		else{
			putText(src_contour, "OCR:", ocr_pt, FONT, 0.8, Scalar(0,0,255), 2, LINE_8, false);
		}
		
		prev_ocr = outText;	
		
        // Destroy used object and release memory
        api->End();
        delete api;
        delete [] rawText;	
        
		// Print FPS and pass the frame to S6.
        // Start S1 for the next frame.
	    frame_buffer[frame_wr_idx].frame = src_contour.clone();
	    frame_buffer[frame_wr_idx].frame_num = frame_count;
		    
	    clock_gettime(MY_CLOCK_TYPE, &current_time);
        frame_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		fps = 1000.0 / (frame_end - frame_start);
		total_fps += fps;
		one_sec_fps += fps;
		syslog(LOG_CRIT, "Frame %d FPS = %0.1lf", frame_count, fps);
		
		// Update fps to show on frame every 1 second
	    one_sec_counter += (frame_end - frame_start);
	    if (one_sec_counter >= 1.0){
	        frame_buffer[frame_wr_idx].frame_fps = one_sec_fps / (double)one_sec_frame_count;
            sprintf(frame_text2, "FPS: %.01lf", frame_buffer[frame_wr_idx].frame_fps);	    
			one_sec_counter = 0;
			one_sec_fps = 0;
	    }
		frame_wr_idx = (frame_wr_idx + 1) % BUFFER_SIZE;
			
		sem_post(&semS1);
		
        clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		execution_time = service_end - service_start;
		
		if (execution_time > threadParams->service_wcet) threadParams->service_wcet = execution_time;
		
		// Log
		syslog(LOG_CRIT, "S5_ocr_process on core %d for release %llu starts at %lf msec\n", sched_getcpu(), threadParams->serviceCnt, service_start);
		syslog(LOG_CRIT, "S5_ocr_process finishes at %lf msec, C5 = %lf msec\n", service_end, execution_time);		
    }
	
	syslog(LOG_CRIT, "S5_ocr_process WCET = %lf msec\n", threadParams->service_wcet);
    pthread_exit((void *)0);
} // end of S5_ocr_process

/**********************************************************************************
 * @name       S6_frame_write()
 *
 * @brief      { Frame PPM files write-back. }
 *
 * @param[in]  threadp          { Pointer to thread parameter structure. }
 * 
 * @return     None           
 **********************************************************************************/
void *S6_frame_write(void *threadp)
{
	double execution_time = 0.0;
	double service_start = 0.0;
	double service_end = 0.0;
	threadParams_t *threadParams = (threadParams_t *)threadp;

	threadParams->service_wcet = 0.0;
	threadParams->serviceCnt = 0;
	frame_rd_idx = 0;
	
	while (!video_ended || frame_rd_idx != frame_wr_idx){
		
		if (frame_rd_idx == frame_wr_idx){
		    usleep(10000);
			continue;
		}
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_start = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		
		threadParams->serviceCnt++;
		
		putText(src_contour, frame_text2, fps_pt, FONT, 0.8, Scalar(0,0,255), 2, LINE_8, false);
		
		sprintf(contour_file, "contour_result/contour_result%04d.ppm", frame_buffer[frame_rd_idx].frame_num);
	    imwrite(contour_file, frame_buffer[frame_rd_idx].frame);	    
		frame_rd_idx = (frame_rd_idx + 1) % BUFFER_SIZE;
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		execution_time = service_end - service_start;
		
		if (execution_time > threadParams->service_wcet) threadParams->service_wcet = execution_time;
		
		// Log
		syslog(LOG_CRIT, "S6_frame_write on core %d for release %llu starts at %lf msec\n", sched_getcpu(), threadParams->serviceCnt, service_start);
		syslog(LOG_CRIT, "S6_frame_write finishes at %lf msec, C6 = %lf msec\n", service_end, execution_time);
	}
	
	syslog(LOG_CRIT, "S6_frame_write WCET = %lf msec\n", threadParams->service_wcet);
    pthread_exit((void *)0);
} // end of S6_frame_write

/**********************************************************************************
 * @name       S7_plate_write()
 *
 * @brief      { License plate JPG files write-back. }
 *
 * @param[in]  threadp          { Pointer to thread parameter structure. }
 * 
 * @return     None           
 **********************************************************************************/
void *S7_plate_write(void *threadp)
{
	double execution_time = 0.0;
	double service_start = 0.0;
	double service_end = 0.0;
	threadParams_t *threadParams = (threadParams_t *)threadp;

	threadParams->service_wcet = 0.0;
	threadParams->serviceCnt = 0;
	plate_rd_idx = 0;
	
	while (!video_ended || plate_rd_idx != plate_wr_idx){
		
		if (plate_rd_idx == plate_wr_idx){
		    usleep(10000);
			continue;
		}
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_start = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		
		threadParams->serviceCnt++;
		
		sprintf(plate_file, "plate_result/plate_result%04d.jpg", plate_buffer[plate_rd_idx].frame_num);
	    imwrite(plate_file, plate_buffer[plate_rd_idx].frame);    
		plate_rd_idx = (plate_rd_idx + 1) % BUFFER_SIZE;
		
		clock_gettime(MY_CLOCK_TYPE, &current_time);
        service_end = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0) - start_time;
		execution_time = service_end - service_start;
		
		if (execution_time > threadParams->service_wcet) threadParams->service_wcet = execution_time;
		
		// Log
		syslog(LOG_CRIT, "S7_plate_write on core %d for release %llu starts at %lf msec\n", sched_getcpu(), threadParams->serviceCnt, service_start);
		syslog(LOG_CRIT, "S7_plate_write finishes at %lf msec, C7 = %lf msec\n", service_end, execution_time);
	}
	
	syslog(LOG_CRIT, "S7_plate_write WCET = %lf msec\n", threadParams->service_wcet);
    pthread_exit((void *)0);
} // end of S7_plate_write

/**********************************************************************************
 * @name       print_scheduler()
 *
 * @brief      { Print scheduler info. }
 *
 * @param[in]  None
 * 
 * @return     None           
 **********************************************************************************/
void print_scheduler(void)
{
   int schedType, scope;

   schedType = sched_getscheduler(getpid());

   switch(schedType)
   {
     case SCHED_FIFO:
           printf("Pthread Policy is SCHED_FIFO\n");
           break;
     case SCHED_OTHER:
           printf("Pthread Policy is SCHED_OTHER\n");
       break;
     case SCHED_RR:
           printf("Pthread Policy is SCHED_OTHER\n");
           break;
     default:
       printf("Pthread Policy is UNKNOWN\n");
   }

   pthread_attr_getscope(&main_attr, &scope);

   if(scope == PTHREAD_SCOPE_SYSTEM)
     printf("PTHREAD SCOPE SYSTEM\n");
   else if (scope == PTHREAD_SCOPE_PROCESS)
     printf("PTHREAD SCOPE PROCESS\n");
   else
     printf("PTHREAD SCOPE UNKNOWN\n");

}

/**********************************************************************************
 * Main function        
 **********************************************************************************/
int main( int argc, char** argv )
{
    const char* default_file = "plate_vid.mp4";
	const char* filename;
	const char* debug_str = "debug";
	
	int rc, i;
	cpu_set_t threadcpu;
	pid_t mainpid;
    int rt_max_prio, rt_min_prio, cpuidx;
	
	if (argc == 3){
		if (strcmp(argv[2], debug_str) == 0)
			debug_mode = true;
		
		filename = argv[1];
		cap.open(filename);
	}	
	else if (argc == 2){
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
	
	// initialize the sequencer semaphores
    if (sem_init (&semS1, 0, 0)) { printf ("Failed to initialize S1 semaphore\n"); exit (SYSTEM_ERROR); }
    if (sem_init (&semS2, 0, 0)) { printf ("Failed to initialize S2 semaphore\n"); exit (SYSTEM_ERROR); }
    if (sem_init (&semS3, 0, 0)) { printf ("Failed to initialize S3 semaphore\n"); exit (SYSTEM_ERROR); }
    if (sem_init (&semS4, 0, 0)) { printf ("Failed to initialize S4 semaphore\n"); exit (SYSTEM_ERROR); }
    if (sem_init (&semS5, 0, 0)) { printf ("Failed to initialize S5 semaphore\n"); exit (SYSTEM_ERROR); }
	if (sem_init (&semS6, 0, 0)) { printf ("Failed to initialize S6 semaphore\n"); exit (SYSTEM_ERROR); }
	if (sem_init (&semS7, 0, 0)) { printf ("Failed to initialize S7 semaphore\n"); exit (SYSTEM_ERROR); }
	
	// set main program parameter
	mainpid = getpid();

    rt_max_prio = sched_get_priority_max(SCHED_FIFO);
    rt_min_prio = sched_get_priority_min(SCHED_FIFO);

    rc = sched_getparam(mainpid, &main_param);
    main_param.sched_priority = rt_max_prio;
    rc = sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
    if(rc < 0) perror("main_param");
    print_scheduler();
	
	// set rt threads parameter
    for(i=0; i < NUM_THREADS; i++)
    {
        // run S1 to S5 on core 0
        if (i < 5){
            CPU_ZERO(&threadcpu);
            cpuidx=(0);
            CPU_SET(cpuidx, &threadcpu);
        }
		// run S6 on core 1
		else if(i == 5){	
			CPU_ZERO(&threadcpu);
            cpuidx=(1);
            CPU_SET(cpuidx, &threadcpu);
		}
        // run S7 on core 2
        else{
            CPU_ZERO(&threadcpu);
            cpuidx=(2);
            CPU_SET(cpuidx, &threadcpu);
        }

        rc = pthread_attr_init(&rt_sched_attr[i]);
        rc = pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
        rc = pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
        rc = pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

        rt_param[i].sched_priority = rt_max_prio - i;
        rc = pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);
        
		if (rc < 0) perror("rt_param");
		
        threadParams[i].threadIdx = i + 1;
    }
	
	
	clock_gettime(MY_CLOCK_TYPE, &current_time);
    start_time = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0);
	printf("Start license plate detection\n");	
	sem_post(&semS1);
	
	// S1_frame_read
    rc = pthread_create(&threads[0], &rt_sched_attr[0], S1_frame_read, (void *)&(threadParams[0]));
	if(rc < 0) perror("pthread_create for S1_frame_read");
	else printf("pthread_create successful for S1_frame_read\n");
	
	// S2_preprocess
	rc = pthread_create(&threads[1], &rt_sched_attr[1], S2_preprocess, (void *)&(threadParams[1]));
	if(rc < 0) perror("pthread_create for S2_preprocess");
	else printf("pthread_create successful for S2_preprocess\n");
	
	// S3_contour_selection
	threadParams[2].minAR = 1.5;
	threadParams[2].maxAR = 6.0;
	rc = pthread_create(&threads[2], &rt_sched_attr[2], S3_contour_selection, (void *)&(threadParams[2]));
	if(rc < 0) perror("pthread_create for S3_contour_selection");
	else printf("pthread_create successful for S3_contour_selection\n");
	
	// S4_post_process
	rc = pthread_create(&threads[3], &rt_sched_attr[3], S4_post_process, (void *)&(threadParams[3]));
	if(rc < 0) perror("pthread_create for S4_post_process");
	else printf("pthread_create successful for S4_post_process\n");
	
	// S5_ocr_process
	rc = pthread_create(&threads[4], &rt_sched_attr[4], S5_ocr_process, (void *)&(threadParams[4]));
	if(rc < 0) perror("pthread_create for S5_ocr_process");
	else printf("pthread_create successful for S5_ocr_process\n");
	
	// S6_frame_write
    rc = pthread_create(&threads[5], &rt_sched_attr[5], S6_frame_write, (void *)&(threadParams[5]));
	if(rc < 0) perror("pthread_create for S6_frame_write");
	else printf("pthread_create successful for S6_frame_write\n");
	
	// S7_plate_write
	rc = pthread_create(&threads[6], &rt_sched_attr[6], S7_plate_write, (void *)&(threadParams[6]));
	if(rc < 0) perror("pthread_create for S7_plate_write");
	else printf("pthread_create successful for S7_plate_write\n");
	
    for(i=0;i<NUM_THREADS;i++)
    {
        if(rc = pthread_join(threads[i], NULL) < 0)
		perror("main pthread_join");
	else
		printf("joined thread %d\n", i);
    }
    
	clock_gettime(MY_CLOCK_TYPE, &current_time);
    end_time = (double)current_time.tv_sec*1000.0 + ((double)current_time.tv_nsec/1000000.0);
	//avg_fps = (double)frame_count / (end_time - start_time);
	avg_fps = total_fps / (double)frame_count;
	printf("Video ended at %lf.\n", (end_time - start_time)/1000.0);
	printf("Average frame rate: %.1lf fps\n", avg_fps);
	printf("Total frames = %d\n", frame_count);
			
	syslog(LOG_CRIT, "Average frame rate: %.1lf fps. Total frames = %d", avg_fps, frame_count);

	
	cap.release();
    destroyAllWindows();
	
    return 0;
}
