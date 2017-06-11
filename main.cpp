#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "GeometricRecognizer.h"
#include "PathWriter.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
using namespace DollarRecognizer;
/**
* @function main
*/

// Dataset ds325 Paths
/*
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds325\\fast_circles\\%06d_depth.tiff");					299
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds325\\gestures_two_hands\\%06d_depth.tiff");			299
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds325\\gestures_two_hands_swap\\%06d_depth.tiff");		299
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds325\\sequence_closed_hand\\%06d_depth.tiff");			299
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds325\\sequence_open_hand\\%06d_depth.tiff");			299
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds325\\sequence_small_shapes\\%06d_depth.tiff");		299
*/
// Dataset ds536 Paths
/*
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\circle_ccw\\%06d_depth.tif")						201
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\circle_ccw_far\\%06d_depth.tif")					243
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\circle_ccw_hand\\%06d_depth.tif")				143
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\circle_sequence\\%06d_depth.tif")				285
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\multiple_shapes_1\\%06d_depth.tif")				760
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\rectangle_ccw\\%06d_depth.tif")					686
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\rectangle_cw\\%06d_depth.tif")					796
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\star\\%06d_depth.tif")							529
("C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds536\\zigzag\\%06d_depth.tif")
*/
bool two_hands = false;
bool one_hand = true;
bool no_contour = false;

int findBiggestContour(vector<vector<Point>> contours);
int secondBiggestContour(vector<vector<Point>> contours);

double dist(Point2D pt1, Point2D pt2)
{
	double dx = pt2.x - pt1.x;
	double dy = pt2.y - pt1.y;
	return sqrt(dx*dx + dy*dy);
}

static Rect pointSetBoundingRect( const Mat& points , Mat m)
{
    int npoints = points.checkVector(2);
    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i;
    Point ptxmin , ptymin , ptxmax , ptymax;

    if( npoints == 0 )
        return Rect();

    const Point* pts = points.ptr<Point>();
    Point pt = pts[0];

    ptxmin = ptymin = ptxmax = ptymax = pt;
    xmin = xmax = pt.x;
    ymin = ymax = pt.y;

    for( i = 1; i < npoints; i++ )
    {
        pt = pts[i];

        if( xmin > pt.x )
        {
            xmin = pt.x;
            ptxmin = pt;
        }
        if( xmax < pt.x )
        {
            xmax = pt.x;
            ptxmax = pt;
        }
        if( ymin > pt.y )
        {
            ymin = pt.y;
            ptymin = pt;
        }
        if( ymax < pt.y )
        {
            ymax = pt.y;
            ptymax = pt;
        }
    }
    //ellipse( m, ptxmin, Size( 3, 3), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
    //ellipse( m, ptxmax, Size( 3, 3), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
    //ellipse( m, ptymin, Size( 3, 3), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
    //ellipse( m, ptymax, Size( 3, 3), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );

	return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}


Mat return_convexhull(Mat img)
{
	int thresh = 50;
	int max_thresh = 255;
	Mat src_copy = img.clone();
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
		
	/// Detect edges using Threshold
	threshold(src_copy, threshold_output, thresh, 255, THRESH_BINARY );

	/// Find contours
	findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	/// Find the convex hull object for each contour
	if(one_hand)
	{
		int indexlargest = findBiggestContour(contours);	
		vector<vector<Point> >hull( contours.size() );
		convexHull( Mat(contours[indexlargest]), hull[indexlargest], false );
	
		Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3);
		Scalar color = Scalar( 255, 255, 0 );
		drawContours( drawing, hull, indexlargest, color, 1, 8, vector<Vec4i>(), 0, Point() );
		//drawContours( drawing, contours, indexlargest, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
	
		return drawing;
	}
	if(two_hands)
	{
		int indexlargest = findBiggestContour(contours);	
		int secondlargest = secondBiggestContour(contours);
		vector<vector<Point> >hull( contours.size() );
		convexHull( Mat(contours[indexlargest]), hull[indexlargest], false );
		convexHull( Mat(contours[secondlargest]), hull[secondlargest], false );
	

		Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3);
		Scalar color = Scalar( 255, 255, 0 );
		drawContours( drawing, hull, indexlargest, color, 1, 8, vector<Vec4i>(), 0, Point() );
		//drawContours( drawing, contours, indexlargest, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
		drawContours( drawing, hull, secondlargest, color, 1, 8, vector<Vec4i>(), 0, Point() );
		//drawContours( drawing, contours, secondlargest, Scalar(0,0,255), 1, 8, vector<Vec4i>(), 0, Point() );
	
		return drawing;
	}
}

int findBiggestContour(vector<vector<Point>> contours)
{
	int indexOfBiggestContour = -1;
    int sizeOfBiggestContour = 0;

    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > sizeOfBiggestContour){
            sizeOfBiggestContour = contours[i].size();
            indexOfBiggestContour = i;
        }
    }
    return indexOfBiggestContour;
}

int secondBiggestContour(vector<vector<Point>> contours)
{
	int indexOfSecondBiggestContour = -1;
    int sizeOfSecondBiggestContour = 0;

	int indexOfBiggestContour = findBiggestContour(contours);

    for (int i = 0; i < contours.size(); i++){
        if(contours[i].size() > sizeOfSecondBiggestContour && contours[i].size() < contours[indexOfBiggestContour].size()){
            sizeOfSecondBiggestContour = contours[i].size();
            indexOfSecondBiggestContour = i;
        }
    }
    return indexOfSecondBiggestContour;
}

Mat kmeans_clustering(Mat src)
{
	src.convertTo(src, CV_32F);
	Mat samples(src.rows * src.cols, 3, CV_32F);
	for( int y = 0; y < src.rows; y++ )
		for( int x = 0; x < src.cols; x++ )
			for( int z = 0; z < 3; z++)
				samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];


	int clusterCount = 15;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers );


	Mat new_image( src.size(), src.type() );
	for( int y = 0; y < src.rows; y++ )
		for( int x = 0; x < src.cols; x++ )
		{ 
		int cluster_idx = labels.at<int>(y + x*src.rows,0);
		new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
		new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
		new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
		}
	return new_image;
}

int main( int argc, const char** argv )
{		
	GeometricRecognizer g;
	Path2D points, points1;
	g.loadTemplates();
	vector<string> gesturesList;
    gesturesList.push_back("Circle");
	gesturesList.push_back("Rectangle");
	gesturesList.push_back("Triangle");
	gesturesList.push_back("Star");
	gesturesList.push_back("0");
    g.activateTemplates(gesturesList);
	
	namedWindow("results", 1);
	int iminThreshold = 70;
    createTrackbar("minThreshold", "results", &iminThreshold, 150);

	int i=0; 
	Point old_pt;
	char path[500];
	Mat clustered_image;
	string shape = "Shape";
	string shape1 = "";
	vector<Point> vis;

	for(;;)
    {
        int minThreshold = iminThreshold;
		int maxThreshold = 300;
	
		if(i++<300)
		{
			Mat depthimage;
			sprintf(path,"C:\\Users\\Imran Zafar\\Documents\\dataset3\\dataset\\ds325\\sequence_small_shapes\\%06d_depth.tiff", i);
            depthimage = imread(path,CV_LOAD_IMAGE_ANYCOLOR|CV_LOAD_IMAGE_ANYDEPTH);
			Mat depthf;
			depthimage.convertTo(depthf, CV_8UC1, 255.0/2048.0);
			Mat visp(depthimage.rows, depthimage.cols, CV_8UC3);

			float sumX = 0;
			float sumY = 0;
			float totalPixels = 0;

			Mat hand(depthimage.rows, depthimage.cols, CV_8UC1);
				
			for(int x = 0; x<depthimage.cols; x++)
			{
				for(int y = 0; y<depthimage.rows; y++)
				{
					int offset = x + y*depthimage.cols;
					float d = depthf.data[offset];
				
					// extracting the hand
					if (d>minThreshold && d<maxThreshold)
						hand.data[offset] = 0;
					else
					{
						hand.data[offset] = 255;
						
						// computing centroid for marker
						if(one_hand && no_contour)
						{
							sumX += x;
							sumY += y;
							totalPixels++;
						}						
					}
				}
			}

			if(one_hand)
			{
				if(no_contour)
				{
					float avgX = sumX/totalPixels;
					float avgY = sumY/totalPixels;
			
					// creating a vector of centroids to give to stroke recognizer
					points.push_back(Point2D(avgX, avgY));

					//if(i==40||i==80||i==120||i==160||i==200||i==240||i==280)		// for fast circles dataset
					//if(i==120||i==299)											// for closed hand
					//if(i==80||i==200)												// for open hand
					if(i==120||i==200||i==299)										// small shapes
					//if(i==100||i==200||i==285)
					//if(i==150||i==320||i==490||i==680)
					//if(i==260||i==520||i==796)
					//if(i==260||i==529)												// star
					{
						RecognitionResult r = g.recognize(points);
						cout << "Recognized gesture: " << r.name << endl;
						cout << "1$ Score:" << r.score << endl;
						points.clear();
					}

					circle(hand, Point(avgX,avgY), 6, Scalar(0, 0, 255), -1, 8);
					imshow("depth",depthimage);
					imshow("hand", hand);
				}
				else
				{
					Mat results;
					Mat src = hand;
					Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
					Mat hull = return_convexhull(src);

					vector<vector<Point>> contours;
					vector<Vec4i> hierarchy;

					findContours(src, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);
					int indexOfBiggestContour = findBiggestContour(contours);

					Rect minRect;
					if(indexOfBiggestContour != -1)
					{
						Scalar color(255, 0, 255);
						drawContours( dst, contours, indexOfBiggestContour, color, -1, 8, hierarchy );
						minRect = pointSetBoundingRect( Mat(contours[indexOfBiggestContour]), dst );
						rectangle( dst, minRect, Scalar(255,0,0), 4, 8 );
						//circle(dst, Point((2*minRect.x+minRect.width)/2,(minRect.y)), 6, Scalar(0, 0, 255), -1, 8);
						circle(dst, Point((2*minRect.x+minRect.width)/2,(2*minRect.y+minRect.height)/2), 6, Scalar(255, 0, 0), -1, 8);
						//points.push_back(Point2D((2*minRect.x+minRect.width)/2,(minRect.y)));
						points.push_back(Point2D((2*minRect.x+minRect.width)/2,(2*minRect.y+minRect.height)/2));
						vis.push_back(Point((2*minRect.x+minRect.width)/2,(2*minRect.y+minRect.height)/2));	
						putText(dst, shape, Point(10,50), FONT_HERSHEY_SIMPLEX, 2.0,Scalar(255,0,255),2);
						
						//if(i==40||i==80||i==120||i==160||i==200||i==240||i==280)		// for fast circles dataset
						//if(i==70||i==299)											// for closed hand
						//if(i==130||i==270)												// for open hand
						if(i==120||i==230||i==299)										// small shapes
						//if(i==100||i==200||i==285)
						//if(i==150||i==320||i==490||i==680)
						//if(i==260||i==520||i==796)
						//if(i==260||i==529)												// star
						{
							RecognitionResult r = g.recognize(points);
							cout << "Recognized gesture: " << r.name << endl;
							cout << "1$ Score:" << r.score << endl;	
							shape = r.name;
							points.clear();
						}						
					}
					putText(dst, shape, Point(10,50), FONT_HERSHEY_SIMPLEX, 2.0,Scalar(255,0,255),2);
								
					for(int v = 0; v<vis.size(); v++)
						circle(visp, Point(vis[v].x,vis[v].y), 3, Scalar(0, 0, 255), -1, 8);

					vconcat(hull, dst, results);	
					imshow("results", results);
					imshow("depth", depthimage);
					imshow("path", visp);
				}
				
				if(i==299)
				{
					vis.clear();
					visp.setTo(Scalar(0,0,0));
					imshow("path", visp);
				}
			}
	
			if(two_hands)
			{
				Mat results;
				Mat src = hand;
				Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
				Mat hull = return_convexhull(src);

				vector<vector<Point>> contours;
				vector<Vec4i> hierarchy;

				//clustered_image = kmeans_clustering(src);
				//clustered_image.convertTo(clustered_image, CV_8UC1, 255.0/2048.0);
				findContours(src, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);
				int indexOfBiggestContour = findBiggestContour(contours);
				int indexOfSecondBiggestContour = secondBiggestContour(contours);

				Rect minRect1, minRect2;
				
				if (indexOfBiggestContour != -1 && indexOfSecondBiggestContour != -1)
				{		
					Scalar color(255, 0, 255);
					drawContours( dst, contours, indexOfBiggestContour, color, -1, 8, hierarchy );
					drawContours( dst, contours, indexOfSecondBiggestContour, color, -1, 8, hierarchy );
					minRect1 = pointSetBoundingRect( Mat(contours[indexOfBiggestContour]),dst );
					minRect2 = pointSetBoundingRect( Mat(contours[indexOfSecondBiggestContour]),dst );
					rectangle( dst, minRect1, Scalar(255,0,0), 4, 8 );
					rectangle( dst, minRect2, Scalar(255,0,0), 4, 8 );
					
					//putText(dst, shape, Point(10,50), FONT_HERSHEY_SIMPLEX, 1.0,Scalar(255,0,255),2);
						
					// pointer at the tip of the hand
					/*
					circle(dst, Point((2*minRect1.x+minRect1.width)/2,(minRect1.y)), 6, Scalar(0, 0, 0), -1, 8);
					circle(dst, Point((2*minRect2.x+minRect2.width)/2,(minRect2.y)), 6, Scalar(0, 0, 0), -1, 8);
					points.push_back(Point2D((2*minRect1.x+minRect1.width)/2,(minRect1.y)));
					points1.push_back(Point2D((2*minRect2.x+minRect2.width)/2,(minRect2.y)));
					*/

					// pointer in the middle of the hand
					circle(dst, Point((2*minRect1.x+minRect1.width)/2,(2*minRect1.y+minRect1.height)/2), 6, Scalar(0, 0, 0), -1, 8);
					circle(dst, Point((2*minRect2.x+minRect2.width)/2,(2*minRect2.y+minRect2.height)/2), 6, Scalar(0, 0, 0), -1, 8);
					points.push_back(Point2D((2*minRect2.x+minRect2.width)/2,(2*minRect2.y+minRect2.height)/2));
					points1.push_back(Point2D((2*minRect2.x+minRect2.width)/2,(2*minRect2.y+minRect2.height)/2));

					
					if(i==100||i==200||i==299)				// two hands swap
					//if(i==80||i==250||i==299)					// two hands gestures
					//if(i==150||i==500||i==750)					// multiple shapes with two hands
					{
						RecognitionResult r1 = g.recognize(points);
						RecognitionResult r2 = g.recognize(points1);
						cout << "Recognized gesture Hand 1: " << r1.name << endl;
						cout << "1$ Score Hand 1:" << r1.score << endl;
						cout << "Recognized gesture Hand 2: " << r2.name << endl;
						cout << "1$ Score Hand 2:" << r2.score << endl;
						
						shape = r1.name;
						shape1 = shape + ", " + r2.name;

						points.clear();
						points1.clear();
					}

					//putText(dst, shape1, Point(10,50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255,0,255),2);
				
					vconcat(hull, dst, results);
					imshow("results", results);
					imshow("depth", depthimage);
				}	
			}

			char k = waitKey(1);
			if (k==27)
			{
				destroyAllWindows();
				break;
			}	
		}
				
		else
		{
			i=0;
			shape = "Shape";
		}
	}
	
	destroyAllWindows();
	return 0;
}