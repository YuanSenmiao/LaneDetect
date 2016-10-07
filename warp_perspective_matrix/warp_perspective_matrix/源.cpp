#include <opencv2\opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
	CvCapture *capture = cvCreateFileCapture("G:\\opencv\\workspace\\Lane_Detect\\Lane_Detect\\1.avi ");//注意读取不在工程目录下文件时的路径要加双斜杠
	//CvCapture *capture = cvCreateFileCapture("G:\\opencv\\workspace\\Lane_Detect\\Lane_Detect\\2.avi ");//注意读取不在工程目录下文件时的路径要加双斜杠
	IplImage *img_capture = cvQueryFrame(capture);

	//设置ROI参数
	// 1.avi
	int x = 0, y = 157;
	int w = img_capture->width, h = 256;

	//2.avi
	//	int x = 0, y = 270;
	//	int w = img_capture->width, h = 150;

	//   1.avi
	int alfai = 12.;//摄像机最大水平视角
	int betai = 90.;//摄像机最大垂直视角
	int gamai = 90.;//摄像机与车道线夹角
	int focusi = 793;	//焦距
	int disi = 209;	//距离
	
	/*   2.avi
	int alfai = 5.;//摄像机最大水平视角
	int betai = 90.;//摄像机最大垂直视角
	int gamai = 90.;//摄像机与车道线夹角
	int focusi = 880;	//焦距
	int disi = 170;	//距离
	*/

	//创建显示窗口
	cvNamedWindow("OriginalView");
	cvNamedWindow("IPMView");
	//调试完毕后不再需要滑块调节参数，已储存xml文件
	//createTrackbar("alfa", "IPMView", &alfai, 180);
	//createTrackbar("beta", "IPMView", &betai, 180);
	//createTrackbar("gamma", "IPMView", &gamai, 180);
	//createTrackbar("f", "IPMView", &focusi, 2000);
	//createTrackbar("d", "IPMView", &disi, 2000);

	Mat IPM;
	Mat source;

	int frame = 0;				//当前帧数
	while (img_capture!=NULL)	//当读取到空的时候，即视频播放完毕停止处理
	{
		frame++;				//帧数更新
		cout << frame << endl;
		img_capture = cvQueryFrame(capture);//不断读取下一帧
		source = cvQueryFrame(capture);
		Mat ImageCut = source(cvRect(x, y, w, h));//将原图像的ROI赋给新图像ImageCut

		cvSetImageROI(img_capture, cvRect(x, y, w, h));	//设置ROI		
		imshow("OriginalView", ImageCut);
		//------------------------------------

		double focusd = (double)focusi;
		double disd = (double)disi;
		double alfad = ((double)alfai - 90.) * CV_PI / 180;
		double betad = ((double)betai - 90.) * CV_PI / 180;
		double gamad = ((double)gamai - 90.) * CV_PI / 180;

		Size size = ImageCut.size();
		double width = (double)size.width, height = (double)size.height;
		//----------------------------------

		// 二维坐标到三维转换矩阵
		Mat M1 = (Mat_<double>(4,3) <<
			1, 0, -width / 2,
			0, 1, -height / 2,
			0, 0, 0,
			0, 0, 1);
		//----------------------------------
		//各个方向旋转变换矩阵
		Mat X = (Mat_<double>(4, 4) <<
			1, 0, 0, 0,
			0, cos(alfad), -sin(alfad), 0,
			0, sin(alfad), cos(alfad), 0,
			0, 0, 0, 1
			);

		Mat Y = (Mat_<double>(4, 4) <<
			cos(betad), 0, -sin(betad), 0,
			0, 1, 0, 0,
			sin(betad), 0, cos(betad), 0,
			0, 0, 0, 1);

		Mat Z = (Mat_<double>(4, 4) <<
			cos(gamad), -sin(gamad), 0, 0,
			sin(gamad), cos(gamad), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);

		//三个矩阵相乘得到旋转矩阵
		Mat Rotate = X * Y * Z;
		//-----------------------------------

		//沿Z轴透视变换矩阵
		Mat M2 = (Mat_<double>(4, 4) <<
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, disd,
			0, 0, 0, 1
			);
		//-----------------------------------
		//摄像头坐标转换矩阵
		Mat M3 = (Mat_<double>(3, 4) <<
			focusd, 0, width / 2, 0,
			0, focusd, height / 2, 0,
			0, 0, 1, 0
			);
		//-----------------------------------

		//求取反透视变换矩阵
		Mat warp_perspective_matrix = M3 * (M2 * (Rotate * M1));
		CvMat map_matrix = warp_perspective_matrix;
		CvFileStorage *fs = cvOpenFileStorage("3.xml", 0, CV_STORAGE_WRITE);
		cvWrite(fs, "WarpMatrix", &map_matrix, cvAttrList(0, 0));
		warpPerspective(ImageCut, IPM, warp_perspective_matrix, ImageCut.size(), INTER_CUBIC | WARP_INVERSE_MAP);
		imshow("IPMView", IPM);

		char c = cvWaitKey(100);
		if (c == 27) //27就是对应键盘上的ESC建，如果没有按键盘的话C为-1
			break;
	}
	cvReleaseCapture(&capture);

	cvWaitKey(30);

	return 0;
}