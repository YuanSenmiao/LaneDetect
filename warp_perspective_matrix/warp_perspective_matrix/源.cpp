#include <opencv2\opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
	CvCapture *capture = cvCreateFileCapture("G:\\opencv\\workspace\\Lane_Detect\\Lane_Detect\\1.avi ");//ע���ȡ���ڹ���Ŀ¼���ļ�ʱ��·��Ҫ��˫б��
	//CvCapture *capture = cvCreateFileCapture("G:\\opencv\\workspace\\Lane_Detect\\Lane_Detect\\2.avi ");//ע���ȡ���ڹ���Ŀ¼���ļ�ʱ��·��Ҫ��˫б��
	IplImage *img_capture = cvQueryFrame(capture);

	//����ROI����
	// 1.avi
	int x = 0, y = 157;
	int w = img_capture->width, h = 256;

	//2.avi
	//	int x = 0, y = 270;
	//	int w = img_capture->width, h = 150;

	//   1.avi
	int alfai = 12.;//��������ˮƽ�ӽ�
	int betai = 90.;//��������ֱ�ӽ�
	int gamai = 90.;//������복���߼н�
	int focusi = 793;	//����
	int disi = 209;	//����
	
	/*   2.avi
	int alfai = 5.;//��������ˮƽ�ӽ�
	int betai = 90.;//��������ֱ�ӽ�
	int gamai = 90.;//������복���߼н�
	int focusi = 880;	//����
	int disi = 170;	//����
	*/

	//������ʾ����
	cvNamedWindow("OriginalView");
	cvNamedWindow("IPMView");
	//������Ϻ�����Ҫ������ڲ������Ѵ���xml�ļ�
	//createTrackbar("alfa", "IPMView", &alfai, 180);
	//createTrackbar("beta", "IPMView", &betai, 180);
	//createTrackbar("gamma", "IPMView", &gamai, 180);
	//createTrackbar("f", "IPMView", &focusi, 2000);
	//createTrackbar("d", "IPMView", &disi, 2000);

	Mat IPM;
	Mat source;

	int frame = 0;				//��ǰ֡��
	while (img_capture!=NULL)	//����ȡ���յ�ʱ�򣬼���Ƶ�������ֹͣ����
	{
		frame++;				//֡������
		cout << frame << endl;
		img_capture = cvQueryFrame(capture);//���϶�ȡ��һ֡
		source = cvQueryFrame(capture);
		Mat ImageCut = source(cvRect(x, y, w, h));//��ԭͼ���ROI������ͼ��ImageCut

		cvSetImageROI(img_capture, cvRect(x, y, w, h));	//����ROI		
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

		// ��ά���굽��άת������
		Mat M1 = (Mat_<double>(4,3) <<
			1, 0, -width / 2,
			0, 1, -height / 2,
			0, 0, 0,
			0, 0, 1);
		//----------------------------------
		//����������ת�任����
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

		//����������˵õ���ת����
		Mat Rotate = X * Y * Z;
		//-----------------------------------

		//��Z��͸�ӱ任����
		Mat M2 = (Mat_<double>(4, 4) <<
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, disd,
			0, 0, 0, 1
			);
		//-----------------------------------
		//����ͷ����ת������
		Mat M3 = (Mat_<double>(3, 4) <<
			focusd, 0, width / 2, 0,
			0, focusd, height / 2, 0,
			0, 0, 1, 0
			);
		//-----------------------------------

		//��ȡ��͸�ӱ任����
		Mat warp_perspective_matrix = M3 * (M2 * (Rotate * M1));
		CvMat map_matrix = warp_perspective_matrix;
		CvFileStorage *fs = cvOpenFileStorage("3.xml", 0, CV_STORAGE_WRITE);
		cvWrite(fs, "WarpMatrix", &map_matrix, cvAttrList(0, 0));
		warpPerspective(ImageCut, IPM, warp_perspective_matrix, ImageCut.size(), INTER_CUBIC | WARP_INVERSE_MAP);
		imshow("IPMView", IPM);

		char c = cvWaitKey(100);
		if (c == 27) //27���Ƕ�Ӧ�����ϵ�ESC�������û�а����̵Ļ�CΪ-1
			break;
	}
	cvReleaseCapture(&capture);

	cvWaitKey(30);

	return 0;
}