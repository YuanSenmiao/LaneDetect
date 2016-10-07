#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <opencv2\opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	/**************************   Wrap Matrix   ****************************/

	CvCapture *capture = cvCreateFileCapture("1.avi ");//ע���ȡ���ڹ���Ŀ¼���ļ�ʱ��·��Ҫ��˫б��
	IplImage *img_capture = cvQueryFrame(capture);

	//��ȡ����
	CvMemStorage *memstorageTest_wrap_matrix = cvCreateMemStorage(0);
	CvFileStorage *warp_read_wrap_matrix = cvOpenFileStorage(
		"1.xml",
		memstorageTest_wrap_matrix,
		CV_STORAGE_READ
		);//�������ڵ�xml�ļ���

	CvMat *map_matrix = cvCreateMat(3, 3, CV_32FC1);
	map_matrix = (CvMat*)cvReadByName(
		warp_read_wrap_matrix,
		NULL,
		"WarpMatrix",
		NULL
		);//��ȡ����ע��˫���������Ǿ��������

	const int stateNum = 4;
	const int measureNum = 2;
	Vector <CvKalman*> kalman;//state(x,y,detaX,detaY)
	kalman.resize(5);
	CvMat* process_noise = cvCreateMat(stateNum, 1, CV_32FC1);
	CvMat* measurement = cvCreateMat(measureNum, 1, CV_32FC1);//measurement(x,y)
	CvRNG rng = cvRNG(-1);
	float A[stateNum][stateNum] = {//transition matrix
		1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1
	};

	const int winHeight = 256;
	const int winWidth = 856;
	for (int i = 0; i <= 4; i++)
	{
		kalman[i] = cvCreateKalman(stateNum, measureNum, 0);
		memcpy(kalman[i]->transition_matrix->data.fl, A, sizeof(A));
		cvSetIdentity(kalman[i]->measurement_matrix, cvRealScalar(1));
		cvSetIdentity(kalman[i]->process_noise_cov, cvRealScalar(1e-5));
		cvSetIdentity(kalman[i]->measurement_noise_cov, cvRealScalar(1e-1));
		cvSetIdentity(kalman[i]->error_cov_post, cvRealScalar(1));
		//initialize post state of kalman filter at random
		cvRandArr(&rng, kalman[i]->state_post, CV_RAND_UNI, cvRealScalar(0), cvRealScalar(winHeight>winWidth ? winWidth : winHeight));
	}

	// 1.avi
	int x = 0, y = 157;
	int width = img_capture->width, height = 256;

	cvNamedWindow("OriginalView");	//������ʾ����
	cvNamedWindow("IPMview");

	int frame_wrap_matrix = 0;		//��ǰ֡��
	while (img_capture != NULL)	//����ȡ���յ�ʱ�򣬼���Ƶ���������ֹͣ����
	{
		frame_wrap_matrix++;
		cout << frame_wrap_matrix << endl;
		img_capture = cvQueryFrame(capture);			//���϶�ȡ��һ֡	

		cvSetImageROI(img_capture, cvRect(x, y, width, height));	//����ROI
		IplImage *ImageCut = cvCreateImage(cvGetSize(img_capture), 8, 3);
		cvCopy(img_capture, ImageCut);								//��ԭͼ���ROI������ͼ��ImageCut
		IplImage *ImageIPM = cvCreateImage(cvGetSize(ImageCut), 8, 3);	//�������ڱ任��ͼ��
		IplImage *ImageHough = cvCreateImage(cvGetSize(ImageCut), 8, 3);
		IplImage *ImageFin = cvCreateImage(cvGetSize(ImageCut), 8, 3);


		cvWarpPerspective(ImageCut, ImageIPM, map_matrix);//��͸�ӱ任
		cvShowImage("OriginalView", ImageCut);

		/***********************   ��Ƶ����任******************************/
		//srcֱ����ImageIPM
		if (!ImageIPM)
		{
			//cout << "src load error..." << endl;
			system("pause");
			exit(-1);
		}

		cvCopy(ImageIPM, ImageHough);
		// ��Ե��⣬������߾��ȣ��ҶȻ�����ʴ�����ͣ�ƽ������
		IplImage* color_dst = cvCreateImage(cvGetSize(ImageIPM), 8, 3);
		IplImage* dst = cvCreateImage(cvGetSize(ImageIPM), 8, 1);
		IplImage* gray_dst = cvCreateImage(cvGetSize(ImageIPM), 8, 1);
		IplImage* dilate_dst = cvCreateImage(cvGetSize(ImageIPM), 8, 1);
		IplImage* erode_dst = cvCreateImage(cvGetSize(ImageIPM), 8, 1);
		IplImage* smooth_dst = cvCreateImage(cvGetSize(ImageIPM), 8, 1);
		cvCvtColor(ImageIPM, gray_dst, CV_BGR2GRAY);
		cvDilate(gray_dst, dilate_dst);
		cvErode(dilate_dst, erode_dst);
		cvSmooth(erode_dst, smooth_dst);

		//cvThreshold(smooth_dst, dst, 80, 250, CV_THRESH_BINARY);
		cvCanny(smooth_dst, dst, 40, 80, 3);
		cvShowImage("IPMview", dst);

		CvMemStorage *storage = cvCreateMemStorage();
		CvSeq *lines = 0;
		vector<Point> ls, rs;
		lines = cvHoughLines2(dst, storage, CV_HOUGH_STANDARD, 1, CV_PI / 180, 35, 0, 0);
		printf("Lines number: %d\n", lines->total);

		CvPoint center;
		center = cvPoint(ImageHough->width / 1.9, ImageHough->height / 1.5);
		//----------------------------------------------------------------
		for (int i = 0; i<lines->total; i++)
		{
			float *line_1 = (float *)cvGetSeqElem(lines, i);
			float rho = line_1[0];
			float theta = line_1[1];
			CvPoint pt1, pt2;
			double a = cos(theta);
			double b = sin(theta);
			if (fabs(a)<0.001)
			{
				pt1.x = pt2.x = cvRound(rho);
				pt1.y = 0;
				pt2.y = dst->height;
			}
			else if (fabs(b)<0.001)
			{
				pt1.y = pt2.y = cvRound(rho);
				pt1.x = 0;
				pt2.x = dst->width;
			}
			else
			{
				pt1.x = 0;
				pt1.y = cvRound(rho / b);
				pt2.x = cvRound(rho / a);
				pt2.y = 0;
			}

			/*************************  ֱ��ɸѡ����� ************************/
			double y0 = (double)pt1.y - (double)pt2.y;
			double x0 = (double)pt1.x - (double)pt2.x;

			//һ��ɸѡ�� ����б��ɸѡ������ֱ���߶�
			if (x0 != 0 && (y0 / x0 > 50 || y0 / x0 < -50))
			{
				double k;
				k = y0 / x0;
				int dy = pt1.y;
				int aimy = ImageHough->height;
				double dis1 = max(pt1.x, pt2.x) - center.x;

				//����ɸѡ�������������߾���
				if (fabs(dis1) < 100)	//��ǰ��ʻ�ĳ���������
				{
					if (max(pt1.x, pt2.x) < center.x)
					{
						//cvLine(ImageHough, pt1, pt2, CV_RGB(255, 0, 0), 1, 8);
						Point pt3 = Point((pt1.x + pt2.x) / 10, (pt1.y + pt2.y) / 10);
						Point pt1_t = Point((aimy / 2 - dy) / k, aimy / 2);
						Point pt2_t = Point((aimy / 4 - dy) / k, aimy / 4);
						ls.push_back(pt1_t);
						ls.push_back(pt2_t);
					}
					else if (pt1.y < 0)
					{
						//cvLine(ImageHough, pt1, pt2, CV_RGB(0, 255, 0), 4, 8);
						rs.push_back(pt1);
						rs.push_back(pt2);
						Point pt1_t = Point((aimy / 2 - dy) / k, aimy / 2);
						Point pt2_t = Point((aimy / 4 - dy) / k, aimy / 4);
						rs.push_back(pt1_t);
						rs.push_back(pt2_t);
					}
				}
				/*else  if (fabs(dis1) < 160 && fabs(dis1) > 100)//�ǵ�ǰ��ʻ�ĳ������ߵ���
				{
				cvLine(ImageHough, pt1, pt2, CV_RGB(0, 0, 255), 2, 8);//�ý�ϸ��ɫ�߻���
				}*/
				//--------------------------------------
			}
		}

		//Left lane
		if (!ls.empty())
		{
			//�������ֱ�ߵ�����
			Vec4f line_2;
			//ֱ����Ϻ���
			fitLine(Mat(ls), line_2, CV_DIST_L2, 0, 0.01, 0.01);
			int lx0 = line_2[2] - 10000 * line_2[0], ly0 = line_2[3] - 10000 * line_2[1];
			int lx1 = line_2[2] + 10000 * line_2[0], ly1 = line_2[3] + 10000 * line_2[1];

			//2.kalman prediction
			const CvMat* prediction0 = cvKalmanPredict(kalman[0], 0);
			CvPoint predict_pt0 = cvPoint((int)prediction0->data.fl[0], (int)prediction0->data.fl[1]);
			//3.update measurement
			measurement->data.fl[0] = lx0;
			measurement->data.fl[1] = ly0;
			//4.update
			cvKalmanCorrect(kalman[0], measurement);

			//2.kalman prediction
			const CvMat* prediction1 = cvKalmanPredict(kalman[1], 0);
			CvPoint predict_pt1 = cvPoint((int)prediction1->data.fl[0], (int)prediction1->data.fl[1]);
			//3.update measurement
			measurement->data.fl[0] = lx1;
			measurement->data.fl[1] = ly1;
			//4.update
			cvKalmanCorrect(kalman[1], measurement);

			double kl2 = line_2[1] / line_2[0];
			//����ɸѡ����ֹ��Ϊ�㲻������������ֱ����Ϻ������������ˮƽ�ߣ���б���ٴ�ɸ��ˮƽ��
			if (kl2 > 50 || kl2 < -50)
			{
				CvPoint tmp0, tmp1;
				tmp0.x = predict_pt0.x + 5; tmp0.y = predict_pt0.y;
				tmp1.x = predict_pt1.x + 5; tmp1.y = predict_pt1.y;
				cvLine(ImageHough, tmp0, tmp1, CV_RGB(0, 255, 0), 1, CV_AA);//kalman Line
				tmp0.x = predict_pt0.x - 10; tmp0.y = predict_pt0.y;
				tmp1.x = predict_pt1.x - 10; tmp1.y = predict_pt1.y;
				cvLine(ImageHough, tmp0, tmp1, CV_RGB(0, 255, 0), 1, CV_AA);//kalman Line
				cvLine(ImageHough, cvPoint(lx0, ly0), cvPoint(lx1, ly1), CV_RGB(255, 0, 0), 3, CV_AA);//���ֱ���üӴֺ��߻���
			}
				cout << lx0 << " " << ly0 << ", " << lx1 << " " << ly1 << endl;
				cout << predict_pt0.x << " " << predict_pt0.y << ", " << predict_pt1.x << " " << predict_pt1.y << endl;
		}

		//Right lane
		if (!rs.empty())
		{
			//�������ֱ�ߵ�����
			Vec4f line_3;
			//ֱ����Ϻ���
			fitLine(Mat(rs), line_3, CV_DIST_L2, 0, 0.01, 0.01);
			int rx0 = line_3[2] - 10000 * line_3[0], ry0 = line_3[3] - 10000 * line_3[1];
			int rx1 = line_3[2] + 10000 * line_3[0], ry1 = line_3[3] + 10000 * line_3[1];

			//2.kalman prediction
			const CvMat* prediction2 = cvKalmanPredict(kalman[2], 0);
			CvPoint predict_pt2 = cvPoint((int)prediction2->data.fl[0], (int)prediction2->data.fl[1]);
			//3.update measurement
			measurement->data.fl[0] = rx0;
			measurement->data.fl[1] = ry0;
			//4.update
			cvKalmanCorrect(kalman[2], measurement);

			//2.kalman prediction
			const CvMat* prediction3 = cvKalmanPredict(kalman[3], 0);
			CvPoint predict_pt3 = cvPoint((int)prediction3->data.fl[0], (int)prediction3->data.fl[1]);
			//3.update measurement
			measurement->data.fl[0] = rx1;
			measurement->data.fl[1] = ry1;
			//4.update
			cvKalmanCorrect(kalman[3], measurement);

			//����ɸѡ����ֹ��Ϊ�㲻������������ֱ����Ϻ������������ˮƽ�ߣ���б���ٴ�ɸ��ˮƽ��
			double kl3 = line_3[1] / line_3[0];
			if (kl3 > 50 || kl3 < -50)
			{
				CvPoint tmp0, tmp1;
				tmp0.x = predict_pt2.x + 5; tmp0.y = predict_pt2.y;
				tmp1.x = predict_pt3.x + 5; tmp1.y = predict_pt3.y;
				cvLine(ImageHough, tmp0, tmp1, CV_RGB(0, 255, 0), 1, CV_AA);//kalman Line
				tmp0.x = predict_pt2.x - 10; tmp0.y = predict_pt2.y;
				tmp1.x = predict_pt3.x - 10; tmp1.y = predict_pt3.y;
				cvLine(ImageHough, tmp0, tmp1, CV_RGB(0, 255, 0), 1, CV_AA);//kalman Line
				cvLine(ImageHough, cvPoint(rx0, ry0), cvPoint(rx1, ry1), CV_RGB(255, 0, 0), 3, CV_AA);//���ֱ���üӴֺ��߻���

			}
			cout << rx0 << " " << ry0 << ", " << rx1 << " " << ry1 << endl;
			cout << predict_pt2.x << " " << predict_pt2.y << ", " << predict_pt3.x << " " << predict_pt3.y << endl;
		}

		//---------------------------------
		cvShowImage("houghlines", ImageHough);

		// Invert map_matrix
		CvMat *inv_map_matrix = cvCreateMat(3, 3, CV_32FC1);
		cvInvert(map_matrix, inv_map_matrix);
		cvWarpPerspective(ImageHough, ImageFin, inv_map_matrix);
		cvNamedWindow("Final", 1);
		cvShowImage("Final", ImageFin);
		//---------------------------------
		char c = cvWaitKey(10);
		if (c == 27) //27���Ƕ�Ӧ�����ϵ�ESC�������û�а����̵Ļ�CΪ-1
			break;

		cvReleaseImage(&dst);
		cvReleaseImage(&color_dst);
		cvClearSeq(lines);
		cvReleaseMemStorage(&storage);
		cvReleaseImage(&ImageCut);
		cvReleaseImage(&ImageIPM);
	}

	cvReleaseCapture(&capture);
	cvWaitKey(0);

	return 0;
}