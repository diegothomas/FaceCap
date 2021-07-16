#ifndef __UTILITY_H
#define __UTILITY


class Utility
{
public:
	Utility();
	~Utility();

	/*******************************************************************/
	/*******Functions and variables for face feature detector***********/
	/*******************************************************************/
	static void drawFaceHelper(cv::Mat& img, const cv::Mat& X, int start, int end, bool circle = false);
	static void fillFaceHelper(cv::Mat& img, const cv::Mat& X, int start, int end, cv::Scalar& colordown, cv::Scalar& colorup);
	static void drawFace(cv::Mat& img, const cv::Mat& X);
	static void drawFacePoints(cv::Mat& img, const cv::Mat& X);
	static void drawPose(cv::Mat& img, const cv::Mat& rot);

	static bool compareRect(cv::Rect r1, cv::Rect r2);
	static Eigen::VectorXd ParallelRelaxation(Eigen::MatrixXd Q_inv, Eigen::VectorXd x0, Eigen::VectorXd lb, Eigen::VectorXd ub);

	/*******************************************************************/
	/****END**Functions and variables for face feature detector*********/
	/*******************************************************************/

	//static cl_kernel LoadKernel(string filename, string Kernelname, cl_context context, cl_device_id device);

	static void GetWeightedNormal(MyMesh *TheMesh, Face *triangle, float *nmle);
	static void getWeightsB(float *weights, float u, float v, TextUV uv1, TextUV uv2, TextUV uv3);
	static void getWeights(float *weights, float u, float v, TextUV uv1, TextUV uv2, TextUV uv3);
	static void DrawTriangle(MyMesh *TheMesh, Face *triangle, float *data, USHORT color);
	static bool IsInTriangle(MyMesh *TheMesh, Face *triangle, int i, int j);
};

#endif