#include "stdafx.h"
#include "utility.h"

cv::Scalar BLUE(255, 0, 0);
cv::Scalar GREEN(0, 255, 0);
cv::Scalar RED(0, 0, 255);
cv::Scalar FACECOLOR(50, 255, 50);

cv::Scalar LEFTEYECOLORDOWN(255, 0, 0);
cv::Scalar LEFTEYECOLORUP(255, 255, 0);
cv::Scalar RIGHTEYECOLORDOWN(0, 255, 0);
cv::Scalar RIGHTEYECOLORUP(0, 255, 255);
cv::Scalar MOUTHCOLORDOWN(0, 0, 255);
cv::Scalar MOUTHCOLORUP(255, 0, 255);

cv::Vec3b LEFTEYEUP(255, 0, 0);
cv::Vec3b LEFTEYEDOWN(0, 255, 0);
cv::Vec3b RIGHTEYEUP(0, 0, 255);
cv::Vec3b RIGHTEYEDOWN(255, 0, 255);
cv::Vec3b LIPUP(255, 255, 0);
cv::Vec3b LIPDOWN(0, 255, 255);

Utility::Utility()
{

}

Utility::~Utility()
{

}

void Utility::drawFaceHelper(cv::Mat& img, const cv::Mat& X, int start, int end, bool circle)
{
	int thickness = 1;
	int lineType = CV_AA;
	for (int i = start; i < end; i++) {
		line(img, cv::Point(X.at<float>(0, i), X.at<float>(1, i)),
			cv::Point(X.at<float>(0, i + 1), X.at<float>(1, i + 1)),
			FACECOLOR, thickness, lineType);
	}
	if (circle) {
		line(img, cv::Point(X.at<float>(0, end), X.at<float>(1, end)),
			cv::Point(X.at<float>(0, start), X.at<float>(1, start)),
			FACECOLOR, thickness, lineType);
	}
}

void Utility::fillFaceHelper(cv::Mat& img, const cv::Mat& X, int start, int end, cv::Scalar& colordown, cv::Scalar& colorup)
{

	float scale = sqrt((X.at<float>(0, start + (end - start) / 2) - X.at<float>(0, start)) * (X.at<float>(0, start + (end - start) / 2) - X.at<float>(0, start)) +
		(X.at<float>(1, start + (end - start) / 2) - X.at<float>(1, start)) * (X.at<float>(1, start + (end - start) / 2) - X.at<float>(1, start)));

	int thickness = 1;
	int lineType = CV_AA;
	cv::Point Quad[4];
	for (int i = start; i <= end; i++) {

		//compute normal vector
		int k = i == start ? end : i - 1;
		int l = i == end ? start : i + 1;
		float nmle[2];
		nmle[0] = (X.at<float>(1, l) - X.at<float>(1, k));
		nmle[1] = -(X.at<float>(0, l) - X.at<float>(0, k));
		float val = sqrt(nmle[0] * nmle[0] + nmle[1] * nmle[1]);
		nmle[0] = nmle[0] / val;
		nmle[1] = nmle[1] / val;

		Quad[0] = cv::Point(X.at<float>(0, i), X.at<float>(1, i));
		Quad[1] = cv::Point(X.at<float>(0, i) + nmle[0] * (scale / 2.0f), X.at<float>(1, i) + nmle[1] * (scale / 2.0f));

		//compute normal vector
		k = i;
		l = i + 1 > end - 1 ? start + 1 : i + 2;
		nmle[0] = (X.at<float>(1, l) - X.at<float>(1, k));
		nmle[1] = -(X.at<float>(0, l) - X.at<float>(0, k));
		val = sqrt(nmle[0] * nmle[0] + nmle[1] * nmle[1]);
		nmle[0] = nmle[0] / val;
		nmle[1] = nmle[1] / val;

		if (i == end) {
			Quad[3] = cv::Point(X.at<float>(0, start), X.at<float>(1, start));
			Quad[2] = cv::Point(X.at<float>(0, start) + nmle[0] * (scale / 2.0f), X.at<float>(1, start) + nmle[1] * (scale / 2.0f));
		}
		else {
			Quad[3] = cv::Point(X.at<float>(0, i + 1), X.at<float>(1, i + 1));
			Quad[2] = cv::Point(X.at<float>(0, i + 1) + nmle[0] * (scale / 2.0f), X.at<float>(1, i + 1) + nmle[1] * (scale / 2.0f));
		}

		if (i > start + (end - start) / 2)
			fillConvexPoly(img, Quad, 4, colordown, lineType);
		else
			fillConvexPoly(img, Quad, 4, colorup, lineType);
	}
}

void Utility::drawFace(cv::Mat& img, const cv::Mat& X)
{
	//// left eyebrow
	//drawFaceHelper(img, X, 0, 4);
	//// right eyebrow
	//drawFaceHelper(img, X, 5, 9);
	//// nose
	//drawFaceHelper(img, X, 10, 13);
	//// under nose
	//drawFaceHelper(img, X, 14, 18);
	//// left eye
	//drawFaceHelper(img, X, 19, 24, true);
	//// right eye
	//drawFaceHelper(img, X, 25, 30, true);
	//// mouth contour
	//drawFaceHelper(img, X, 31, 42, true);
	//// inner mouth
	//drawFaceHelper(img, X, 43, 48, true);
	//// contour
	//if (X.cols > 49)
	//	drawFaceHelper(img, X, 49, 65);

	//// draw points
	//for (int i = 0; i < X.cols; i++)
	//	cv::circle(img, cv::Point((int)X.at<float>(0, i), (int)X.at<float>(1, i)), 1, FACECOLOR, -1);

	// Fill left eye (right for user)
	Utility::fillFaceHelper(img, X, 19, 24, LEFTEYECOLORDOWN, LEFTEYECOLORUP);
	// Fill right eye (left for user)
	Utility::fillFaceHelper(img, X, 25, 30, RIGHTEYECOLORDOWN, RIGHTEYECOLORUP);

	// Fill inner mouth
	cv::Mat Mouth = cv::Mat::zeros(2, 8, CV_32F);
	Mouth.at<float>(0, 0) = X.at<float>(0, 31); Mouth.at<float>(1, 0) = X.at<float>(1, 31);
	Mouth.at<float>(0, 1) = X.at<float>(0, 43); Mouth.at<float>(1, 1) = X.at<float>(1, 43);
	Mouth.at<float>(0, 2) = X.at<float>(0, 44); Mouth.at<float>(1, 2) = X.at<float>(1, 44);
	Mouth.at<float>(0, 3) = X.at<float>(0, 45); Mouth.at<float>(1, 3) = X.at<float>(1, 45);
	Mouth.at<float>(0, 4) = X.at<float>(0, 37); Mouth.at<float>(1, 4) = X.at<float>(1, 37);
	Mouth.at<float>(0, 5) = X.at<float>(0, 46); Mouth.at<float>(1, 5) = X.at<float>(1, 46);
	Mouth.at<float>(0, 6) = X.at<float>(0, 47); Mouth.at<float>(1, 6) = X.at<float>(1, 47);
	Mouth.at<float>(0, 7) = X.at<float>(0, 48); Mouth.at<float>(1, 7) = X.at<float>(1, 48);
	Utility::fillFaceHelper(img, Mouth, 0, 7, MOUTHCOLORDOWN, MOUTHCOLORUP);
}

void Utility::drawFacePoints(cv::Mat& img, const cv::Mat& X)
{
	//// left eyebrow
	//drawFaceHelper(img, X, 0, 4);
	//// right eyebrow
	//drawFaceHelper(img, X, 5, 9);
	//// nose
	//drawFaceHelper(img, X, 10, 13);
	//// under nose
	//drawFaceHelper(img, X, 14, 18);
	//// left eye
	//drawFaceHelper(img, X, 19, 24, true);
	//// right eye
	//drawFaceHelper(img, X, 25, 30, true);
	//// mouth contour
	//drawFaceHelper(img, X, 31, 42, true);
	//// inner mouth
	//drawFaceHelper(img, X, 43, 48, true);
	//// contour
	//if (X.cols > 49)
	//	drawFaceHelper(img, X, 49, 65);

	// draw points
	for (int i = 0; i < X.cols; i++)
		cv::circle(img, cv::Point((int)X.at<float>(0, i), (int)X.at<float>(1, i)), 1, FACECOLOR, -1);
}

void Utility::drawPose(cv::Mat& img, const cv::Mat& rot)
{
	int loc[2] = { 70, 70 };
	int thickness = 2;
	int lineType = CV_AA;
	float lineL = 50.f;

	cv::Mat P = (cv::Mat_<float>(3, 4) <<
		0, lineL, 0, 0,
		0, 0, -lineL, 0,
		0, 0, 0, -lineL);
	P = rot.rowRange(0, 2)*P;
	P.row(0) += loc[0];
	P.row(1) += loc[1];
	cv::Point p0(P.at<float>(0, 0), P.at<float>(1, 0));

	line(img, p0, cv::Point(P.at<float>(0, 1), P.at<float>(1, 1)), BLUE, thickness, lineType);
	line(img, p0, cv::Point(P.at<float>(0, 2), P.at<float>(1, 2)), GREEN, thickness, lineType);
	line(img, p0, cv::Point(P.at<float>(0, 3), P.at<float>(1, 3)), RED, thickness, lineType);
}

bool Utility::compareRect(cv::Rect r1, cv::Rect r2)
{
	return r1.height < r2.height;
}

// Implement Paraleele Relaxation algorithm 2.1. in paper [A parallel relaxation method for quadratic programming problems with interval constraints]
Eigen::VectorXd Utility::ParallelRelaxation(Eigen::MatrixXd Q_inv, Eigen::VectorXd x0, Eigen::VectorXd lb, Eigen::VectorXd ub) 
{
	int n = x0.size();
	double w = 0.02;
	int max_iter = 1000;// 30000;
	Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd Delta = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd Gamma = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd xres = x0;
	Eigen::VectorXd S = Eigen::VectorXd::Zero(n);

	/*double theta;
	double w_min = 1.0;
	for (int i = 0; i < n; i++) {
	double val = 0.0;
	for (int j = 0; j < n; j++) {
	if (i == j)
	continue;
	val += abs(Q_inv(i, j));
	}
	theta = (2.0 / Q_inv(i, i)) * val;
	double w_curr = min(1 / theta, 3.0 / (2.0 + theta));
	w_min = min(w_min, w_curr);
	}
	cout << "w_max: " <<  w_min << endl;*/

	bool converged = false;
	int iter = 0;
	while (!converged) {

		for (int i = 0; i < n; i++) {
			Delta(i) = (lb(i) - xres(i)) / Q_inv(i, i);
			Gamma(i) = (ub(i) - xres(i)) / Q_inv(i, i);
			S(i) = MyMedian(u(i), w*Delta(i), w*Gamma(i));
		}
		u = u - S;

		xres = xres + Q_inv * S;

		iter++;
		converged = (iter > max_iter || S.norm() < 1.0e-10);
	}

	//cout << "number of inner loops: " << iter << "S.norm(): " << S.norm() << endl;
	return xres;
}

//cl_kernel Utility::LoadKernel(string filename, string Kernelname, cl_context context, cl_device_id device) 
//{
//	cl_int ret;
//	std::ifstream file(filename);
//	checkErr(file.is_open() ? CL_SUCCESS : -1, filename.c_str());
//	std::string prog(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
//	const char * code = prog.c_str();
//	cl_program lProgram = clCreateProgramWithSource(context, 1, &code, 0, &ret);
//	ret = clBuildProgram(lProgram, 1, &device, "", 0, 0);
//	checkErr(ret, "Program::build()");
//
//	cl_kernel kernel = clCreateKernel(lProgram, Kernelname.c_str(), &ret);
//	checkErr(ret, (Kernelname + string("::Kernel()")).c_str());
//	return kernel;
//}

void Utility::GetWeightedNormal(MyMesh *TheMesh, Face *triangle, float *nmle)
{
	double v1[3];
	double v2[3];
	double b[3];
	double h[3];

	v1[0] = double(TheMesh->_vertices[triangle->_v2]->_x - TheMesh->_vertices[triangle->_v1]->_x);
	v1[1] = double(TheMesh->_vertices[triangle->_v2]->_y - TheMesh->_vertices[triangle->_v1]->_y);
	v1[2] = double(TheMesh->_vertices[triangle->_v2]->_z - TheMesh->_vertices[triangle->_v1]->_z);

	v2[0] = double(TheMesh->_vertices[triangle->_v3]->_x - TheMesh->_vertices[triangle->_v1]->_x);
	v2[1] = double(TheMesh->_vertices[triangle->_v3]->_y - TheMesh->_vertices[triangle->_v1]->_y);
	v2[2] = double(TheMesh->_vertices[triangle->_v3]->_z - TheMesh->_vertices[triangle->_v1]->_z);

	double nrm = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
	b[0] = v1[0] / nrm;
	b[1] = v1[1] / nrm;
	b[2] = v1[2] / nrm;
	double proj = b[0] * v2[0] + b[1] * v2[1] + b[2] * v2[2];
	h[0] = v2[0] - proj*b[0];
	h[1] = v2[1] - proj*b[1];
	h[2] = v2[2] - proj*b[2];
	double hauteur = sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
	double area = nrm * hauteur / 2.0;

	double tmp[3];
	tmp[0] = v1[1] * v2[2] - v1[2] * v2[1];
	tmp[1] = -v1[0] * v2[2] + v1[2] * v2[0];
	tmp[2] = v1[0] * v2[1] - v1[1] * v2[0];

	nrm = sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1] + tmp[2] * tmp[2]);
	nmle[0] = float(tmp[0] / nrm)*area;
	nmle[1] = float(tmp[1] / nrm)*area;
	nmle[2] = float(tmp[2] / nrm)*area;
}


void Utility::getWeightsB(float *weights, float u, float v, TextUV uv1, TextUV uv2, TextUV uv3) 
{
	if (u == uv1._u && v == uv1._v) {
		weights[0] = 1.0f;
		weights[1] = 0.0f;
		weights[2] = 0.0f;
		return;
	}
	if (u == uv2._u && v == uv2._v) {
		weights[0] = 0.0f;
		weights[1] = 1.0f;
		weights[2] = 0.0f;
		return;
	}
	if (u == uv3._u && v == uv3._v) {
		weights[0] = 0.0f;
		weights[1] = 0.0f;
		weights[2] = 1.0f;
		return;
	}

	// test if flat triangle
	TextUV e0 = uv2 - uv1;
	TextUV e1 = uv3 - uv1;

	if ((e0._u*e1._v - e0._v*e1._u) == 0.0f) { // flat triangle
											   // find two extrema
		if ((e0._u *e1._u + e0._v*e1._v) <= 0.0f) { // extrema are v2 and v3 since e0 and e1 are in opposite direction
			weights[0] = 0.0f;
			weights[1] = 1.0f / (1.0f + sqrt((u - uv2._u)*(u - uv2._u) + (v - uv2._v)*(v - uv2._v)));
			weights[2] = 1.0f / (1.0f + sqrt((u - uv3._u)*(u - uv3._u) + (v - uv3._v)*(v - uv3._v)));
			return;
		}

		e0 = uv1 - uv2;
		e1 = uv3 - uv2;
		if ((e0._u *e1._u + e0._v*e1._v) <= 0.0f) { // extrema are v1 and v3 since e0 and e1 are in opposite direction
			weights[0] = 1.0f / (1.0f + sqrt((u - uv1._u)*(u - uv1._u) + (v - uv1._v)*(v - uv1._v)));
			weights[1] = 0.0f;
			weights[2] = 1.0f / (1.0f + sqrt((u - uv3._u)*(u - uv3._u) + (v - uv3._v)*(v - uv3._v)));
			return;
		}

		e0 = uv1 - uv3;
		e1 = uv2 - uv3;
		if ((e0._u *e1._u + e0._v*e1._v) <= 0.0f) { // extrema are v1 and v2 since e0 and e1 are in opposite direction
			weights[0] = 1.0f / (1.0f + sqrt((u - uv1._u)*(u - uv1._u) + (v - uv1._v)*(v - uv1._v)));
			weights[1] = 1.0f / (1.0f + sqrt((u - uv2._u)*(u - uv2._u) + (v - uv2._v)*(v - uv2._v)));
			weights[2] = 0.0f;
			return;
		}

		cout << "Missing case!!" << endl;
		return;
	}

	double A[2];
	double tmp;
	double B[2];

	// Compute percentage of ctrl point 1
	A[0] = uv3._u - uv2._u;
	A[1] = uv3._v - uv2._v;
	tmp = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = tmp == 0.0 ? 0.0 : A[0] / tmp;
	A[1] = tmp == 0.0 ? 0.0 : A[1] / tmp;

	B[0] = u - uv1._u;
	B[1] = v - uv1._v;
	tmp = sqrt(B[0] * B[0] + B[1] * B[1]);
	B[0] = tmp == 0.0 ? 0.0 : B[0] / tmp;
	B[1] = tmp == 0.0 ? 0.0 : B[1] / tmp;

	double num, den, lambda;
	if (B[0] != 0.0) {
		num = uv1._v - uv2._v + (uv2._u - uv1._u)*(B[1] / B[0]);
		den = A[1] - A[0] * (B[1] / B[0]);
	}
	else {
		num = uv1._u - uv2._u;
		den = A[0];
	}

	if (den != 0.0) {
		lambda = num / den;
	}
	else {
		//if ((fabs(A[0] + B[0]) < 1.0e-6 && fabs(A[1] + B[1]) < 1.0e-6) || (fabs(A[0] - B[0]) < 1.0e-6 && fabs(A[1] - B[1]) < 1.0e-6) || (A[0] == 0.0 && A[1] == 0.0)) { // flat triangle
		//	lambda = 0.0;
		//}
		//else {
		cout << "den nul 0" << endl;
		return;
		//}
	}


	double inter_pos[2];
	inter_pos[0] = uv2._u + lambda*A[0];
	inter_pos[1] = uv2._v + lambda*A[1];

	A[0] = inter_pos[0] - uv1._u;
	A[1] = inter_pos[1] - uv1._v;
	double val = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = inter_pos[0] - u;
	A[1] = inter_pos[1] - v;
	weights[0] = float(sqrt(A[0] * A[0] + A[1] * A[1]));
	weights[0] = val == 0.0 ? 0.0f : float(weights[0] / val);

	// Compute percentage of ctrl point 2
	A[0] = uv1._u - uv3._u;
	A[1] = uv1._v - uv3._v;
	tmp = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = tmp == 0.0 ? 0.0 : A[0] / tmp;
	A[1] = tmp == 0.0 ? 0.0 : A[1] / tmp;

	B[0] = u - uv2._u;
	B[1] = v - uv2._v;
	tmp = sqrt(B[0] * B[0] + B[1] * B[1]);
	B[0] = tmp == 0.0 ? 0.0 : B[0] / tmp;
	B[1] = tmp == 0.0 ? 0.0 : B[1] / tmp;

	if (B[0] != 0.0) {
		num = uv2._v - uv3._v + (uv3._u - uv2._u)*(B[1] / B[0]);
		den = A[1] - A[0] * (B[1] / B[0]);
	}
	else {
		num = uv2._u - uv3._u;
		den = A[0];
	}

	if (den != 0.0) {
		lambda = num / den;
	}
	else {
		//if ((fabs(A[0] + B[0]) < 1.0e-6 && fabs(A[1] + B[1]) < 1.0e-6) || (fabs(A[0] - B[0]) < 1.0e-6 && fabs(A[1] - B[1]) < 1.0e-6) || (A[0] == 0.0 && A[1] == 0.0)) { // flat triangle
		//	lambda = 0.0;
		//}
		//else {
		cout << "den nul 1" << endl;
		//	cout << "u v " << u << " " << v << endl;
		//	cout << "u1 v1 " << uv1._u << " " << uv1._v << endl;
		//	cout << "u2 v2 " << uv2._u << " " << uv2._v << endl;
		//	cout << "u3 v3 " << uv3._u << " " << uv3._v << endl;
		//	cout << "A " << A[0] << " " << A[1] << endl;
		//	cout << "B " << B[0] << " " << B[1] << endl;
		//	weights[0] = -1.0f;
		//	int tmp;
		//	cin >> tmp;
		//	weights[0] = -1.0f;
		return;
		//}
	}

	inter_pos[0] = uv3._u + lambda*A[0];
	inter_pos[1] = uv3._v + lambda*A[1];

	A[0] = inter_pos[0] - uv2._u;
	A[1] = inter_pos[1] - uv2._v;
	val = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = inter_pos[0] - u;
	A[1] = inter_pos[1] - v;
	weights[1] = float(sqrt(A[0] * A[0] + A[1] * A[1]));
	weights[1] = val == 0.0 ? 0.0f : float(weights[1] / val);

	// Compute percentage of ctrl point 3
	A[0] = uv1._u - uv2._u;
	A[1] = uv1._v - uv2._v;
	tmp = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = tmp == 0.0 ? 0.0 : A[0] / tmp;
	A[1] = tmp == 0.0 ? 0.0 : A[1] / tmp;

	B[0] = u - uv3._u;
	B[1] = v - uv3._v;
	tmp = sqrt(B[0] * B[0] + B[1] * B[1]);
	B[0] = tmp == 0.0 ? 0.0 : B[0] / tmp;
	B[1] = tmp == 0.0 ? 0.0 : B[1] / tmp;

	if (B[0] != 0.0) {
		num = uv3._v - uv2._v + (uv2._u - uv3._u)*(B[1] / B[0]);
		den = A[1] - A[0] * (B[1] / B[0]);
	}
	else {
		num = uv3._u - uv2._u;
		den = A[0];
	}

	if (den != 0.0) {
		lambda = num / den;
	}
	else {
		//if ((fabs(A[0] + B[0]) < 1.0e-6 && fabs(A[1] + B[1]) < 1.0e-6) || (fabs(A[0] - B[0]) < 1.0e-6 && fabs(A[1] - B[1]) < 1.0e-6) || (A[0] == 0.0 && A[1] == 0.0)) { // flat triangle
		//	lambda = 0.0;
		//}
		//else {
		cout << "den nul 2" << endl;
		//weights[0] = -1.0f;
		return;
		//}
	}

	inter_pos[0] = uv2._u + lambda*A[0];
	inter_pos[1] = uv2._v + lambda*A[1];

	A[0] = inter_pos[0] - uv3._u;
	A[1] = inter_pos[1] - uv3._v;
	val = sqrt(A[0] * A[0] + A[1] * A[1]);
	A[0] = inter_pos[0] - u;
	A[1] = inter_pos[1] - v;
	weights[2] = float(sqrt(A[0] * A[0] + A[1] * A[1]));
	weights[2] = val == 0.0 ? 0.0f : float(weights[2] / val);
	return;
}

void Utility::getWeights(float *weights, float u, float v, TextUV uv1, TextUV uv2, TextUV uv3) 
{
	weights[0] = 1.0f / (1.0f + sqrt((u - uv1._u)*(u - uv1._u) + (v - uv1._v)*(v - uv1._v)));
	weights[1] = 1.0f / (1.0f + sqrt((u - uv2._u)*(u - uv2._u) + (v - uv2._v)*(v - uv2._v)));
	weights[2] = 1.0f / (1.0f + sqrt((u - uv3._u)*(u - uv3._u) + (v - uv3._v)*(v - uv3._v)));
	return;
}

void Utility::DrawTriangle(MyMesh *TheMesh, Face *triangle, float *data, USHORT color) 
{
	TextUV v0 = TextUV(round(TheMesh->_uvs[triangle->_t1]->_u*float(BumpHeight)), round(TheMesh->_uvs[triangle->_t1]->_v*float(BumpWidth)));
	TextUV v1 = TextUV(round(TheMesh->_uvs[triangle->_t2]->_u*float(BumpHeight)), round(TheMesh->_uvs[triangle->_t2]->_v*float(BumpWidth)));
	TextUV v2 = TextUV(round(TheMesh->_uvs[triangle->_t3]->_u*float(BumpHeight)), round(TheMesh->_uvs[triangle->_t3]->_v*float(BumpWidth)));

	//Compute the three edges of the triangle and line equation
	TextUV e0 = v1 - v0;
	float lambda0 = (v1._u - v0._u) == 0.0 ? 1.0e12 : (v1._v - v0._v) / (v1._u - v0._u);
	float c0 = lambda0 > 1.0e10 ? 0.0 : v0._v - lambda0*v0._u;
	TextUV e1 = v2 - v0;
	float lambda1 = (v2._u - v0._u) == 0.0 ? 1.0e12 : (v2._v - v0._v) / (v2._u - v0._u);
	float c1 = lambda1 > 1.0e10 ? 0.0 : v0._v - lambda1*v0._u;
	TextUV e2 = v2 - v1;
	float lambda2 = (v2._u - v1._u) == 0.0 ? 1.0e12 : (v2._v - v1._v) / (v2._u - v1._u);
	float c2 = lambda2 > 1.0e10 ? 0.0 : v1._v - lambda2*v1._u;

	int min_u = int(round(min(v0._u, min(v1._u, v2._u)))); // get min of vertical coordinates (= y coordinates)
	int max_u = int(round(max(v0._u, max(v1._u, v2._u)))); // get max of vertical coordinates (= y coordinates)

														   // draw triangle line by line
	for (int i = min_u; i < max_u + 1; i++) {
		// Compute intersection between line {y=i (i.e. u= i)} and all 3 edges
		TextUV inter0 = TextUV(i, lambda0*i + c0);
		if (lambda0 > 1.0e10) {
			inter0._u = -1.0;
			inter0._v = -1.0;
		}
		TextUV inter1 = TextUV(i, lambda1*i + c1);
		if (lambda1 > 1.0e10) {
			inter1._u = -1.0;
			inter1._v = -1.0;
		}
		TextUV inter2 = TextUV(i, lambda2*i + c2);
		if (lambda2 > 1.0e10) {
			inter2._u = -1.0;
			inter2._v = -1.0;
		}

		// identify valid intersections
		bool valid0 = true;
		bool valid1 = true;
		bool valid2 = true;
		if (((inter0 - v0)._u*e0._u + (inter0 - v0)._v*e0._v) < 0.0 || ((inter0 - v1)._u*e0._u + (inter0 - v1)._v*e0._v) > 0.0 || inter0._u == -1) {
			valid0 = false;
		}
		if (((inter1 - v0)._u*e1._u + (inter1 - v0)._v*e1._v) < 0.0 || ((inter1 - v2)._u*e1._u + (inter1 - v2)._v*e1._v) > 0.0 || inter1._u == -1) {
			valid1 = false;
		}
		if (((inter2 - v1)._u*e2._u + (inter2 - v1)._v*e2._v) < 0.0 || ((inter2 - v2)._u*e2._u + (inter2 - v2)._v*e2._v) > 0.0 || inter2._u == -1) {
			valid2 = false;
		}

		int min_v = BumpWidth;
		int max_v = 0;

		if (valid0) {
			min_v = min(min_v, int(round(inter0._v)));
			max_v = max(max_v, int(round(inter0._v)));
		}

		if (valid1) {
			min_v = min(min_v, int(round(inter1._v)));
			max_v = max(max_v, int(round(inter1._v)));
		}

		if (valid2) {
			min_v = min(min_v, int(round(inter2._v)));
			max_v = max(max_v, int(round(inter2._v)));
		}

		for (int j = min_v; j < max_v + 1; j++) {
			data[4 * (i*BumpWidth + j) + 2] = color;
		}

	}

	return;
}

bool Utility::IsInTriangle(MyMesh *TheMesh, Face *triangle, int i, int j) 
{
	TextUV v0 = TextUV(TheMesh->_uvs[triangle->_t1]->_u*float(BumpHeight), TheMesh->_uvs[triangle->_t1]->_v*float(BumpWidth));
	TextUV v1 = TextUV(TheMesh->_uvs[triangle->_t2]->_u*float(BumpHeight), TheMesh->_uvs[triangle->_t2]->_v*float(BumpWidth));
	TextUV v2 = TextUV(TheMesh->_uvs[triangle->_t3]->_u*float(BumpHeight), TheMesh->_uvs[triangle->_t3]->_v*float(BumpWidth));
	TextUV v = TextUV(float(i), float(j));

	TextUV e0 = v0 - v;
	TextUV e1 = v1 - v;
	TextUV e2 = v2 - v;


	float p0 = e0._u*e1._v - e0._v*e1._u;
	float p1 = e1._u*e2._v - e1._v*e2._u;
	float p2 = e2._u*e0._v - e2._v*e0._u;

	return (p0 >= 0.0f && p1 >= 0.0f && p2 >= 0.0f) || (p0 <= 0.0f && p1 <= 0.0f && p2 <= 0.0f);
}
