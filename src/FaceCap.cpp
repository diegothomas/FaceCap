#include "stdafx.h"
#include "FaceCap.h"
#include "Utility.h"

FaceCap::FaceCap() :
	_draw(false),
	_idx(10),
	_idx_curr(0),
	_lvl(3),
	_landmarkOK(false)
{
	cout << "============== FaceCap Constructor ================" << endl;
	// set up face tracker
	_sdm = unique_ptr<SDM>(new SDM("..\\data\\models\\tracker_model49_131.bin"));
	_hpe = unique_ptr<HPE>(new HPE());
	_restart = true;
	_restartRe = true;
	_minScore = 0.1f;
	_minScoreRe = 0.1f;

	_max_iter[0] = 1;
	_max_iter[1] = 3;
	_max_iter[2] = 3;

	_max_iterPR[0] = 1;
	_max_iterPR[1] = 3;
	_max_iterPR[2] = 3;

	_TransfoD2RGB(0, 0) = 0.999984; _TransfoD2RGB(0, 1) = 0.005468185; _TransfoD2RGB(0, 2) = 0.00108126; _TransfoD2RGB(0, 3) = 0.0240602;
	_TransfoD2RGB(1, 0) = -0.00545742; _TransfoD2RGB(1, 1) = 0.999977; _TransfoD2RGB(1, 2) = -0.00403509; _TransfoD2RGB(1, 3) = 0.000744274;
	_TransfoD2RGB(2, 0) = -0.00110927; _TransfoD2RGB(2, 1) = 0.00402909; _TransfoD2RGB(2, 2) = 0.999991; _TransfoD2RGB(2, 3) = 0.00025272;
	_TransfoD2RGB(3, 0) = 0.0; _TransfoD2RGB(3, 1) = 0.0; _TransfoD2RGB(3, 2) = 0.0; _TransfoD2RGB(3, 3) = 1.0;

	//For KinectV1
	_IntrinsicRGB[0] = 516.8;
	_IntrinsicRGB[1] = 519.6;
	_IntrinsicRGB[2] = 311.2;
	_IntrinsicRGB[3] = 231.1;
	_IntrinsicRGB[9] = 1.0;
	_IntrinsicRGB[10] = 8000.0;

	//int size_tables = ((NB_BS - 1)*(NB_BS - 2)) / 2 + 2*(NB_BS - 1) + 1; // +1 to compute residual error
	int size_tables = ((NB_BS)*(NB_BS + 1)) / 2;

	TABLE_I = (int *)malloc(size_tables * sizeof(int));
	TABLE_J = (int *)malloc(size_tables * sizeof(int));
	_Qinv = (float *)malloc((NB_BS - 1)*(NB_BS - 1) * sizeof(float));

	int indx = 0;
	for (int i = 0; i < NB_BS; i++) {
		for (int j = i; j < NB_BS; j++) {
			TABLE_I[indx] = i;
			TABLE_J[indx] = j;
			//cout << TABLE_I[indx] << " ";
			indx++;
		}
		//cout << " " << endl;
	}

	cout << "FaceCap: Load Image" << endl;

	cv::Mat imgL = cv::imread(string("..\\data\\img\\LabelsMask.bmp"), CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat img0 = cv::imread(string("..\\data\\img\\Weights-240.png"), CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat img1 = cv::imread(string("..\\data\\img\\Labels-240.png"), CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat img2 = cv::imread(string("..\\data\\img\\FrontFace.png"), CV_LOAD_IMAGE_UNCHANGED);
	cv::Mat img3 = cv::imread(string("..\\data\\img\\Labelsb-240.png"), CV_LOAD_IMAGE_UNCHANGED);
	
	cout << "FaceCap: Finished Loading Image" << endl;

	_pbuff = (unsigned int *)malloc(sizeof(unsigned int)* cDepthHeight*cDepthWidth);
	_klabels = (int *)malloc(cDepthHeight*cDepthWidth * sizeof(int));
	_klabelsIn = NULL; // (int *)malloc(cDepthHeight*cDepthWidth*sizeof(int));

	_vertices = (MyPoint **)malloc(cDepthHeight*cDepthWidth * sizeof(MyPoint *));
	for (int i = 0; i < cDepthHeight; i++) {
		for (int j = 0; j < cDepthWidth; j++) {
			_vertices[i*cDepthWidth + j] = NULL;
		}
	}

	for (int k = 0; k < 28; k++) {
		_Vtx[k] = (float *)malloc(3 * NBVertices * sizeof(float));
	}
	_imgD = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC4);
	_imgC = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);
	_imgS = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);

	_CoordMapping = (int *)malloc(2 * cDepthHeight*cDepthWidth * sizeof(int));
	_CoordMappingD2RGB = (int *)malloc(2 * cDepthHeight*cDepthWidth * sizeof(int));

	for (int i = 0; i < 10; i++) {
		_color_in[i] = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);
		_VMap_in[i] = cv::Mat(cDepthHeight, cDepthWidth, CV_32FC4);
		_NMap_in[i] = cv::Mat(cDepthHeight, cDepthWidth, CV_32FC4);

	}
	_depth_in[0] = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC4);
	_depth_in[1] = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC4);

	_depthIn = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC3);
	_VMap = cv::Mat(cDepthHeight, cDepthWidth, CV_32FC4);
	_NMap = cv::Mat(cDepthHeight, cDepthWidth, CV_32FC4);
	_RGBMap = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC4);
	_segmentedIn = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);

	_VMapBump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
	_NMapBump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
	_RGBMapBump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);

	_WeightMap = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
	_Bump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
	_BumpSwap = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
	_LabelsMask = cv::Mat(BumpHeight, BumpWidth, CV_8UC4);

	_FaceSegment = cv::Mat(BumpHeight, BumpWidth, CV_8UC4);

	_triangles = (FaceGPU *)malloc(8518 * sizeof(FaceGPU));
	_verticesList = (Point3DGPU *)malloc(49 * 4325 * sizeof(Point3DGPU));

	_verticesBump = (Point3DGPU *)malloc(BumpHeight * BumpWidth * sizeof(Point3DGPU));
	_VerticesBS = (float *)malloc(NB_BS * 6 * BumpHeight * BumpWidth * sizeof(float));

	_x_raw = cv::Mat(cDepthHeight, cDepthWidth, CV_32F);
	_y_raw = cv::Mat(cDepthHeight, cDepthWidth, CV_32F);
	_ones = cv::Mat::ones(cDepthHeight, cDepthWidth, CV_32F);
	_grad_x = cv::Mat::zeros(cDepthHeight, cDepthWidth, CV_32FC4);
	_grad_y = cv::Mat::zeros(cDepthHeight, cDepthWidth, CV_32FC4);
	_grad_x_bump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
	_grad_y_bump = cv::Mat(BumpHeight, BumpWidth, CV_32FC4);
	_ones_bump = cv::Mat::ones(BumpHeight, BumpWidth, CV_32F);

	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			_Bump.at<cv::Vec4f>(i, j)[0] = 0.0f;
			_Bump.at<cv::Vec4f>(i, j)[1] = 0.0f; // Mask
			_Bump.at<cv::Vec4f>(i, j)[2] = float(img1.at<cv::Vec3w>(i*FACT_BUMP, j*FACT_BUMP)[0]) - 1.0f;
			_Bump.at<cv::Vec4f>(i, j)[3] = img3.at<cv::Vec4b>(i*FACT_BUMP, j*FACT_BUMP)[2] > 100 ? -1.0f : 0.0f;
			_LabelsMask.at<cv::Vec4b>(i, j)[0] = imgL.at<cv::Vec3b>(i*FACT_BUMP, j*FACT_BUMP)[0];
			_LabelsMask.at<cv::Vec4b>(i, j)[1] = imgL.at<cv::Vec3b>(i*FACT_BUMP, j*FACT_BUMP)[1];
			_LabelsMask.at<cv::Vec4b>(i, j)[2] = imgL.at<cv::Vec3b>(i*FACT_BUMP, j*FACT_BUMP)[2];
			_LabelsMask.at<cv::Vec4b>(i, j)[3] = img2.at<cv::Vec4b>(i*FACT_BUMP, j*FACT_BUMP)[2] > 100 ? 1 : 0;
			_WeightMap.at<cv::Vec4f>(i, j)[0] = float(img0.at<cv::Vec3w>(i*FACT_BUMP, j*FACT_BUMP)[0]) / 65535.0f;
			_WeightMap.at<cv::Vec4f>(i, j)[1] = float(img0.at<cv::Vec3w>(i*FACT_BUMP, j*FACT_BUMP)[1]) / 65535.0f;
			_WeightMap.at<cv::Vec4f>(i, j)[2] = float(img0.at<cv::Vec3w>(i*FACT_BUMP, j*FACT_BUMP)[2]) / 65535.0f;
			_FaceSegment.at<cv::Vec4b>(i, j)[0] = 255;
		}
	}

	for (int i = 0; i < 16; i++)
		_Pose[i] = 0.0;
	_Pose[0] = 1.0; _Pose[5] = 1.0; _Pose[10] = 1.0; _Pose[15] = 1.0;

	_outbuff = (double *)malloc(50 * sizeof(double));
	_outbuffJTJ = (double *)malloc(1176 * sizeof(double));
	_outbuffGICP = (double *)malloc(29 * sizeof(double));
	_outbuffReduce = (float *)malloc(size_tables * sizeof(float));
	_outbuffResolved = (float *)malloc((NB_BS - 1) * sizeof(float));

	_BCPU = Eigen::MatrixXd::Zero(NB_BS, BumpWidth*BumpHeight);

	_landmarksBump = cv::Mat(2, 43, CV_32SC1);
	_landmarks = cv::Mat(2, 43, CV_32FC1);
	_landmarks_prev = cv::Mat(2, 43, CV_32FC1);

	for (int k = 0; k < 10; k++) {
		_landmarks_in[k] = cv::Mat(2, 43, CV_32FC1);
		for (int i = 0; i < 43; i++) {
			_landmarks_in[k].at<float>(0, i) = 0.0f;
			_landmarks_in[k].at<float>(1, i) = 0.0f;
		}
	}

	_pRect.x = 0;
	_pRect.y = 0;
	_pRect.width = cDepthWidth;
	_pRect.height = cDepthHeight;

	for (int i = 0; i < 49; i++)
		_BlendshapeCoeff[i] = 0.0;
	_BlendshapeCoeff[0] = 1.0;

	_Rotation = Eigen::Matrix3f::Identity();
	_Translation = Eigen::Vector3f::Zero();
	_Rotation_inv = Eigen::Matrix3f::Identity();
	_Translation_inv = Eigen::Vector3f::Zero();	
}

FaceCap::~FaceCap()
{
	cout << "Finishing" << endl;

	cout << "Free CPU" << endl;
	for (int i = 0; i < cDepthHeight; i++) {
		for (int j = 0; j < cDepthWidth; j++) {
			if (_vertices[i*cDepthWidth + j] != NULL)
				std::free(_vertices[i*cDepthWidth + j]);
		}
	}
	std::free(_vertices);

	std::free(_verticesBump);
	std::free(_VerticesBS);
	std::free(_triangles);
	std::free(_verticesList);

	std::free(_outbuff);
	std::free(_outbuffJTJ);
	std::free(_outbuffGICP);
	std::free(_outbuffReduce);
	std::free(_outbuffResolved);

	std::free(TABLE_I);
	std::free(TABLE_J);
	std::free(_Qinv);
	std::free(_pbuff);

	std::free(_klabels);
	//std::free(_klabelsIn);

	cout << "Free sensor" << endl;
	/*if (_sensorManager != NULL) {
	delete _sensorManager;
	}
	*/
	cout << "Done" << endl;

	/*while (!_depth.empty()) {
	_CoordMaps.pop();
	_CoordMapsD2RGB.pop();
	_color.pop();
	_segmented_color.pop();
	_ptsQ.pop();
	_depth.pop();
	_ptsRect.pop();
	_klabelsSet.pop();
	}*/
}

bool FaceCap::StartSensor(){
	_sensorManager = new SensorManager();

	/* Check for sensor */
	if (!_sensorManager->init()){
		return false;	// couldn't find depth sensor
	}
	return true;
}

void FaceCap::Draw(bool color, bool bump) {

	//glBegin(GL_POINTS);
	glColor4f(1.0, 1.0, 1.0, 1.0);

	float pt[3];
	float nmle[3];
	Point3DGPU *currV;
	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			if (_Bump.at<cv::Vec4f>(i, j)[1] > 0.0) {
				currV = &_verticesBump[i*BumpWidth + j];

				// Transform points to match tracking
				if (bump) {
					pt[0] = _VMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
					pt[1] = _VMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
					pt[2] = _VMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(2, 2) + _Translation_inv(2);
					
					//pt[0] = _VMapBump.at<cv::Vec4f>(i, j)[0];
					//pt[1] = _VMapBump.at<cv::Vec4f>(i, j)[1];
					//pt[2] = _VMapBump.at<cv::Vec4f>(i, j)[2];

					nmle[0] = _NMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(0, 0) + _NMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(0, 1) + _NMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(0, 2);
					nmle[1] = _NMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(1, 0) + _NMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(1, 1) + _NMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(1, 2);
					nmle[2] = _NMapBump.at<cv::Vec4f>(i, j)[0] * _Rotation_inv(2, 0) + _NMapBump.at<cv::Vec4f>(i, j)[1] * _Rotation_inv(2, 1) + _NMapBump.at<cv::Vec4f>(i, j)[2] * _Rotation_inv(2, 2);
				}
				else {
					pt[0] = currV->_x * _Rotation_inv(0, 0) + currV->_y * _Rotation_inv(0, 1) + currV->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
					pt[1] = currV->_x * _Rotation_inv(1, 0) + currV->_y * _Rotation_inv(1, 1) + currV->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
					pt[2] = currV->_x * _Rotation_inv(2, 0) + currV->_y * _Rotation_inv(2, 1) + currV->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

					nmle[0] = currV->_Nx * _Rotation_inv(0, 0) + currV->_Ny * _Rotation_inv(0, 1) + currV->_Nz * _Rotation_inv(0, 2);
					nmle[1] = currV->_Nx * _Rotation_inv(1, 0) + currV->_Ny * _Rotation_inv(1, 1) + currV->_Nz * _Rotation_inv(1, 2);
					nmle[2] = currV->_Nx * _Rotation_inv(2, 0) + currV->_Ny * _Rotation_inv(2, 1) + currV->_Nz * _Rotation_inv(2, 2);
				}

				if (_RGBMapBump.at<cv::Vec4f>(i, j)[0] == 255.0f && _RGBMapBump.at<cv::Vec4f>(i, j)[1] == 0.0f && _RGBMapBump.at<cv::Vec4f>(i, j)[2] == 0.0f)
					glPointSize(3.0);

				glBegin(GL_POINTS);
				if (color) {
					glColor4f(_RGBMapBump.at<cv::Vec4f>(i, j)[0] / 255.0, _RGBMapBump.at<cv::Vec4f>(i, j)[1] / 255.0, _RGBMapBump.at<cv::Vec4f>(i, j)[2] / 255.0, 1.0);
				}
				glNormal3f(nmle[0], nmle[1], nmle[2]);
				glVertex3f(pt[0], pt[1], pt[2]);
				glEnd();
				glPointSize(1.0);
				/*glNormal3f(currV->_TNx, currV->_TNy, currV->_TNz);
				glVertex3f(currV->_Tx, currV->_Ty, currV->_Tz);*/
			}
		}
	}

	//glEnd();
}

void FaceCap::DrawBlendedMesh(vector<MyMesh *> Blendshape) {
	MyMesh *RefMesh = Blendshape[0];


	vector<float *> Pts;
	vector<float *> Normals;
	vector<float> Count;
	for (vector<Point3D<float> *>::iterator it = RefMesh->_vertices.begin(); it != RefMesh->_vertices.end(); it++) {
		float *p = (float *)malloc(3 * sizeof(float));
		p[0] = 0.0f; p[1] = 0.0f; p[2] = 0.0f;
		float *n = (float *)malloc(3 * sizeof(float));
		n[0] = 0.0f; n[1] = 0.0f; n[2] = 0.0f;
		Pts.push_back(p);
		Normals.push_back(n);
		Count.push_back(0.0f);
	}

	float Pt[3];
	float tmpPt[3];
	float tmpNmle[3];
	MyPoint *s1, *s2, *s3;
	float d1, d2, d3;
	float m1, m2, m3;
	//cout << "size: " << RefMesh->_vertices.size() << endl;
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		//cout << "v1: " << (*it)->_v1 << endl;
		s1 = RefMesh->_vertices[(*it)->_v1];
		s2 = RefMesh->_vertices[(*it)->_v2];
		s3 = RefMesh->_vertices[(*it)->_v3];

		///cout << "u,v: " << Myround(s1->_u*float(BumpHeight)) << ", " << Myround(s1->_v*float(BumpWidth)) << endl;
		d1 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t1]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t1]->_v*float(BumpWidth)))[0] / 1000.0f;
		m1 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t1]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t1]->_v*float(BumpWidth)))[1];
		d2 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t2]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t2]->_v*float(BumpWidth)))[0] / 1000.0f;
		m2 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t2]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t2]->_v*float(BumpWidth)))[1];
		d3 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t3]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t3]->_v*float(BumpWidth)))[0] / 1000.0f;
		m3 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t3]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t3]->_v*float(BumpWidth)))[1];
		if (m1 == 0.0f || m2 == 0.0f || m3 == 0.0f)
			continue;

		//S1
		//cout << "d1: " << d1 << endl;
		tmpPt[0] = s1->_x + d1 * s1->_Nx;
		tmpPt[1] = s1->_y + d1 * s1->_Ny;
		tmpPt[2] = s1->_z + d1 * s1->_Nz;

		Pt[0] = tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2);
		Pt[1] = tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2);
		Pt[2] = tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2);

		for (int i = 1; i < 28; i++) {

			tmpPt[0] = (Blendshape[i]->_vertices[(*it)->_v1]->_x - s1->_x) + d1 * (Blendshape[i]->_vertices[(*it)->_v1]->_Nx - s1->_Nx);
			tmpPt[1] = (Blendshape[i]->_vertices[(*it)->_v1]->_y - s1->_y) + d1 * (Blendshape[i]->_vertices[(*it)->_v1]->_Ny - s1->_Ny);
			tmpPt[2] = (Blendshape[i]->_vertices[(*it)->_v1]->_z - s1->_z) + d1 * (Blendshape[i]->_vertices[(*it)->_v1]->_Nz - s1->_Nz);

			Pt[0] = Pt[0] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2));
			Pt[1] = Pt[1] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2));
			Pt[2] = Pt[2] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2));

		}
		Pts[(*it)->_v1][0] = Pt[0] + _Translation_inv(0);
		Pts[(*it)->_v1][1] = Pt[1] + _Translation_inv(1);
		Pts[(*it)->_v1][2] = Pt[2] + _Translation_inv(2);
		//cout << "Pts1: " << Pts[(*it)->_v1][0] << ", " << Pts[(*it)->_v1][1] << ", " << Pts[(*it)->_v1][2] << endl;

		//S2
		tmpPt[0] = s2->_x + d2 * s2->_Nx;
		tmpPt[1] = s2->_y + d2 * s2->_Ny;
		tmpPt[2] = s2->_z + d2 * s2->_Nz;

		Pt[0] = tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2);
		Pt[1] = tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2);
		Pt[2] = tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2);

		for (int i = 1; i < 28; i++) {

			tmpPt[0] = (Blendshape[i]->_vertices[(*it)->_v2]->_x - s2->_x) + d2 * (Blendshape[i]->_vertices[(*it)->_v2]->_Nx - s2->_Nx);
			tmpPt[1] = (Blendshape[i]->_vertices[(*it)->_v2]->_y - s2->_y) + d2 * (Blendshape[i]->_vertices[(*it)->_v2]->_Ny - s2->_Ny);
			tmpPt[2] = (Blendshape[i]->_vertices[(*it)->_v2]->_z - s2->_z) + d2 * (Blendshape[i]->_vertices[(*it)->_v2]->_Nz - s2->_Nz);

			Pt[0] = Pt[0] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2));
			Pt[1] = Pt[1] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2));
			Pt[2] = Pt[2] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2));

		}
		Pts[(*it)->_v2][0] = Pt[0] + _Translation_inv(0);
		Pts[(*it)->_v2][1] = Pt[1] + _Translation_inv(1);
		Pts[(*it)->_v2][2] = Pt[2] + _Translation_inv(2);

		//S3
		tmpPt[0] = s3->_x + d3 * s3->_Nx;
		tmpPt[1] = s3->_y + d3 * s3->_Ny;
		tmpPt[2] = s3->_z + d3 * s3->_Nz;

		Pt[0] = tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2);
		Pt[1] = tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2);
		Pt[2] = tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2);

		for (int i = 1; i < 28; i++) {

			tmpPt[0] = (Blendshape[i]->_vertices[(*it)->_v3]->_x - s3->_x) + d3 * (Blendshape[i]->_vertices[(*it)->_v3]->_Nx - s3->_Nx);
			tmpPt[1] = (Blendshape[i]->_vertices[(*it)->_v3]->_y - s3->_y) + d3 * (Blendshape[i]->_vertices[(*it)->_v3]->_Ny - s3->_Ny);
			tmpPt[2] = (Blendshape[i]->_vertices[(*it)->_v3]->_z - s3->_z) + d3 * (Blendshape[i]->_vertices[(*it)->_v3]->_Nz - s3->_Nz);

			Pt[0] = Pt[0] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2));
			Pt[1] = Pt[1] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2));
			Pt[2] = Pt[2] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2));

		}
		Pts[(*it)->_v3][0] = Pt[0] + _Translation_inv(0);
		Pts[(*it)->_v3][1] = Pt[1] + _Translation_inv(1);
		Pts[(*it)->_v3][2] = Pt[2] + _Translation_inv(2);

		tmpNmle[0] = (Pts[(*it)->_v2][1] - Pts[(*it)->_v1][1])*(Pts[(*it)->_v3][2] - Pts[(*it)->_v1][2]) - (Pts[(*it)->_v2][2] - Pts[(*it)->_v1][2])*(Pts[(*it)->_v3][1] - Pts[(*it)->_v1][1]);
		tmpNmle[1] = (Pts[(*it)->_v2][2] - Pts[(*it)->_v1][2])*(Pts[(*it)->_v3][0] - Pts[(*it)->_v1][0]) - (Pts[(*it)->_v2][0] - Pts[(*it)->_v1][0])*(Pts[(*it)->_v3][2] - Pts[(*it)->_v1][2]);
		tmpNmle[2] = (Pts[(*it)->_v2][0] - Pts[(*it)->_v1][0])*(Pts[(*it)->_v3][1] - Pts[(*it)->_v1][1]) - (Pts[(*it)->_v2][1] - Pts[(*it)->_v1][1])*(Pts[(*it)->_v3][0] - Pts[(*it)->_v1][0]);

		float scal = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);

		Normals[(*it)->_v1][0] = Normals[(*it)->_v1][0] + tmpNmle[0] / scal;
		Normals[(*it)->_v1][1] = Normals[(*it)->_v1][1] + tmpNmle[1] / scal;
		Normals[(*it)->_v1][2] = Normals[(*it)->_v1][2] + tmpNmle[2] / scal;

		Normals[(*it)->_v2][0] = Normals[(*it)->_v2][0] + tmpNmle[0] / scal;
		Normals[(*it)->_v2][1] = Normals[(*it)->_v2][1] + tmpNmle[1] / scal;
		Normals[(*it)->_v2][2] = Normals[(*it)->_v2][2] + tmpNmle[2] / scal;

		Normals[(*it)->_v3][0] = Normals[(*it)->_v3][0] + tmpNmle[0] / scal;
		Normals[(*it)->_v3][1] = Normals[(*it)->_v3][1] + tmpNmle[1] / scal;
		Normals[(*it)->_v3][2] = Normals[(*it)->_v3][2] + tmpNmle[2] / scal;

		Count[(*it)->_v1] = Count[(*it)->_v1] + 1.0f;
		Count[(*it)->_v2] = Count[(*it)->_v2] + 1.0f;
		Count[(*it)->_v3] = Count[(*it)->_v3] + 1.0f;

	}


	glColor4f(1.0, 1.0, 1.0, 1.0);
	glBegin(GL_TRIANGLES);

	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		s1 = RefMesh->_vertices[(*it)->_v1];
		s2 = RefMesh->_vertices[(*it)->_v2];
		s3 = RefMesh->_vertices[(*it)->_v3];

		m1 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t1]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t1]->_v*float(BumpWidth)))[1];
		m2 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t2]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t2]->_v*float(BumpWidth)))[1];
		m3 = _Bump.at<cv::Vec4f>(Myround(RefMesh->_uvs[(*it)->_t3]->_u*float(BumpHeight)), Myround(RefMesh->_uvs[(*it)->_t3]->_v*float(BumpWidth)))[1];
		if (m1 == 0.0f || m2 == 0.0f || m3 == 0.0f)
			continue;

		tmpNmle[0] = Normals[(*it)->_v1][0] / Count[(*it)->_v1];
		tmpNmle[1] = Normals[(*it)->_v1][1] / Count[(*it)->_v1];
		tmpNmle[2] = Normals[(*it)->_v1][2] / Count[(*it)->_v1];
		float scal = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);

		glNormal3f(tmpNmle[0]/scal, tmpNmle[1]/scal, tmpNmle[2]/scal);
		glVertex3f(Pts[(*it)->_v1][0], Pts[(*it)->_v1][1], Pts[(*it)->_v1][2]);
		//cout << Pts[(*it)->_v1][0] << ", " << Pts[(*it)->_v1][1] << ", " << Pts[(*it)->_v1][2] << endl;


		tmpNmle[0] = Normals[(*it)->_v2][0] / Count[(*it)->_v2];
		tmpNmle[1] = Normals[(*it)->_v2][1] / Count[(*it)->_v2];
		tmpNmle[2] = Normals[(*it)->_v2][2] / Count[(*it)->_v2];
		scal = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);

		glNormal3f(tmpNmle[0] / scal, tmpNmle[1] / scal, tmpNmle[2] / scal);
		glVertex3f(Pts[(*it)->_v2][0], Pts[(*it)->_v2][1], Pts[(*it)->_v2][2]);


		tmpNmle[0] = Normals[(*it)->_v3][0] / Count[(*it)->_v3];
		tmpNmle[1] = Normals[(*it)->_v3][1] / Count[(*it)->_v3];
		tmpNmle[2] = Normals[(*it)->_v3][2] / Count[(*it)->_v3];
		scal = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);

		glNormal3f(tmpNmle[0] / scal, tmpNmle[1] / scal, tmpNmle[2] / scal);
		glVertex3f(Pts[(*it)->_v3][0], Pts[(*it)->_v3][1], Pts[(*it)->_v3][2]);

	}

	glEnd();

	for (int i = 0; i < Pts.size(); i++) {
		free(Pts[i]);
		free(Normals[i]);
	}
	Pts.clear();
	Normals.clear();
	Count.clear();

	/*glColor4f(1.0, 1.0, 1.0, 1.0);
	glBegin(GL_TRIANGLES);

	float Pt[3];
	float tmpPt[3];
	float tmpNmle[3];
	MyPoint *s1, *s2, *s3;
	float d1, d2, d3;
	float m1, m2, m3;
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		s1 = RefMesh->_vertices[(*it)->_v1];
		s2 = RefMesh->_vertices[(*it)->_v2];
		s3 = RefMesh->_vertices[(*it)->_v3];

		d1 = _Bump.at<cv::Vec4f>(s1->_u, s1->_v)[0];
		m1 = _Bump.at<cv::Vec4f>(s1->_u, s1->_v)[1];
		d2 = _Bump.at<cv::Vec4f>(s2->_u, s2->_v)[0];
		m2 = _Bump.at<cv::Vec4f>(s2->_u, s2->_v)[1];
		d3 = _Bump.at<cv::Vec4f>(s3->_u, s3->_v)[0];
		m3 = _Bump.at<cv::Vec4f>(s3->_u, s3->_v)[1];
		if (m1 == 0 || m2 == 0 || m3 == 0)
			continue;

		/*tmpPt[0] = s1->_x;
		tmpPt[1] = s1->_y;
		tmpPt[2] = s1->_z;
		tmpNmle[0] = s1->_Nx;
		tmpNmle[1] = s1->_Ny;
		tmpNmle[2] = s1->_Nz;*/

		/*tmpPt[0] = s1->_x + d1 * s1->_Nx;
		tmpPt[1] = s1->_y + d1 * s1->_Ny;
		tmpPt[2] = s1->_z + d1 * s1->_Nz;

		Pt[0] = tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2);
		Pt[1] = tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2);
		Pt[2] = tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2);

		for (int i = 1; i < 28; i++) {
			/*tmpPt[0] = tmpPt[0] - _BlendshapeCoeff[i] * (s1->_x - Blendshape[i]->_vertices[(*it)->_v1]->_x);
			tmpPt[1] = tmpPt[1] - _BlendshapeCoeff[i] * (s1->_y - Blendshape[i]->_vertices[(*it)->_v1]->_y);
			tmpPt[2] = tmpPt[2] - _BlendshapeCoeff[i] * (s1->_z - Blendshape[i]->_vertices[(*it)->_v1]->_z);

			tmpNmle[0] = tmpNmle[0] - _BlendshapeCoeff[i] * (s1->_Nx - Blendshape[i]->_vertices[(*it)->_v1]->_Nx);
			tmpNmle[1] = tmpNmle[1] - _BlendshapeCoeff[i] * (s1->_Ny - Blendshape[i]->_vertices[(*it)->_v1]->_Ny);
			tmpNmle[2] = tmpNmle[2] - _BlendshapeCoeff[i] * (s1->_Nz - Blendshape[i]->_vertices[(*it)->_v1]->_Nz);*/

		/*	tmpPt[0] = (Blendshape[i]->_vertices[(*it)->_v1]->_x - s1->_x) + d1 * (Blendshape[i]->_vertices[(*it)->_v1]->_Nx - s1->_Nx);
			tmpPt[1] = (Blendshape[i]->_vertices[(*it)->_v1]->_y - s1->_y) + d1 * (Blendshape[i]->_vertices[(*it)->_v1]->_Ny - s1->_Ny);
			tmpPt[2] = (Blendshape[i]->_vertices[(*it)->_v1]->_z - s1->_z) + d1 * (Blendshape[i]->_vertices[(*it)->_v1]->_Nz - s1->_Nz);

			Pt[0] = Pt[0] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(0, 0) + tmpPt[1] * _Rotation_inv(0, 1) + tmpPt[2] * _Rotation_inv(0, 2));
			Pt[1] = Pt[1] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(1, 0) + tmpPt[1] * _Rotation_inv(1, 1) + tmpPt[2] * _Rotation_inv(1, 2));
			Pt[2] = Pt[2] + _BlendshapeCoeff[i] * (tmpPt[0] * _Rotation_inv(2, 0) + tmpPt[1] * _Rotation_inv(2, 1) + tmpPt[2] * _Rotation_inv(2, 2));

		}
		Pt[0] = Pt[0] + _Translation_inv(0);
		Pt[1] = Pt[1] + _Translation_inv(1);
		Pt[2] = Pt[2] + _Translation_inv(2);

		float nrm = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);
		tmpNmle[0] = tmpNmle[0] / nrm;
		tmpNmle[1] = tmpNmle[1] / nrm;
		tmpNmle[2] = tmpNmle[2] / nrm;

		glNormal3f(tmpNmle[0], tmpNmle[1], tmpNmle[2]);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);

		tmpPt[0] = s2->_x;
		tmpPt[1] = s2->_y;
		tmpPt[2] = s2->_z;
		tmpNmle[0] = s2->_Nx;
		tmpNmle[1] = s2->_Ny;
		tmpNmle[2] = s2->_Nz;

		for (int i = 1; i < 28; i++) {
			tmpPt[0] = tmpPt[0] - _BlendshapeCoeff[i] * (s2->_x - Blendshape[i]->_vertices[(*it)->_v2]->_x);
			tmpPt[1] = tmpPt[1] - _BlendshapeCoeff[i] * (s2->_y - Blendshape[i]->_vertices[(*it)->_v2]->_y);
			tmpPt[2] = tmpPt[2] - _BlendshapeCoeff[i] * (s2->_z - Blendshape[i]->_vertices[(*it)->_v2]->_z);

			tmpNmle[0] = tmpNmle[0] - _BlendshapeCoeff[i] * (s2->_Nx - Blendshape[i]->_vertices[(*it)->_v2]->_Nx);
			tmpNmle[1] = tmpNmle[1] - _BlendshapeCoeff[i] * (s2->_Ny - Blendshape[i]->_vertices[(*it)->_v2]->_Ny);
			tmpNmle[2] = tmpNmle[2] - _BlendshapeCoeff[i] * (s2->_Nz - Blendshape[i]->_vertices[(*it)->_v2]->_Nz);
		}
		nrm = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);
		tmpNmle[0] = tmpNmle[0] / nrm;
		tmpNmle[1] = tmpNmle[1] / nrm;
		tmpNmle[2] = tmpNmle[2] / nrm;

		glNormal3f(tmpNmle[0], tmpNmle[1], tmpNmle[2]);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);


		tmpPt[0] = s3->_x;
		tmpPt[1] = s3->_y;
		tmpPt[2] = s3->_z;
		tmpNmle[0] = s3->_Nx;
		tmpNmle[1] = s3->_Ny;
		tmpNmle[2] = s3->_Nz;

		for (int i = 1; i < 28; i++) {
			tmpPt[0] = tmpPt[0] - _BlendshapeCoeff[i] * (s3->_x - Blendshape[i]->_vertices[(*it)->_v3]->_x);
			tmpPt[1] = tmpPt[1] - _BlendshapeCoeff[i] * (s3->_y - Blendshape[i]->_vertices[(*it)->_v3]->_y);
			tmpPt[2] = tmpPt[2] - _BlendshapeCoeff[i] * (s3->_z - Blendshape[i]->_vertices[(*it)->_v3]->_z);

			tmpNmle[0] = tmpNmle[0] - _BlendshapeCoeff[i] * (s3->_Nx - Blendshape[i]->_vertices[(*it)->_v3]->_Nx);
			tmpNmle[1] = tmpNmle[1] - _BlendshapeCoeff[i] * (s3->_Ny - Blendshape[i]->_vertices[(*it)->_v3]->_Ny);
			tmpNmle[2] = tmpNmle[2] - _BlendshapeCoeff[i] * (s3->_Nz - Blendshape[i]->_vertices[(*it)->_v3]->_Nz);
		}
		nrm = sqrt(tmpNmle[0] * tmpNmle[0] + tmpNmle[1] * tmpNmle[1] + tmpNmle[2] * tmpNmle[2]);
		tmpNmle[0] = tmpNmle[0] / nrm;
		tmpNmle[1] = tmpNmle[1] / nrm;
		tmpNmle[2] = tmpNmle[2] / nrm;

		glNormal3f(tmpNmle[0], tmpNmle[1], tmpNmle[2]);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);
	}

	glEnd();*/
}

void FaceCap::DrawQuad(int color, bool bump) {

	glBegin(GL_QUADS);
	glColor4f(1.0, 1.0, 1.0, 1.0);

	int indx_i[4] = { 0, 1, 1, 0 };
	int indx_j[4] = { 0, 0, 1, 1 };

	float pt[3];
	float nmle[3];
	Point3DGPU *currV;
	for (int i = 0; i < BumpHeight-1; i++) {
		for (int j = 0; j < BumpWidth-1; j++) {
			//if (_Bump.at<cv::Vec4f>(i, j)[1] > 0.0f && _Bump.at<cv::Vec4f>(i + 1, j)[1] > 0.0f && _Bump.at<cv::Vec4f>(i + 1, j + 1)[1] > 0.0f && _Bump.at<cv::Vec4f>(i, j + 1)[1] > 0.0f) {
			if (_VMapBump.at<cv::Vec4f>(i, j)[0] != 0.0f && _VMapBump.at<cv::Vec4f>(i + 1, j)[0] != 0.0f &&_VMapBump.at<cv::Vec4f>(i + 1, j + 1)[0] != 0.0f && _VMapBump.at<cv::Vec4f>(i, j + 1)[0] != 0.0f) {
				for (int k = 0; k < 4; k++) {
					currV = &_verticesBump[(i + indx_i[k])*BumpWidth + (j + indx_j[k])];

					// Transform points to match tracking
					if (bump) {
						pt[0] = _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
						pt[1] = _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
						pt[2] = _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

						//pt[0] = _VMapBump.at<cv::Vec4f>(i, j)[0];
						//pt[1] = _VMapBump.at<cv::Vec4f>(i, j)[1];
						//pt[2] = _VMapBump.at<cv::Vec4f>(i, j)[2];

						nmle[0] = _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(0, 0) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(0, 1) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(0, 2);
						nmle[1] = _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(1, 0) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(1, 1) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(1, 2);
						nmle[2] = _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] * _Rotation_inv(2, 0) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] * _Rotation_inv(2, 1) + _NMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] * _Rotation_inv(2, 2);
					}
					else {
						pt[0] = currV->_x * _Rotation_inv(0, 0) + currV->_y * _Rotation_inv(0, 1) + currV->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
						pt[1] = currV->_x * _Rotation_inv(1, 0) + currV->_y * _Rotation_inv(1, 1) + currV->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
						pt[2] = currV->_x * _Rotation_inv(2, 0) + currV->_y * _Rotation_inv(2, 1) + currV->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

						nmle[0] = currV->_Nx * _Rotation_inv(0, 0) + currV->_Ny * _Rotation_inv(0, 1) + currV->_Nz * _Rotation_inv(0, 2);
						nmle[1] = currV->_Nx * _Rotation_inv(1, 0) + currV->_Ny * _Rotation_inv(1, 1) + currV->_Nz * _Rotation_inv(1, 2);
						nmle[2] = currV->_Nx * _Rotation_inv(2, 0) + currV->_Ny * _Rotation_inv(2, 1) + currV->_Nz * _Rotation_inv(2, 2);
					}

					if (color == 0) {
						glColor4f((nmle[0] + 1.0) / 2.0, (nmle[1] + 1.0) / 2.0, (nmle[2] + 1.0) / 2.0, 1.0);
					}
					if (color == 1) {
						glColor4f(_RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] / 255.0, _RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] / 255.0, _RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] / 255.0, 1.0);
					}
					if (color == 2) {
						glColor4f(_RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] / 255.0, _RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] / 255.0, _RGBMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] / 255.0, pt[2] / 10.0f); 
					}
					glNormal3f(nmle[0], nmle[1], nmle[2]);
					glVertex3f(pt[0], pt[1], pt[2]);
					//if (pt[2] > -0.5f) {
					//	cout << "pt: " << pt[0] << ", " << pt[1] << ", " << pt[2] << "; bump: " << _Bump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] << ", " << _Bump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] << endl;
					//	cout << "VMap: " << _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[0] << ", " << _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[1] << ", " << _VMapBump.at<cv::Vec4f>(i + indx_i[k], j + indx_j[k])[2] << endl;
					//}
				}
			}
		}
	}

	glEnd();
}

void FaceCap::DrawVBO(bool color) {

	/* on passe en mode VBO */
	glBindBuffer(GL_ARRAY_BUFFER, _VBO);

	glVertexPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
	glNormalPointer(GL_FLOAT, 0, BUFFER_OFFSET(BumpHeight*BumpWidth * 3 * sizeof(float)));
	if (color)
		glColorPointer(4, GL_FLOAT, 0, BUFFER_OFFSET(BumpHeight*BumpWidth * 3 * sizeof(float) + BumpHeight*BumpWidth * 3 * sizeof(float)));

	/* activation des tableaux de donnees */
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	if (color)
		glEnableClientState(GL_COLOR_ARRAY);

	/* rendu indices */
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _Index);
	glDrawElements(GL_QUADS, 4 * (BumpWidth - 1)*(BumpHeight - 1), GL_UNSIGNED_INT, BUFFER_OFFSET(0));
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	/* rendu points */
	//glDrawArrays(GL_POINTS, 0, _width*_height);


	if (color)
		glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
};

void FaceCap::DrawRect(bool color) {

	//glBegin(GL_POINTS);
	//glColor4f(1.0, 1.0, 1.0, 1.0);

	//MyPoint *currPt = _vertices[(_pRect.y + _pRect.height / 2)*cDepthWidth + (_pRect.x + _pRect.width / 2)];
	//unsigned short ref_depth = currPt == NULL ? 65000 : currPt->_z;

	int i, j;
//	for (i = _pRect.y; i < _pRect.y + _pRect.height; i++) {
//		for (j = _pRect.x; j < _pRect.x + _pRect.width; j++) {

	for (i = 0; i < cDepthHeight; i++) {
		for (j = 0; j < cDepthWidth; j++) {
			/*currPt = _vertices[i*cDepthWidth + j];
			if (currPt == NULL)
			continue;*/

			//if (-currPt->_z > ref_depth + 0.5)
			//	continue;

			//cout << "_VMap: " << _VMap.at<cv::Vec4f>(i, j)[0] << ", " << _VMap.at<cv::Vec4f>(i, j)[1] << ", " << _VMap.at<cv::Vec4f>(i, j)[2] << endl;
			if (_RGBMap.at<cv::Vec4b>(i, j)[2] == 0 && _RGBMap.at<cv::Vec4b>(i, j)[1] == 255 && _RGBMap.at<cv::Vec4b>(i, j)[0] == 0)
				glPointSize(3.0);
			glBegin(GL_POINTS);
			if (color)
				glColor4f(float(_RGBMap.at<cv::Vec4b>(i, j)[2]) / 255.0f, float(_RGBMap.at<cv::Vec4b>(i, j)[1]) / 255.0f, float(_RGBMap.at<cv::Vec4b>(i, j)[0]) / 255.0f, 0.2f);
			else
				glColor4f((_NMap.at<cv::Vec4f>(i, j)[0] + 1.0f) / 2.0f, (_NMap.at<cv::Vec4f>(i, j)[1] + 1.0f) / 2.0f, (_NMap.at<cv::Vec4f>(i, j)[2] + 1.0f) / 2.0f, 0.1f);
			glNormal3f(_NMap.at<cv::Vec4f>(i, j)[0], _NMap.at<cv::Vec4f>(i, j)[1], _NMap.at<cv::Vec4f>(i, j)[2]);
			glVertex3f(_VMap.at<cv::Vec4f>(i, j)[0], _VMap.at<cv::Vec4f>(i, j)[1], _VMap.at<cv::Vec4f>(i, j)[2]);
			glEnd();
			glPointSize(1.0);

			/*glNormal3f(currPt->_Nx, currPt->_Ny, currPt->_Nz);
			glVertex3f(currPt->_x, currPt->_y, currPt->_z);
			glColor3f(currPt->_R, currPt->_G, currPt->_B);*/
		}
	}
	glColor4f(1.0, 1.0, 1.0, 1.0);
	//glEnd();
}

void FaceCap::DrawLandmark(int i, vector<MyMesh *> Blendshape){
	if (_landmarks.at<float>(0, i) == 0.0f && _landmarks.at<float>(1, i) == 0.0f) {
		cout << "landmark null" << endl;
		return;
	}
	/*cout << "rows: " << _landmarks.rows << endl;
	cout << "cols: " << _landmarks.cols << endl;*/

	MyPoint *LandMark = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2]);
	if (LandMark->_x == 0.0f && LandMark->_y == 0.0f && LandMark->_z == 0.0f) {
		glEnd();
		delete LandMark;
		cout << "landmark VMAP null" << endl;
		return;
	}

	glBegin(GL_POINTS);
	//glColor4f(1.0, 0.0, 0.0, 1.0);
	//glVertex3f(LandMark->_x, LandMark->_y, LandMark->_z);

	/*glEnd();
	delete LandMark;
	return;*/

	//MyMesh * refMesh = Blendshape[0];
	//glColor4f(0.0, 0.0, 1.0, 1.0);
	//glVertex3f(refMesh->_vertices[FACIAL_LANDMARKS[i]]->_x, refMesh->_vertices[FACIAL_LANDMARKS[i]]->_y, refMesh->_vertices[FACIAL_LANDMARKS[i]]->_z);

	int u = _landmarksBump.at<int>(0, i);
	int v = _landmarksBump.at<int>(1, i);

	if (u < 0 || u > BumpHeight - 1 || v < 0 || v > BumpWidth - 1) {
		glEnd();
		delete LandMark;
		cout << "Out of bound" << endl;
		return;
	}

	if (_Bump.at<cv::Vec4f>(u, v)[1] == 0.0f) {
		glEnd();
		delete LandMark;
		cout << "Mask null" << endl;
		return;
	}

	float pt[3];
	pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
	pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
	pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

	/*pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0];
	pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[1];
	pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[2];*/

	/*float ptRef[3];
	ptRef[0] = _Vtx[0][3 * i]; ptRef[1] = _Vtx[0][3 * i + 1]; ptRef[2] = _Vtx[0][3 * i + 2];

	for (int k = 1; k < 28; k++) {
		pt[0] = pt[0] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i] - ptRef[0]);
		pt[1] = pt[1] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i + 1] - ptRef[1]);
		pt[2] = pt[2] + _BlendshapeCoeff[k] * (_Vtx[k][3 * i + 2] - ptRef[2]);
	}
	pt[0] = pt[0] + _Translation_inv(0);
	pt[1] = pt[1] + _Translation_inv(1);
	pt[2] = pt[2] + _Translation_inv(2);*/

	glColor4f(0.0, 1.0, 0.0, 1.0);
	glVertex3f(pt[0], pt[1], pt[2]);

	glEnd();
	delete LandMark;

}

int FaceCap::Update() {
	//if (_idx > 950)
	//	return 0;

	char filename_buff[100];
	cv::Mat depth;
	cv::Mat color;
	cv::Mat color_origin;
	cv::Mat color_coord;

	// enable depth camera
	if (_bSensor) {
		_sensorManager->update();

		depth = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC3);
		color = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);

		//_CoordMappingIn = (int *)malloc(2 * cDepthHeight*cDepthWidth*sizeof(int));
		//std::memset(_CoordMappingIn, -1, 2 * cDepthHeight*cDepthWidth*sizeof(int));
		//_CoordMappingInD2RGB = (int *)malloc(2 * cDepthHeight*cDepthWidth*sizeof(int));

		BYTE* h_colorFrame = _sensorManager->getColorFrame();
		USHORT* h_depthFrame = _sensorManager->getDepthFrame();
		LONG* h_colorCoord = _sensorManager->getColorCoord();

		float pt[3];
		float pt_T[3];
		float p_indx[2];
		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {

				LONG colorForDepthX = h_colorCoord[(i*cDepthWidth + j) * 2];
				LONG colorForDepthY = h_colorCoord[(i*cDepthWidth + j) * 2 + 1];

				// check if the color coordinates lie within the range of the color map
				if (colorForDepthX > 0 && colorForDepthX < cDepthWidth - 1 && colorForDepthY > 0 && colorForDepthY < cDepthHeight - 1)
				{
					color.at<cv::Vec3b>(i, j)[0] = h_colorFrame[(colorForDepthY*cDepthWidth + colorForDepthX) * 4];
					color.at<cv::Vec3b>(i, j)[1] = h_colorFrame[(colorForDepthY*cDepthWidth + colorForDepthX) * 4 + 1];
					color.at<cv::Vec3b>(i, j)[2] = h_colorFrame[(colorForDepthY*cDepthWidth + colorForDepthX) * 4 + 2];
					depth.at<cv::Vec3w>(i, j)[0] = h_depthFrame[i*cDepthWidth + j];
					depth.at<cv::Vec3w>(i, j)[1] = h_depthFrame[i*cDepthWidth + j];
					depth.at<cv::Vec3w>(i, j)[2] = h_depthFrame[i*cDepthWidth + j];
				}
			}
		}
	}
	else{
		sprintf_s(filename_buff, "%s\\Depth_%d.tiff", _path, _idx);
		depth = cv::imread(string(filename_buff), CV_LOAD_IMAGE_UNCHANGED);
		if (!depth.data) {
			cout << "no depth " << filename_buff << endl;
			return 3;
		}

		sprintf_s(filename_buff, "%s\\RGB_%d.tiff", _path, _idx);
		color = cv::imread(string(filename_buff), CV_LOAD_IMAGE_UNCHANGED);
		if (!color.data) {
			cout << "no color " << filename_buff << endl;
			return 3;
		}

	}

	if (_bSensor) {
		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {
				_imgD.at<cv::Vec4w>(i, j)[0] = depth.at<cv::Vec3w>(i, j)[0];
				_imgD.at<cv::Vec4w>(i, j)[1] = depth.at<cv::Vec3w>(i, j)[1];
				_imgD.at<cv::Vec4w>(i, j)[2] = depth.at<cv::Vec3w>(i, j)[2];
				_imgD.at<cv::Vec4w>(i, j)[3] = 0;
				_imgS.at<cv::Vec4b>(i, j)[0] = color.at<cv::Vec3b>(i, j)[0];
				_imgS.at<cv::Vec4b>(i, j)[1] = color.at<cv::Vec3b>(i, j)[1];
				_imgS.at<cv::Vec4b>(i, j)[2] = color.at<cv::Vec3b>(i, j)[2];
				_imgS.at<cv::Vec4b>(i, j)[3] = 0;
			}
		}
	}
	else {
		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {
				_imgD.at<cv::Vec4w>(i, j)[0] = depth.at<unsigned short>(i, j);
				_imgD.at<cv::Vec4w>(i, j)[1] = depth.at<unsigned short>(i, j);
				_imgD.at<cv::Vec4w>(i, j)[2] = depth.at<unsigned short>(i, j);
				_imgD.at<cv::Vec4w>(i, j)[3] = 0;
				_imgS.at<cv::Vec4b>(i, j)[0] = color.at<cv::Vec3b>(i, j)[0];
				_imgS.at<cv::Vec4b>(i, j)[1] = color.at<cv::Vec3b>(i, j)[1];
				_imgS.at<cv::Vec4b>(i, j)[2] = color.at<cv::Vec3b>(i, j)[2];
				_imgS.at<cv::Vec4b>(i, j)[3] = 0;
			}
		}
	}
	_imgS.copyTo(_imgC);

	/*cv::imshow("Color image", _imgS);
	cv::imshow("Depth image", _imgD);
	cv::waitKey(0);*/

	/*char filename[100];
	sprintf_s(filename, "Seq\\KinectV1-2\\RGB_%d.tiff", _idx);
	cv::imwrite(filename, color);
	sprintf_s(filename, "Seq\\KinectV1-2\\Depth_%d.tiff", _idx);
	cv::imwrite(filename, depth);*/

	_idx++;
	return 2;
}

void FaceCap::Pop()
{
	if (!_depth.empty()) {
		_CoordMaps.pop();
		_CoordMapsD2RGB.pop();
		_color.pop();
		_segmented_color.pop();
		_ptsQ.pop();
		_depth.pop();
		_ptsRect.pop();
		_klabelsSet.pop();
	}
}

int FaceCap::LoadToSave(int k) {
	if (_ptsQ.empty())
		return -1;

	/*if (_idx_curr < 10) {
		_idx_curr++;
		return 0;
	}*/

	_depth.front().copyTo(_depth_in[k]);
	_color.front().copyTo(_color_in[k]);

	_idx_thread[k] = _idx_curr;	

	_idx_curr++;
	return 1;
}

int FaceCap::SaveData(int k) {

	cv::Mat depth;
	cv::Mat color;
	depth = cv::Mat(cDepthHeight, cDepthWidth, CV_16UC1);
	color = cv::Mat(cDepthHeight, cDepthWidth, CV_8UC3);

	for (int i = 0; i < cDepthHeight; i++) {
		for (int j = 0; j < cDepthWidth; j++) {
			//depth.at<cv::Vec3w>(i, j)[0] = _depth_in[k].at<cv::Vec4w>(i, j)[0];
			//depth.at<cv::Vec3w>(i, j)[1] = _depth_in[k].at<cv::Vec4w>(i, j)[1];
			//depth.at<cv::Vec3w>(i, j)[2] = _depth_in[k].at<cv::Vec4w>(i, j)[2];
			depth.at<unsigned short>(i, j) = _depth_in[k].at<cv::Vec4w>(i, j)[0];

			color.at<cv::Vec3b>(i, j)[0] = _color_in[k].at<cv::Vec4b>(i, j)[0];
			color.at<cv::Vec3b>(i, j)[1] = _color_in[k].at<cv::Vec4b>(i, j)[1];
			color.at<cv::Vec3b>(i, j)[2] = _color_in[k].at<cv::Vec4b>(i, j)[2];
		}
	}

	//cv::imshow("Color image", color);
	/*cv::imshow("Depth image", depth);
	cv::waitKey(100);*/

	char filename[100];

	sprintf_s(filename, "%s\\RGB_%d.tiff", _path, _idx_thread[k]);
	cv::imwrite(filename, color);

	sprintf_s(filename, "%s\\Depth_%d.tiff", _path, _idx_thread[k]);
	cv::imwrite(filename, depth);

	depth.release();
	color.release();

	return 0;
}

bool FaceCap::Compute3DDataCPU(int idx) {
	//if (_ptsQ.empty())
	//	return false;

	//std::memcpy(_klabels, _klabelsSet.front(), _FaceL.height*_FaceL.width*sizeof(int));

	//cv::Mat tmpLm = _ptsQ.front();
	int nbLM = _pts.cols;
	if (nbLM < 43) {
		for (int i = 0; i < 43; i++) {
			_landmarks_in[idx].at<float>(0, i) = 0.0;
			_landmarks_in[idx].at<float>(1, i) = 0.0;
		}
	}
	else {
		int count_invalid = 0;
		for (int i = 0; i < 43; i++) {
			_landmarks_in[idx].at<float>(0, i) = _pts.at <float>(0, i);
			_landmarks_in[idx].at<float>(1, i) = _pts.at <float>(1, i);
		}
	}
	_idx_curr++;

	//cv::Mat depth_curr = _depth.front();
	//cv::Mat segmented_curr = _segmented_color.front();
	//_RGBMap = _color.front();
	_imgC.copyTo(_color_in[idx]);

	// Bilateral filter ??

	// Compute Vertex map
	std::vector<cv::Mat> channels(3);
	// split depth_curr:
	split(_imgD, channels);
	cv::Mat d = channels[0];
	d.convertTo(d, CV_32F, (_IntrinsicRGB[9] / _IntrinsicRGB[10]));

	cv::Mat x = d.mul(_x_raw);
	cv::Mat y = d.mul(_y_raw);

	std::vector<cv::Mat> matrices = { x, y, d, _ones };
	cv::merge(matrices, _VMap_in[idx]);

	// Compute Normal map
	cv::Mat kernel = cv::Mat::zeros(3, 3, CV_32F);
	kernel.at<float>(1, 0) = 1.0f;
	kernel.at<float>(1, 2) = -1.0f;
	filter2D(_VMap_in[idx], _grad_x, -1, kernel);
	kernel = cv::Mat::zeros(3, 3, CV_32F);
	kernel.at<float>(0, 1) = -1.0f;
	kernel.at<float>(2, 1) = 1.0f;
	filter2D(_VMap_in[idx], _grad_y, -1, kernel);

	std::vector<cv::Mat> channelsX(4);
	split(_grad_x, channelsX);
	std::vector<cv::Mat> channelsY(4);
	split(_grad_y, channelsY);

	x = channelsX[1].mul(channelsY[2]) - channelsX[2].mul(channelsY[1]);
	y = channelsX[2].mul(channelsY[0]) - channelsX[0].mul(channelsY[2]);
	cv::Mat z = channelsX[0].mul(channelsY[1]) - channelsX[1].mul(channelsY[0]);
	cv::Mat mag_f(x.mul(x) + y.mul(y) + z.mul(z));
	cv::sqrt(mag_f, mag_f);
	x = x.mul(1/mag_f);
	y = y.mul(1/mag_f);
	z = z.mul(1/mag_f);

	std::vector<cv::Mat> matricesN = { x, y, z, _ones };
	cv::merge(matricesN, _NMap_in[idx]);

	/*cout << "_NMap: " << _NMap.at<cv::Vec4f>(300, 110)[0] << ", " << _NMap.at<cv::Vec4f>(300, 110)[1] << ", " << _NMap.at<cv::Vec4f>(300, 110)[2] << endl;
	cv::imshow("NMap", _NMap);
	cv::waitKey(0);*/

	return true;
}

void FaceCap::ComputeNormalesDepth() {
	// Compute normals
	MyPoint *currPt;
	float p1[3];
	float p2[3];
	float p3[3];
	float n_p[3];
	float n_p1[3];
	float n_p2[3];
	float n_p3[3];
	float n_p4[3];
	float norm_n;
	for (int i = 1; i < cDepthHeight - 1; i++) {
		for (int j = 1; j < cDepthWidth - 1; j++) {

			if (_depth.front().at<cv::Vec3w>(i, j)[2] == 0)
				continue;

			currPt = _vertices[i*cDepthWidth + j];

			unsigned short n_tot = 0;

			p1[0] = currPt->_x;
			p1[1] = currPt->_y;
			p1[2] = currPt->_z;

			n_p1[0] = 0.0; n_p1[1] = 0.0; n_p1[2] = 0.0;
			n_p2[0] = 0.0; n_p2[1] = 0.0; n_p2[2] = 0.0;
			n_p3[0] = 0.0; n_p3[1] = 0.0; n_p3[2] = 0.0;
			n_p4[0] = 0.0; n_p4[1] = 0.0; n_p4[2] = 0.0;

			////////////////////////// Triangle 1 /////////////////////////////////
			if (_vertices[(i + 1)*cDepthWidth + j] == NULL || _vertices[i*cDepthWidth + (j + 1)] == NULL)
				goto TRIANGLE2;

			p2[0] = _vertices[(i + 1)*cDepthWidth + j]->_x;
			p2[1] = _vertices[(i + 1)*cDepthWidth + j]->_y;
			p2[2] = _vertices[(i + 1)*cDepthWidth + j]->_z;

			p3[0] = _vertices[i*cDepthWidth + (j + 1)]->_x;
			p3[1] = _vertices[i*cDepthWidth + (j + 1)]->_y;
			p3[2] = _vertices[i*cDepthWidth + (j + 1)]->_z;

			if (p2[2] != 0.0 && p3[2] != 0.0) {
				n_p1[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p1[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p1[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p1[0] * n_p1[0] + n_p1[1] * n_p1[1] + n_p1[2] * n_p1[2]);

				if (norm_n != 0.0) {
					n_p1[0] = n_p1[0] / sqrt(norm_n);
					n_p1[1] = n_p1[1] / sqrt(norm_n);
					n_p1[2] = n_p1[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 2 /////////////////////////////////
		TRIANGLE2:
			if (_vertices[(i - 1)*cDepthWidth + j] == NULL || _vertices[i*cDepthWidth + (j + 1)] == NULL)
				goto TRIANGLE3;

			p2[0] = _vertices[i*cDepthWidth + (j + 1)]->_x;
			p2[1] = _vertices[i*cDepthWidth + (j + 1)]->_y;
			p2[2] = _vertices[i*cDepthWidth + (j + 1)]->_z;

			p3[0] = _vertices[(i - 1)*cDepthWidth + j]->_x;
			p3[1] = _vertices[(i - 1)*cDepthWidth + j]->_y;
			p3[2] = _vertices[(i - 1)*cDepthWidth + j]->_z;

			if (p2[2] != 0.0 && p3[2] != 0.0) {
				n_p2[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p2[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p2[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p2[0] * n_p2[0] + n_p2[1] * n_p2[1] + n_p2[2] * n_p2[2]);

				if (norm_n != 0.0) {
					n_p2[0] = n_p2[0] / sqrt(norm_n);
					n_p2[1] = n_p2[1] / sqrt(norm_n);
					n_p2[2] = n_p2[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 3 /////////////////////////////////
		TRIANGLE3:
			if (_vertices[(i - 1)*cDepthWidth + j] == NULL || _vertices[i*cDepthWidth + (j - 1)] == NULL)
				goto TRIANGLE4;

			p2[0] = _vertices[(i - 1)*cDepthWidth + j]->_x;
			p2[1] = _vertices[(i - 1)*cDepthWidth + j]->_y;
			p2[2] = _vertices[(i - 1)*cDepthWidth + j]->_z;

			p3[0] = _vertices[i*cDepthWidth + (j - 1)]->_x;
			p3[1] = _vertices[i*cDepthWidth + (j - 1)]->_y;
			p3[2] = _vertices[i*cDepthWidth + (j - 1)]->_z;

			if (p2[2] != 0.0 && p3[2] != 0.0) {
				n_p3[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p3[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p3[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p3[0] * n_p3[0] + n_p3[1] * n_p3[1] + n_p3[2] * n_p3[2]);

				if (norm_n != 0) {
					n_p3[0] = n_p3[0] / sqrt(norm_n);
					n_p3[1] = n_p3[1] / sqrt(norm_n);
					n_p3[2] = n_p3[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 4 /////////////////////////////////
		TRIANGLE4:
			if (_vertices[(i + 1)*cDepthWidth + j] == NULL || _vertices[i*cDepthWidth + (j - 1)] == NULL)
				goto ENDNORMPROC;

			p2[0] = _vertices[i*cDepthWidth + (j - 1)]->_x;
			p2[1] = _vertices[i*cDepthWidth + (j - 1)]->_y;
			p2[2] = _vertices[i*cDepthWidth + (j - 1)]->_z;

			p3[0] = _vertices[(i + 1)*cDepthWidth + j]->_x;
			p3[1] = _vertices[(i + 1)*cDepthWidth + j]->_y;
			p3[2] = _vertices[(i + 1)*cDepthWidth + j]->_z;

			if (p2[2] != 0.0 && p3[2] != 0.0) {
				n_p4[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p4[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p4[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p4[0] * n_p4[0] + n_p4[1] * n_p4[1] + n_p4[2] * n_p4[2]);

				if (norm_n != 0) {
					n_p4[0] = n_p4[0] / sqrt(norm_n);
					n_p4[1] = n_p4[1] / sqrt(norm_n);
					n_p4[2] = n_p4[2] / sqrt(norm_n);

					n_tot++;
				}
			}

		ENDNORMPROC:
			if (n_tot == 0) {
				currPt->_Nx = 0.0f;
				currPt->_Ny = 0.0f;
				currPt->_Nz = 0.0f;
				continue;
			}

			n_p[0] = (n_p1[0] + n_p2[0] + n_p3[0] + n_p4[0]) / float(n_tot);
			n_p[1] = (n_p1[1] + n_p2[1] + n_p3[1] + n_p4[1]) / float(n_tot);
			n_p[2] = (n_p1[2] + n_p2[2] + n_p3[2] + n_p4[2]) / float(n_tot);

			norm_n = sqrt(n_p[0] * n_p[0] + n_p[1] * n_p[1] + n_p[2] * n_p[2]);

			if (norm_n != 0) {
				currPt->_Nx = n_p[0] / norm_n;
				currPt->_Ny = n_p[1] / norm_n;
				currPt->_Nz = n_p[2] / norm_n;
			}
			else {
				currPt->_Nx = 0.0f;
				currPt->_Ny = 0.0f;
				currPt->_Nz = 0.0f;
			}

		}
	}
}

void FaceCap::DetectFeatures(cv::CascadeClassifier *face_cascade, bool draw){
	float score = 0.f;

	if (_restart) {
		int minFaceH = 50;
		cv::Size minFace(minFaceH, minFaceH); // minimum face size to detect
		vector<cv::Rect> faces;
		face_cascade->detectMultiScale(_imgC, faces, 1.2, 2, 0, minFace);
		if (!faces.empty()) {
			cv::Rect& faceL = *max_element(faces.begin(), faces.end(), Utility::compareRect);
			_sdm->detect(_imgC, faceL, _pts, score);
			_pRect = faceL;

			_FaceL.x = _pRect.x - 20;
			_FaceL.y = _pRect.y - 20;
			_FaceL.height = _pRect.height + 40;
			_FaceL.width = _pRect.width + 40;
		}
	}
	else {
		_sdm->track(_imgC, _prevPts, _pts, score);
	}

	if (score > _minScore) {
		_restart = false;
		_prevPts = _pts.clone();
	}
	/*else {
		_restart = true;
	}*/

	if (!_restart) {
		if (_pts.rows > 0) {
			Utility::drawFace(_imgS, _pts);
		}
	}
}
// 

// Re-scale all blendshapes to match user landmarks
bool FaceCap::Rescale(vector<MyMesh *> Blendshape) {
	MyMesh * RefMesh = Blendshape[0];

	if (_landmarks.cols == 0) {
		return false;
	}
	// Compute average factor in X length from outer corner of eyes
	MyPoint *LeftEyeL = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 19))*cDepthWidth + Myround(_landmarks.at<float>(0, 19)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 19))*cDepthWidth + Myround(_landmarks.at<float>(0, 19)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 19))*cDepthWidth + Myround(_landmarks.at<float>(0, 19)))[2]);
	if (LeftEyeL->_x == 0.0f && LeftEyeL->_y == 0.0f && LeftEyeL->_z == 0.0f) {
		cout << "Landmark LeftEyeL NULL" << endl;
		delete LeftEyeL;
		return false;
	}
	MyPoint *RightEyeR = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 28))*cDepthWidth + Myround(_landmarks.at<float>(0, 28)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 28))*cDepthWidth + Myround(_landmarks.at<float>(0, 28)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 28))*cDepthWidth + Myround(_landmarks.at<float>(0, 28)))[2]);
	if (RightEyeR->_x == 0.0f && RightEyeR->_y == 0.0f && RightEyeR->_z == 0.0f) {
		cout << "Landmark RightEyeR NULL" << endl;
		delete LeftEyeL;
		delete RightEyeR;
		return false;
	}
	float eye_dist = sqrt((LeftEyeL->_x - RightEyeR->_x)*(LeftEyeL->_x - RightEyeR->_x) + (LeftEyeL->_y - RightEyeR->_y)*(LeftEyeL->_y - RightEyeR->_y) + (LeftEyeL->_z - RightEyeR->_z)*(LeftEyeL->_z - RightEyeR->_z));
	float eye_dist_mesh = sqrt((RefMesh->Landmark(19)->_x - RefMesh->Landmark(28)->_x)*(RefMesh->Landmark(19)->_x - RefMesh->Landmark(28)->_x) +
		(RefMesh->Landmark(19)->_y - RefMesh->Landmark(28)->_y)*(RefMesh->Landmark(19)->_y - RefMesh->Landmark(28)->_y) +
		(RefMesh->Landmark(19)->_z - RefMesh->Landmark(28)->_z)*(RefMesh->Landmark(19)->_z - RefMesh->Landmark(28)->_z));
	float fact = eye_dist / eye_dist_mesh;
	delete LeftEyeL;
	delete RightEyeR;

	// Compute average factor in X length from inner corner of eyes
	MyPoint *LeftEyeR = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 22))*cDepthWidth + Myround(_landmarks.at<float>(0, 22)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 22))*cDepthWidth + Myround(_landmarks.at<float>(0, 22)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 22))*cDepthWidth + Myround(_landmarks.at<float>(0, 22)))[2]);
	if (LeftEyeR->_x == 0.0f && LeftEyeR->_y == 0.0f && LeftEyeR->_z == 0.0f) {
		cout << "Landmark LeftEyeR NULL" << endl;
		delete LeftEyeR;
		return false;
	}
	MyPoint *RightEyeL = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 25))*cDepthWidth + Myround(_landmarks.at<float>(0, 25)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 25))*cDepthWidth + Myround(_landmarks.at<float>(0, 25)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 25))*cDepthWidth + Myround(_landmarks.at<float>(0, 25)))[2]);
	if (RightEyeL->_x == 0.0f && RightEyeL->_y == 0.0f && RightEyeL->_z == 0.0f) {
		cout << "Landmark RightEyeL NULL" << endl;
		delete LeftEyeR;
		delete RightEyeL;
		return false;
	}
	eye_dist = sqrt((LeftEyeR->_x - RightEyeL->_x)*(LeftEyeR->_x - RightEyeL->_x) + (LeftEyeR->_y - RightEyeL->_y)*(LeftEyeR->_y - RightEyeL->_y) + (LeftEyeR->_z - RightEyeL->_z)*(LeftEyeR->_z - RightEyeL->_z));
	eye_dist_mesh = sqrt((RefMesh->Landmark(22)->_x - RefMesh->Landmark(25)->_x)*(RefMesh->Landmark(22)->_x - RefMesh->Landmark(25)->_x) +
		(RefMesh->Landmark(22)->_y - RefMesh->Landmark(25)->_y)*(RefMesh->Landmark(22)->_y - RefMesh->Landmark(25)->_y) +
		(RefMesh->Landmark(22)->_z - RefMesh->Landmark(25)->_z)*(RefMesh->Landmark(22)->_z - RefMesh->Landmark(25)->_z));
	fact += eye_dist / eye_dist_mesh;
	delete LeftEyeR;
	delete RightEyeL;

	// Compute average factor in X length from mouth
	MyPoint *LeftMouth = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 31))*cDepthWidth + Myround(_landmarks.at<float>(0, 31)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 31))*cDepthWidth + Myround(_landmarks.at<float>(0, 31)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 31))*cDepthWidth + Myround(_landmarks.at<float>(0, 31)))[2]);
	if (LeftMouth->_x == 0.0f && LeftMouth->_y == 0.0f && LeftMouth->_z == 0.0f) {
		cout << "Landmark LeftMouth NULL" << endl;
		delete LeftMouth;
		return false;
	}
	MyPoint *RightMouth = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 37))*cDepthWidth + Myround(_landmarks.at<float>(0, 37)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 37))*cDepthWidth + Myround(_landmarks.at<float>(0, 37)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 37))*cDepthWidth + Myround(_landmarks.at<float>(0, 37)))[2]);
	if (RightMouth->_x == 0.0f && RightMouth->_y == 0.0f && RightMouth->_z == 0.0f) {
		cout << "Landmark RightEyeL NULL" << endl;
		delete LeftMouth;
		delete RightMouth;
		return false;
	}
	eye_dist = sqrt((LeftMouth->_x - RightMouth->_x)*(LeftMouth->_x - RightMouth->_x) + (LeftMouth->_y - RightMouth->_y)*(LeftMouth->_y - RightMouth->_y) + (LeftMouth->_z - RightMouth->_z)*(LeftMouth->_z - RightMouth->_z));
	eye_dist_mesh = sqrt((RefMesh->Landmark(31)->_x - RefMesh->Landmark(37)->_x)*(RefMesh->Landmark(31)->_x - RefMesh->Landmark(37)->_x) +
		(RefMesh->Landmark(31)->_y - RefMesh->Landmark(37)->_y)*(RefMesh->Landmark(31)->_y - RefMesh->Landmark(37)->_y) +
		(RefMesh->Landmark(31)->_z - RefMesh->Landmark(37)->_z)*(RefMesh->Landmark(31)->_z - RefMesh->Landmark(37)->_z));
	fact += eye_dist / eye_dist_mesh;
	delete LeftMouth;
	delete RightMouth;


	// Compute average factor in Y length from nose
	MyPoint *UpNose = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 10))*cDepthWidth + Myround(_landmarks.at<float>(0, 10)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 10))*cDepthWidth + Myround(_landmarks.at<float>(0, 10)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 10))*cDepthWidth + Myround(_landmarks.at<float>(0, 10)))[2]);
	if (UpNose->_x == 0.0f && UpNose->_y == 0.0f && UpNose->_z == 0.0f) {
		cout << "Landmark UpNose NULL" << endl;
		delete UpNose;
		return false;
	}
	MyPoint *DownNose = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16))*cDepthWidth + Myround(_landmarks.at<float>(0, 16)))[0],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16))*cDepthWidth + Myround(_landmarks.at<float>(0, 16)))[1],
		_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16))*cDepthWidth + Myround(_landmarks.at<float>(0, 16)))[2]);
	if (DownNose->_x == 0.0f && DownNose->_y == 0.0f && DownNose->_z == 0.0f) {
		cout << "Landmark DownNose NULL" << endl;
		delete UpNose;
		delete DownNose;
		return false;
	}
	eye_dist = sqrt((UpNose->_x - DownNose->_x)*(UpNose->_x - DownNose->_x) + (UpNose->_y - DownNose->_y)*(UpNose->_y - DownNose->_y) + (UpNose->_z - DownNose->_z)*(UpNose->_z - DownNose->_z));
	eye_dist_mesh = sqrt((RefMesh->Landmark(10)->_x - RefMesh->Landmark(16)->_x)*(RefMesh->Landmark(10)->_x - RefMesh->Landmark(16)->_x) +
		(RefMesh->Landmark(10)->_y - RefMesh->Landmark(16)->_y)*(RefMesh->Landmark(10)->_y - RefMesh->Landmark(16)->_y) +
		(RefMesh->Landmark(10)->_z - RefMesh->Landmark(16)->_z)*(RefMesh->Landmark(10)->_z - RefMesh->Landmark(16)->_z));
	fact += eye_dist / eye_dist_mesh;
	delete UpNose;
	delete DownNose;

	fact = fact / 4.0;
	cout << "Scale factor: " << fact << endl;

	for (vector<MyMesh *>::iterator it = Blendshape.begin(); it != Blendshape.end(); it++) {
		(*it)->Scale(fact);
	}
	return true;
}

bool FaceCap::AlignToFace(vector<MyMesh *> Blendshape, bool inverted) {
	// Rotate the reference mesh and compute translation
	MyMesh * RefMesh = Blendshape[0];

	/// ESTIMATE RIGID ALIGNMENT ////////////
	
	Eigen::Matrix4d Transfo = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d Transfo_prev = Eigen::Matrix4d::Identity();
	Eigen::Matrix<double, 3, 7, Eigen::RowMajor> Jac;
	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
	Eigen::Matrix<double, 6, 1> b;
	Eigen::Matrix<double, 6, 1> result;

	float Mat[27];
	float pt_T[3];
	int iter = 0;
	int _max_iter = 100;
	bool converged = false;
	double lambda = 1.0;
	float prev_res = 1.0e10;

	while (!converged) {
		float residual = 0.0f;
		for (int i = 0; i < 27; i++)
			Mat[i] = 0.0f;

		int count_matches = 0;
		for (int i = 0; i < 43; i++) {

			MyPoint *LANDMARKPT = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0],
												_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1],
												_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2]);
			if (LANDMARKPT->_x == 0.0f && LANDMARKPT->_y == 0.0f && LANDMARKPT->_z == 0.0f) {
				//cout << "Landmark " << i << " NULL" << endl;
				delete LANDMARKPT;
				continue;
			}
			count_matches++;

			MyPoint *LANDMARK_mesh = RefMesh->Landmark(i);
			pt_T[0] = LANDMARK_mesh->_x * Transfo(0, 0) + LANDMARK_mesh->_y * Transfo(0, 1) + LANDMARK_mesh->_z * Transfo(0, 2) + Transfo(0, 3);
			pt_T[1] = LANDMARK_mesh->_x * Transfo(1, 0) + LANDMARK_mesh->_y * Transfo(1, 1) + LANDMARK_mesh->_z * Transfo(1, 2) + Transfo(1, 3);
			pt_T[2] = LANDMARK_mesh->_x * Transfo(2, 0) + LANDMARK_mesh->_y * Transfo(2, 1) + LANDMARK_mesh->_z * Transfo(2, 2) + Transfo(2, 3);

			Jac(0, 0) = 1; Jac(0, 1) = 0.0f; Jac(0, 2) = 0.0f; Jac(0, 3) = 0.0f; Jac(0, 4) = pt_T[2]; Jac(0, 5) = -pt_T[1];
			Jac(0, 6) = LANDMARKPT->_x - pt_T[0];

			Jac(1, 0) = 0.0f; Jac(1, 1) = 1.0f; Jac(1, 2) = 0.0f; Jac(1, 3) = -pt_T[2]; Jac(1, 4) = 0.0f; Jac(1, 5) = pt_T[0];
			Jac(1, 6) = LANDMARKPT->_y - pt_T[1];

			Jac(2, 0) = 0.0f; Jac(2, 1) = 0.0f; Jac(2, 2) = 1.0f; Jac(2, 3) = pt_T[1]; Jac(2, 4) = -pt_T[0]; Jac(2, 5) = 0.0f;
			Jac(2, 6) = LANDMARKPT->_z - pt_T[2];

			int shift = 0;
			for (int k = 0; k < 6; ++k)        //rows
			{
				for (int l = k; l < 7; ++l)          // cols + b
				{
					Mat[shift] = Mat[shift] + Jac(0, k)*Jac(0, l) + Jac(1, k)*Jac(1, l) + Jac(2, k)*Jac(2, l);
					shift++;
				}
			}
			residual += Jac(0, 6)*Jac(0, 6) + Jac(1, 6)*Jac(1, 6) + Jac(2, 6)*Jac(2, 6);

			delete LANDMARKPT;
		}

		if (count_matches < 3) {
			cout << "Not enough landmarks" << endl;
			return false;
		}

		if (residual > prev_res) {
			lambda = lambda / 2.0;
			Transfo = Transfo_prev;
			continue;
		}
		else {
			prev_res = residual;
			Transfo_prev = Transfo;
		}

		int shift = 0;
		for (int i = 0; i < 6; ++i) {  //rows
			for (int j = i; j < 7; ++j)    // cols + b
			{
				double value = double(Mat[shift++]);
				if (j == 6)       // vector b
					b(i) = value;
				else
					A(j, i) = A(i, j) = value;
			}
		}

		//checking nullspace
		float det = A.determinant();

		if (fabs(det) < 1e-15 || det != det)
		{
			if (det != det) std::cout << "qnan" << endl;
			std::cout << "det null" << endl;
			return false;
		}

		result = lambda * A.inverse() * b;

		// Update transformation matrix

		Eigen::Matrix4d delta_transfo = Exponential(result);
		Transfo = delta_transfo*Transfo;

		//std::cout << "Iter: " << iter << std::endl;
		//std::cout << "Transformation: " << Transfo << std::endl;

		iter++;

		if (iter > _max_iter || ((delta_transfo - Eigen::Matrix4d::Identity()).norm() < 1.0e-6)) {
			converged = true;
		}

	}

	//std::cout << "End: " << iter << std::endl;
	//std::cout << "Transformation: " << Transfo << std::endl;

	RefMesh->Transform(Transfo);

	// Affect values to deformed positions and normals
	RefMesh->AffectToTVal();

	for (vector<MyMesh *>::iterator it = Blendshape.begin() + 1; it != Blendshape.end(); it++) {
		(*it)->Transform(Transfo);
	}

	return true;
}

void FaceCap::DataProc() {

	float bumpIn[4];
	float weights[4];
	for (int i = 0; i < BumpHeight; i ++) {
		for (int j = 0; j < BumpWidth; j++) {
			int tid = i*BumpWidth + j;

			bumpIn[0] = _Bump.at<cv::Vec4f>(i, j)[0];
			bumpIn[1] = _Bump.at<cv::Vec4f>(i, j)[1];
			bumpIn[2] = _Bump.at<cv::Vec4f>(i, j)[2];
			bumpIn[3] = _Bump.at<cv::Vec4f>(i, j)[3];

			weights[0] = _WeightMap.at<cv::Vec4f>(i, j)[0];
			weights[1] = _WeightMap.at<cv::Vec4f>(i, j)[1];
			weights[2] = _WeightMap.at<cv::Vec4f>(i, j)[2];
			weights[3] = _WeightMap.at<cv::Vec4f>(i, j)[3];

			if (bumpIn[2] == -1.0f) {
				for (int k = 0; k < NB_BS; k++) {
					_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid] = 0.0f;
					_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 1] = 0.0f;
					_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 2] = 0.0f;
					_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 3] = 0.0f;
					_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 4] = 0.0f;
					_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 5] = 0.0f;
				}
				continue;
			}
			FaceGPU CurrFace = _triangles[int(bumpIn[2])];

			float nmle[3] = { 0.0f, 0.0f, 0.0f };
			float pt[3] = { 0.0f, 0.0f, 0.0f };

			pt[0] = (weights[0] * _verticesList[CurrFace._v1]._x + weights[1] * _verticesList[CurrFace._v2]._x + weights[2] * _verticesList[CurrFace._v3]._x);
			pt[1] = (weights[0] * _verticesList[CurrFace._v1]._y + weights[1] * _verticesList[CurrFace._v2]._y + weights[2] * _verticesList[CurrFace._v3]._y);
			pt[2] = (weights[0] * _verticesList[CurrFace._v1]._z + weights[1] * _verticesList[CurrFace._v2]._z + weights[2] * _verticesList[CurrFace._v3]._z);
			float ptRef[3] = { pt[0], pt[1], pt[2] };

			nmle[0] = (weights[0] * _verticesList[CurrFace._v1]._Nx + weights[1] * _verticesList[CurrFace._v2]._Nx + weights[2] * _verticesList[CurrFace._v3]._Nx);
			nmle[1] = (weights[0] * _verticesList[CurrFace._v1]._Ny + weights[1] * _verticesList[CurrFace._v2]._Ny + weights[2] * _verticesList[CurrFace._v3]._Ny);
			nmle[2] = (weights[0] * _verticesList[CurrFace._v1]._Nz + weights[1] * _verticesList[CurrFace._v2]._Nz + weights[2] * _verticesList[CurrFace._v3]._Nz);
			float tmp = sqrt(nmle[0] * nmle[0] + nmle[1] * nmle[1] + nmle[2] * nmle[2]);
			nmle[0] = nmle[0] / tmp;
			nmle[1] = nmle[1] / tmp;
			nmle[2] = nmle[2] / tmp;

			float nmleRef[3] = { nmle[0], nmle[1], nmle[2] };

			_VerticesBS[6 * tid] = ptRef[0];
			_VerticesBS[6 * tid + 1] = ptRef[1];
			_VerticesBS[6 * tid + 2] = ptRef[2];
			_VerticesBS[6 * tid + 3] = nmleRef[0];
			_VerticesBS[6 * tid + 4] = nmleRef[1];
			_VerticesBS[6 * tid + 5] = nmleRef[2];

			float nTmp[3];
			float pTmp[3];
			for (int k = 1; k < NB_BS; k++) {
				nTmp[0] = (weights[0] * _verticesList[k * 4325 + CurrFace._v1]._Nx + weights[1] * _verticesList[k * 4325 + CurrFace._v2]._Nx + weights[2] * _verticesList[k * 4325 + CurrFace._v3]._Nx);
				nTmp[1] = (weights[0] * _verticesList[k * 4325 + CurrFace._v1]._Ny + weights[1] * _verticesList[k * 4325 + CurrFace._v2]._Ny + weights[2] * _verticesList[k * 4325 + CurrFace._v3]._Ny);
				nTmp[2] = (weights[0] * _verticesList[k * 4325 + CurrFace._v1]._Nz + weights[1] * _verticesList[k * 4325 + CurrFace._v2]._Nz + weights[2] * _verticesList[k * 4325 + CurrFace._v3]._Nz);
				float tmp = sqrt(nTmp[0] * nTmp[0] + nTmp[1] * nTmp[1] + nTmp[2] * nTmp[2]);
				nTmp[0] = nTmp[0] / tmp;
				nTmp[1] = nTmp[1] / tmp;
				nTmp[2] = nTmp[2] / tmp;

				pTmp[0] = (weights[0] * _verticesList[k * 4325 + CurrFace._v1]._x + weights[1] * _verticesList[k * 4325 + CurrFace._v2]._x + weights[2] * _verticesList[k * 4325 + CurrFace._v3]._x);
				pTmp[1] = (weights[0] * _verticesList[k * 4325 + CurrFace._v1]._y + weights[1] * _verticesList[k * 4325 + CurrFace._v2]._y + weights[2] * _verticesList[k * 4325 + CurrFace._v3]._y);
				pTmp[2] = (weights[0] * _verticesList[k * 4325 + CurrFace._v1]._z + weights[1] * _verticesList[k * 4325 + CurrFace._v2]._z + weights[2] * _verticesList[k * 4325 + CurrFace._v3]._z);

				// This blended normal is not really a normal since it may be not normalized
				_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid] = (pTmp[0] - ptRef[0]);
				_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 1] = (pTmp[1] - ptRef[1]);
				_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 2] = (pTmp[2] - ptRef[2]);
				_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 3] = (nTmp[0] - nmleRef[0]);
				_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 4] = (nTmp[1] - nmleRef[1]);
				_VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 5] = (nTmp[2] - nmleRef[2]);
			}
		}
	}
}

bool FaceCap::ElasticRegistrationFull(vector<MyMesh *> Blendshape){

	MyMesh *RefMesh = Blendshape[0];

	/****Compute local tangent plane transforms*****/
	RefMesh->ComputeTgtPlane();

	int iter = 0;
	float pt[3];
	float nmle1[3];
	float nmle2[3];
	float a[3];
	while (iter < 10) {

		/****Point correspondences***/

		/****Solve linear system*****/
		// Build Matrix A
		/*
		A = [-Nx -Ny -Nz 0 ......... 0]
		[0 0 0 -Nx -Ny -Nz 0 ... 0]
		...
		[0............ -Nx -Ny -Nz]
		[0..0...1....-1 .......]
		[0..0...0 1...0 -1 ....]
		[0..0...0 0 1.0 0 -1...]*/

		bool found_coresp;
		float min_dist;
		int p_indx[2];
		int li, ui, lj, uj;
		float DepthP[3];
		float dist;
		float pointClose[3];
		int indx_V = 0;
		int nbMatches = 0;
		int indx_Match = 0;
		for (vector<Point3D<float> *>::iterator it = RefMesh->_vertices.begin(); it != RefMesh->_vertices.end(); it++) {
			// Search for corresponding point
			continue;
			min_dist = 1000.0;
			//cout << "(*it)->_Nz " << (*it)->_Nz << endl;
			if ((*it)->_Nz > 0.0f)
				continue;

			pt[0] = (*it)->_x;// (*it)->_x * _Rotation_inv(0, 0) + (*it)->_y*_Rotation_inv(0, 1) + (*it)->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
			pt[1] = (*it)->_y;//(*it)->_x * _Rotation_inv(1, 0) + (*it)->_y*_Rotation_inv(1, 1) + (*it)->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
			pt[2] = (*it)->_z;//(*it)->_x * _Rotation_inv(2, 0) + (*it)->_y*_Rotation_inv(2, 1) + (*it)->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

			int uv_u = Myround((*it)->_u * float(BumpHeight));
			int uv_v = Myround((*it)->_v * float(BumpWidth));
			//cout << "_LabelsMask.at<cv::Vec4b>(uv_u, uv_v)[3] " << _LabelsMask.at<cv::Vec4b>(uv_u, uv_v)[3] << endl;
			if (_LabelsMask.at<cv::Vec4b>(uv_u, uv_v)[3] > 100)
				continue;

			/*** Projective association ***/
			// Project the point onto the depth image
			p_indx[0] = min(cDepthWidth - 1, max(0, int(round((pt[0] / fabs(pt[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2]))));
			p_indx[1] = min(cDepthHeight - 1, max(0, int(round((pt[1] / fabs(pt[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3]))));

			li = max(p_indx[1] - 2, 0);
			ui = min(p_indx[1] + 3, cDepthHeight);
			lj = max(p_indx[0] - 2, 0);
			uj = min(p_indx[0] + 3, cDepthWidth);

			for (int i = li; i < ui; i++) {
				for (int j = lj; j < uj; j++) {
					DepthP[0] = _VMap.at<cv::Vec4f>(i, j)[0];
					DepthP[1] = _VMap.at<cv::Vec4f>(i, j)[1];
					DepthP[2] = _VMap.at<cv::Vec4f>(i, j)[2];
					if (DepthP[0] == 0.0 && DepthP[1] == 0.0 && DepthP[2] == 0.0f)
						continue;

					dist = sqrt((DepthP[0] - pt[0])*(DepthP[0] - pt[0]) + (DepthP[1] - pt[1])*(DepthP[1] - pt[1]) + (DepthP[2] - pt[2])*(DepthP[2] - pt[2]));

				//	cout << dist << endl;
					if (dist < min_dist) {
						min_dist = dist;
					}
				}
			}

			if (min_dist < 0.05)
				nbMatches++;
		}

		cout << "nbMatches: " << nbMatches << endl;

		int nb_columns = 3 * RefMesh->size();
		int nb_lines = 3 * (43 + nbMatches) + 3 * RefMesh->sizeV();
		SpMat A(nb_lines, nb_columns);
		Eigen::VectorXd b1(nb_lines);
		vector<TrplType> tripletList;

		for (vector<Point3D<float> *>::iterator it = RefMesh->_vertices.begin(); it != RefMesh->_vertices.end(); it++) {
			// Search for corresponding point
			continue;
			found_coresp = false;
			min_dist = 1000.0;
			if ((*it)->_Nz > 0.0f) {
				indx_V++;
				continue;
			}

			pt[0] = (*it)->_x;// (*it)->_x * _Rotation_inv(0, 0) + (*it)->_y*_Rotation_inv(0, 1) + (*it)->_z * _Rotation_inv(0, 2) + _Translation_inv(0);
			pt[1] = (*it)->_y;//(*it)->_x * _Rotation_inv(1, 0) + (*it)->_y*_Rotation_inv(1, 1) + (*it)->_z * _Rotation_inv(1, 2) + _Translation_inv(1);
			pt[2] = (*it)->_z;//(*it)->_x * _Rotation_inv(2, 0) + (*it)->_y*_Rotation_inv(2, 1) + (*it)->_z * _Rotation_inv(2, 2) + _Translation_inv(2);

			int uv_u = Myround((*it)->_u * float(BumpHeight));
			int uv_v = Myround((*it)->_v * float(BumpWidth));
			if (_LabelsMask.at<cv::Vec4b>(uv_u, uv_v)[3] > 100) {
				indx_V++;
				continue;
			}

			/*** Projective association ***/
			// Project the point onto the depth image
			p_indx[0] = min(cDepthWidth - 1, max(0, int(round((pt[0] / fabs(pt[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2]))));
			p_indx[1] = min(cDepthHeight - 1, max(0, int(round((pt[1] / fabs(pt[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3]))));

			li = max(p_indx[1] - 2, 0);
			ui = min(p_indx[1] + 3, cDepthHeight);
			lj = max(p_indx[0] - 2, 0);
			uj = min(p_indx[0] + 3, cDepthWidth);

			for (int i = li; i < ui; i++) {
				for (int j = lj; j < uj; j++) {
					DepthP[0] = _VMap.at<cv::Vec4f>(i, j)[0];
					DepthP[1] = _VMap.at<cv::Vec4f>(i, j)[1];
					DepthP[2] = _VMap.at<cv::Vec4f>(i, j)[2];
					if (DepthP[0] == 0.0 && DepthP[1] == 0.0 && DepthP[2] == 0.0f)
						continue;

					dist = sqrt((DepthP[0] - pt[0])*(DepthP[0] - pt[0]) + (DepthP[1] - pt[1])*(DepthP[1] - pt[1]) + (DepthP[2] - pt[2])*(DepthP[2] - pt[2]));

					if (dist < min_dist) {
						min_dist = dist;
						pointClose[0] = DepthP[0];// DepthP[0] * _Rotation(0, 0) + DepthP[1] * _Rotation(0, 1) + DepthP[2] * _Rotation(0, 2) + _Translation(0);
						pointClose[1] = DepthP[1];//DepthP[0] * _Rotation(1, 0) + DepthP[1] * _Rotation(1, 1) + DepthP[2] * _Rotation(1, 2) + _Translation(1);
						pointClose[2] = DepthP[2];//DepthP[0] * _Rotation(2, 0) + DepthP[1] * _Rotation(2, 1) + DepthP[2] * _Rotation(2, 2) + _Translation(2);
					}
				}
			}

			if (min_dist < 0.05)
				found_coresp = true;

			if (found_coresp) {
				tripletList.push_back(TrplType(3 * indx_Match, 3 * indx_V, 1.0));
				tripletList.push_back(TrplType(3 * indx_Match + 1, 3 * indx_V + 1, 1.0));
				tripletList.push_back(TrplType(3 * indx_Match + 2, 3 * indx_V + 2, 1.0));

				b1(3 * indx_Match) = 1.0*double(pointClose[0]);
				b1(3 * indx_Match + 1) = 1.0*double(pointClose[1]);
				b1(3 * indx_Match + 2) = 1.0*double(pointClose[2]);
				indx_Match++;
			}
			indx_V++;
		}

		/*****************************Add landmarks******************************************/
		float Landmark[3];
		float weight = 10.0f;
		for (int i = 0; i < 36; i++) { //43
			tripletList.push_back(TrplType(3 * (nbMatches + i), 3 * FACIAL_LANDMARKS[i], weight));
			tripletList.push_back(TrplType(3 * (nbMatches + i) + 1, 3 * FACIAL_LANDMARKS[i] + 1, weight));
			tripletList.push_back(TrplType(3 * (nbMatches + i) + 2, 3 * FACIAL_LANDMARKS[i] + 2, weight));

			Landmark[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[0];
			Landmark[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[1];
			Landmark[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[2];
			if (Landmark[0] == 0.0 && Landmark[1] == 0.0 && Landmark[2] == 0.0) {
				cout << "Landmark NULL!!" << endl;
				return false;
			}
			b1(3 * (nbMatches + i)) = weight *double(Landmark[0]);
			b1(3 * (nbMatches + i) + 1) = weight *double(Landmark[1]);
			b1(3 * (nbMatches + i) + 2) = weight *double(Landmark[2]);
		}

		/***************Populate matrix from neighboors of the vertices***************************/
		RefMesh->PopulateMatrix(&tripletList, &b1, 3 * (43 + nbMatches));

		A.setFromTriplets(tripletList.begin(), tripletList.end());

		SpMat MatA(nb_columns, nb_columns);
		Eigen::VectorXd b(nb_columns);
		MatA = A.transpose() * A;
		b = A.transpose() * b1;

		Eigen::SimplicialCholesky<SpMat> chol(MatA);  // performs a Cholesky factorization of A
		Eigen::VectorXd xres = chol.solve(b);    // use the factorization to solve for the given right hand side

		RefMesh->AffectToTVectorT(&xres);

		iter++;
	}

	/***********************Transfer expression deformation******************************/
	Eigen::VectorXd boV(3 * RefMesh->size());
	RefMesh->Deform(&boV);

	if (save_data)
		RefMesh->Write(string(dest_name) + string("\\DeformedMeshes\\Neutral.obj"));

	int indxMesh = 0;
	for (vector<MyMesh *>::iterator itMesh = Blendshape.begin() + 1; itMesh != Blendshape.end(); itMesh++) {

		Eigen::SimplicialCholesky<SpMat> chol(_MatList1[indxMesh]);  // performs a Cholesky factorization of A
		Eigen::VectorXd b = chol.solve(_MatList2[indxMesh] * boV);

		(*itMesh)->AffectToTVector(&b);

		if (save_data)
			(*itMesh)->Write(string(dest_name) + string("\\DeformedMeshes\\") + to_string(indxMesh) + string(".obj"));

		indxMesh++;
	}

	DataProc();

	return true;
}

void FaceCap::ElasticRegistration(vector<MyMesh *> Blendshape){
	if (_landmarks.cols == 0) {
		return;
	}

	MyMesh *RefMesh = Blendshape[0];
	float Landmark[3];

	/****Compute local tangent plane transforms*****/
	RefMesh->ComputeTgtPlane();

	int iter = 0;
	float pt[3];
	float nmle1[3];
	float nmle2[3];
	float a[3];
	while (iter < 3) {

		/****Point correspondences are the facial landmarks***/

		/****Solve linear system*****/
		// Build Matrix A
		/*
		A = [-Nx -Ny -Nz 0 ......... 0]
		[0 0 0 -Nx --Ny -Nz 0 ... 0]
		...
		[0............ -Nx -Ny -Nz]
		[0..0...1....-1 .......]
		[0..0...0 1...0 -1 ....]
		[0..0...0 0 1.0 0 -1...]*/

		int nb_columns = 3 * RefMesh->size();
		int nb_lines = 3 * 43 + 3 * RefMesh->sizeV();
		SpMat A(nb_lines, nb_columns);
		Eigen::VectorXd b1(nb_lines);
		vector<TrplType> tripletList;

		for (int i = 0; i < 43; i++) {
			/*if (i == 14 || i == 15 || i == 17 || i == 18) {
				tripletList.push_back(TrplType(3 * i, 3 * FACIAL_LANDMARKS[16], 1.0));
				tripletList.push_back(TrplType(3 * i + 1, 3 * FACIAL_LANDMARKS[16] + 1, 1.0));
				tripletList.push_back(TrplType(3 * i + 2, 3 * FACIAL_LANDMARKS[16] + 2, 1.0));

				Landmark[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16)), Myround(_landmarks.at<float>(0, 16)))[0];
				Landmark[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16)), Myround(_landmarks.at<float>(0, 16)))[1];
				Landmark[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, 16)), Myround(_landmarks.at<float>(0, 16)))[2];
				if (Landmark[0] == 0.0 && Landmark[1] == 0.0 && Landmark[2] == 0.0) {
					cout << "Landmark NULL!!" << endl;
					return;
				}
				b1(3 * i) = double(Landmark[0]);
				b1(3 * i + 1) = double(Landmark[1]);
				b1(3 * i + 2) = double(Landmark[2]);
				continue;
			}*/
			tripletList.push_back(TrplType(3 * i, 3 * FACIAL_LANDMARKS[i], 1.0));
			tripletList.push_back(TrplType(3 * i + 1, 3 * FACIAL_LANDMARKS[i] + 1, 1.0));
			tripletList.push_back(TrplType(3 * i + 2, 3 * FACIAL_LANDMARKS[i] + 2, 1.0));

			Landmark[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[0];
			Landmark[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[1];
			Landmark[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i)), Myround(_landmarks.at<float>(0, i)))[2];
			if (Landmark[0] == 0.0 && Landmark[1] == 0.0 && Landmark[2] == 0.0) {
				cout << "Landmark NULL!!" << endl;
				return;
			}
			b1(3 * i) = 1.0*double(Landmark[0]);
			b1(3 * i + 1) = 1.0*double(Landmark[1]);
			b1(3 * i + 2) = 1.0*double(Landmark[2]);
		}

		/***************Populate matrix from neighboors of the vertices***************************/
		RefMesh->PopulateMatrix(&tripletList, &b1, 3 * 43);

		A.setFromTriplets(tripletList.begin(), tripletList.end());

		SpMat MatA(nb_columns, nb_columns);
		Eigen::VectorXd b(nb_columns);
		MatA = A.transpose() * A;
		b = A.transpose() * b1;

		Eigen::SimplicialCholesky<SpMat> chol(MatA);  // performs a Cholesky factorization of A
		Eigen::VectorXd xres = chol.solve(b);    // use the factorization to solve for the given right hand side

		RefMesh->AffectToTVectorT(&xres);

		iter++;
	}

	/***********************Transfer expression deformation******************************/
	Eigen::VectorXd boV(3 * RefMesh->size());
	RefMesh->Deform(&boV);

	int indxMesh = 0;
	for (vector<MyMesh *>::iterator itMesh = Blendshape.begin() + 1; itMesh != Blendshape.end(); itMesh++) {

		Eigen::SimplicialCholesky<SpMat> chol(_MatList1[indxMesh]);  // performs a Cholesky factorization of A
		Eigen::VectorXd b = chol.solve(_MatList2[indxMesh] * boV);

		(*itMesh)->AffectToTVector(&b);

		indxMesh++;
	}
}

void FaceCap::ComputeAffineTransfo(vector<MyMesh *> Blendshape) {
	MyMesh * RefMesh = Blendshape[0];

	// Inititialisation
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		vector<Eigen::Matrix3f> TmpList;
		for (int i = 0; i < Blendshape.size() - 1; i++) {
			Eigen::Matrix3f A;
			TmpList.push_back(A);
		}
		_TransfoExpression.push_back(TmpList);
	}

	vector<SpMat> MatList;
	MyMesh * bi;
	float nmle[3];
	float summit4[3];
	float nmlei[3];
	float summit4i[3];
	Eigen::Matrix3f So;
	Eigen::Matrix3f Si;
	Eigen::Matrix3f Scurr;
	int indxFace = 0;

	Face *CurrFace;
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		CurrFace = *it;
		// Compute normal and weight of the face
		Utility::GetWeightedNormal(RefMesh, CurrFace, nmle);

		//Compute tetrahedron for b0
		summit4[0] = (RefMesh->_vertices[CurrFace->_v1]->_x + RefMesh->_vertices[CurrFace->_v2]->_x + RefMesh->_vertices[CurrFace->_v3]->_x) / 3.0 + nmle[0];
		summit4[1] = (RefMesh->_vertices[CurrFace->_v1]->_y + RefMesh->_vertices[CurrFace->_v2]->_y + RefMesh->_vertices[CurrFace->_v3]->_y) / 3.0 + nmle[1];
		summit4[2] = (RefMesh->_vertices[CurrFace->_v1]->_z + RefMesh->_vertices[CurrFace->_v2]->_z + RefMesh->_vertices[CurrFace->_v3]->_z) / 3.0 + nmle[2];

		So(0, 0) = (RefMesh->_vertices[CurrFace->_v2]->_x - RefMesh->_vertices[CurrFace->_v1]->_x);	 So(0, 1) = (RefMesh->_vertices[CurrFace->_v3]->_x - RefMesh->_vertices[CurrFace->_v1]->_x);		So(0, 2) = (summit4[0] - RefMesh->_vertices[CurrFace->_v1]->_x);
		So(1, 0) = (RefMesh->_vertices[CurrFace->_v2]->_y - RefMesh->_vertices[CurrFace->_v1]->_y);	 So(1, 1) = (RefMesh->_vertices[CurrFace->_v3]->_y - RefMesh->_vertices[CurrFace->_v1]->_y);		So(1, 2) = (summit4[1] - RefMesh->_vertices[CurrFace->_v1]->_y);
		So(2, 0) = (RefMesh->_vertices[CurrFace->_v2]->_z - RefMesh->_vertices[CurrFace->_v1]->_z);	 So(2, 1) = (RefMesh->_vertices[CurrFace->_v3]->_z - RefMesh->_vertices[CurrFace->_v1]->_z);		So(2, 2) = (summit4[2] - RefMesh->_vertices[CurrFace->_v1]->_z);

		// Go through all other blendshapes
		int indxMesh = 0;
		for (vector<MyMesh *>::iterator itMesh = Blendshape.begin() + 1; itMesh != Blendshape.end(); itMesh++) {
			bi = (*itMesh);
			// Compute normal and weight of the face
			Utility::GetWeightedNormal(bi, CurrFace, nmlei);

			//Compute tetrahedron for b0
			summit4i[0] = (bi->_vertices[CurrFace->_v1]->_x + bi->_vertices[CurrFace->_v2]->_x + bi->_vertices[CurrFace->_v3]->_x) / 3.0 + nmlei[0];
			summit4i[1] = (bi->_vertices[CurrFace->_v1]->_y + bi->_vertices[CurrFace->_v2]->_y + bi->_vertices[CurrFace->_v3]->_y) / 3.0 + nmlei[1];
			summit4i[2] = (bi->_vertices[CurrFace->_v1]->_z + bi->_vertices[CurrFace->_v2]->_z + bi->_vertices[CurrFace->_v3]->_z) / 3.0 + nmlei[2];

			Si(0, 0) = (bi->_vertices[CurrFace->_v2]->_x - bi->_vertices[CurrFace->_v1]->_x);	 Si(0, 1) = (bi->_vertices[CurrFace->_v3]->_x - bi->_vertices[CurrFace->_v1]->_x);		Si(0, 2) = (summit4i[0] - bi->_vertices[CurrFace->_v1]->_x);
			Si(1, 0) = (bi->_vertices[CurrFace->_v2]->_y - bi->_vertices[CurrFace->_v1]->_y);	 Si(1, 1) = (bi->_vertices[CurrFace->_v3]->_y - bi->_vertices[CurrFace->_v1]->_y);		Si(1, 2) = (summit4i[1] - bi->_vertices[CurrFace->_v1]->_y);
			Si(2, 0) = (bi->_vertices[CurrFace->_v2]->_z - bi->_vertices[CurrFace->_v1]->_z);	 Si(2, 1) = (bi->_vertices[CurrFace->_v3]->_z - bi->_vertices[CurrFace->_v1]->_z);		Si(2, 2) = (summit4i[2] - bi->_vertices[CurrFace->_v1]->_z);

			_TransfoExpression[indxFace][indxMesh] = Si * So.inverse();

			indxMesh++;
		}
		indxFace++;
	}

	// Compute Matrix F that fix points on the backface.

	float nu = 100.0;
	SpMat F(3 * RefMesh->_vertices.size(), 3 * RefMesh->_vertices.size());
	vector<TrplType> tripletListF;
	int indx = 0;
	for (vector<MyPoint *>::iterator it = RefMesh->_vertices.begin(); it != RefMesh->_vertices.end(); it++) {
		if ((*it)->_BackPoint) {
			tripletListF.push_back(TrplType(3 * indx, 3 * indx, 1.0));
			tripletListF.push_back(TrplType(3 * indx + 1, 3 * indx + 1, 1.0));
			tripletListF.push_back(TrplType(3 * indx + 2, 3 * indx + 2, 1.0));
		}

		indx++;
	}
	F.setFromTriplets(tripletListF.begin(), tripletListF.end());

	// Compute Matrix G that transform vertices to edges.
	SpMat G(6 * RefMesh->_triangles.size(), 3 * RefMesh->_vertices.size());
	vector<TrplType> tripletList;
	indx = 0;
	for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
		CurrFace = *it;
		tripletList.push_back(TrplType(6 * indx, 3 * CurrFace->_v1, -1.0));
		tripletList.push_back(TrplType(6 * indx + 1, 3 * CurrFace->_v1 + 1, -1.0));
		tripletList.push_back(TrplType(6 * indx + 2, 3 * CurrFace->_v1 + 2, -1.0));
		tripletList.push_back(TrplType(6 * indx + 3, 3 * CurrFace->_v1, -1.0));
		tripletList.push_back(TrplType(6 * indx + 4, 3 * CurrFace->_v1 + 1, -1.0));
		tripletList.push_back(TrplType(6 * indx + 5, 3 * CurrFace->_v1 + 2, -1.0));


		tripletList.push_back(TrplType(6 * indx, 3 * CurrFace->_v2, 1.0));
		tripletList.push_back(TrplType(6 * indx + 1, 3 * CurrFace->_v2 + 1, 1.0));
		tripletList.push_back(TrplType(6 * indx + 2, 3 * CurrFace->_v2 + 2, 1.0));
		tripletList.push_back(TrplType(6 * indx + 3, 3 * CurrFace->_v3, 1.0));
		tripletList.push_back(TrplType(6 * indx + 4, 3 * CurrFace->_v3 + 1, 1.0));
		tripletList.push_back(TrplType(6 * indx + 5, 3 * CurrFace->_v3 + 2, 1.0));

		indx++;
	}

	G.setFromTriplets(tripletList.begin(), tripletList.end());

	SpMat MatT(3 * RefMesh->_vertices.size(), 3 * RefMesh->_vertices.size());
	SpMat MatTmp(3 * RefMesh->_vertices.size(), 3 * RefMesh->_vertices.size());
	Eigen::VectorXd b(3 * RefMesh->_vertices.size());
	int indxMesh = 0;
	for (vector<MyMesh *>::iterator itMesh = Blendshape.begin() + 1; itMesh != Blendshape.end(); itMesh++) {
		//Compute the affine transfo matrix
		SpMat H(6 * RefMesh->_triangles.size(), 6 * RefMesh->_triangles.size());
		vector<TrplType> tripletListH;
		indx = 0;
		for (vector<Face *>::iterator it = RefMesh->_triangles.begin(); it != RefMesh->_triangles.end(); it++) {
			tripletListH.push_back(TrplType(6 * indx, 6 * indx, _TransfoExpression[indx][indxMesh](0, 0)));		tripletListH.push_back(TrplType(6 * indx, 6 * indx + 1, _TransfoExpression[indx][indxMesh](0, 1)));		tripletListH.push_back(TrplType(6 * indx, 6 * indx + 2, _TransfoExpression[indx][indxMesh](0, 2)));
			tripletListH.push_back(TrplType(6 * indx + 1, 6 * indx, _TransfoExpression[indx][indxMesh](1, 0)));	tripletListH.push_back(TrplType(6 * indx + 1, 6 * indx + 1, _TransfoExpression[indx][indxMesh](1, 1)));	tripletListH.push_back(TrplType(6 * indx + 1, 6 * indx + 2, _TransfoExpression[indx][indxMesh](1, 2)));
			tripletListH.push_back(TrplType(6 * indx + 2, 6 * indx, _TransfoExpression[indx][indxMesh](2, 0)));	tripletListH.push_back(TrplType(6 * indx + 2, 6 * indx + 1, _TransfoExpression[indx][indxMesh](2, 1)));	tripletListH.push_back(TrplType(6 * indx + 2, 6 * indx + 2, _TransfoExpression[indx][indxMesh](2, 2)));

			tripletListH.push_back(TrplType(6 * indx + 3, 6 * indx + 3, _TransfoExpression[indx][indxMesh](0, 0)));	tripletListH.push_back(TrplType(6 * indx + 3, 6 * indx + 4, _TransfoExpression[indx][indxMesh](0, 1)));	tripletListH.push_back(TrplType(6 * indx + 3, 6 * indx + 5, _TransfoExpression[indx][indxMesh](0, 2)));
			tripletListH.push_back(TrplType(6 * indx + 4, 6 * indx + 3, _TransfoExpression[indx][indxMesh](1, 0)));	tripletListH.push_back(TrplType(6 * indx + 4, 6 * indx + 4, _TransfoExpression[indx][indxMesh](1, 1)));	tripletListH.push_back(TrplType(6 * indx + 4, 6 * indx + 5, _TransfoExpression[indx][indxMesh](1, 2)));
			tripletListH.push_back(TrplType(6 * indx + 5, 6 * indx + 3, _TransfoExpression[indx][indxMesh](2, 0)));	tripletListH.push_back(TrplType(6 * indx + 5, 6 * indx + 4, _TransfoExpression[indx][indxMesh](2, 1)));	tripletListH.push_back(TrplType(6 * indx + 5, 6 * indx + 5, _TransfoExpression[indx][indxMesh](2, 2)));

			indx++;
		}

		H.setFromTriplets(tripletListH.begin(), tripletListH.end());

		MatTmp = (G.transpose()*G + nu*F);
		MatT = (G.transpose()*H*G + nu*F);
		_MatList1.push_back(MatTmp);
		_MatList2.push_back(MatT);
		indxMesh++;
	}
}

void FaceCap::MedianFilter(int x, int y, int width, int height) {

	float pix[4];
	float color_out[4];
	float pixel_out[4];
	int size = 1;
	int ll, ul, lr, ur;
	vector<float> tab;
	//float tab[9];
	int count, q;
	float avg, avgR, avgG, avgB, weight;
	float pixel_curr[4], color_curr[4];

	for (int i = y; i < height; i++) {
		for (int j = x; j < width; j++) {
			pixel_out[0] = _Bump.at<cv::Vec4f>(i, j)[0];
			pixel_out[1] = _Bump.at<cv::Vec4f>(i, j)[1];
			pixel_out[2] = _Bump.at<cv::Vec4f>(i, j)[2];
			pixel_out[3] = _Bump.at<cv::Vec4f>(i, j)[3];

			color_out[0] = _RGBMapBump.at<cv::Vec4f>(i, j)[0];
			color_out[1] = _RGBMapBump.at<cv::Vec4f>(i, j)[1];
			color_out[2] = _RGBMapBump.at<cv::Vec4f>(i, j)[2];
			color_out[3] = _RGBMapBump.at<cv::Vec4f>(i, j)[3];

			ll = max(0, (int)i - size);
			ul = min(height, (int)i + size + 1);
			lr = max(0, (int)j - size);
			ur = min(width, (int)j + size + 1);

			count = 0;
			q = 0;
			avg = 0.0f;
			avgR = 0.0f;
			avgG = 0.0f;
			avgB = 0.0f;
			weight = 0.0f;
			tab.clear();
			
			for (int k = ll; k < ul; k++) {
				for (int l = lr; l < ur; l++) {
					pixel_curr[0] = _Bump.at<cv::Vec4f>(k, l)[0];
					pixel_curr[1] = _Bump.at<cv::Vec4f>(k, l)[1];

					color_curr[0] = _RGBMapBump.at<cv::Vec4f>(k, l)[0];
					color_curr[1] = _RGBMapBump.at<cv::Vec4f>(k, l)[1];
					color_curr[2] = _RGBMapBump.at<cv::Vec4f>(k, l)[2];

					if (pixel_curr[1] > 0.0f) {
						tab.push_back(pixel_curr[0]);
						/*q = 0;
						while (q < count && tab[q] > pixel_curr[0])
							q++;
						for (int r = count; r > q; r--)
							tab[r] = tab[r - 1];
						tab[q] = pixel_curr[0];*/
						avg += pixel_curr[0];
						avgR += color_curr[0];
						avgG += color_curr[1];
						avgB += color_curr[2];
						//weight += 1.0f;
						//count++;
					}

				}
			}

			sort(tab.begin(), tab.end());
			count = tab.size();
			weight = float(count);
			if (pixel_out[3] == -1.0f && pixel_out[1] == 0.0f) {
				if (weight > 0.0f) {
					pixel_out[0] = avg / weight;
					pixel_out[1] = 1.0f;
					color_out[0] = avgR / weight;
					color_out[1] = avgG / weight;
					color_out[2] = avgB / weight;
					_RGBMapBump.at<cv::Vec4f>(i, j)[0] = avgR / weight;
					_RGBMapBump.at<cv::Vec4f>(i, j)[1] = avgG / weight;
					_RGBMapBump.at<cv::Vec4f>(i, j)[2] = avgB / weight;
				}
			}
			else {
				if (count > 0)
					pixel_out[0]= tab[count / 2];
			}

			_Bump.at<cv::Vec4f>(i, j)[0] = pixel_out[0];
			_Bump.at<cv::Vec4f>(i, j)[1] = pixel_out[1];
		}
	}
}

void FaceCap::GenerateBump(vector<MyMesh *> Blendshape, int x, int y, int width, int height) {
	float bumpIn[4];  // (bump, mask, label, 0)
	float RGBIn[4];
	float nmle[3] = { 0.0f, 0.0f, 0.0f };
	float pt[3] = { 0.0f, 0.0f, 0.0f };
	int p_indx[2];
	float pt_T[3];
	float nmle_T[3];
	float fact_BP = 1000.0f;
	float bum_val, d, maskIn;
	float VMapInn[4];
	float s[2];
	float s1[2];
	float s2[2];
	float ptIn[4]; // (x,y,z, flag)
	float NMapIn[4];
	float Tnmle[3];
	float min_dist, best_state, fact_curr, length, thresh_dist;
	float pos[3];
	float dir[2];
	float range = 50.0f;
	float nmleIn[4];
	unsigned int flagIn[4];
	int k, l;
	int size = 1, ll, ul, lr, ur;
	float u_vect[3];
	float proj, dist, dist_to_nmle, dist_angle;
	float v_vect[3];
	float weight, new_bump;
	float p1[3];
	unsigned char pixelRGB[4];
	float RGBMapIn[4];

	//unsigned char labelIn[4];
	for (int i = y; i < height; i++) {
		for (int j = x; j < width; j++) {
			int tid = i*BumpWidth + j;
			bumpIn[0] = _Bump.at<cv::Vec4f>(i, j)[0];
			bumpIn[1] = _Bump.at<cv::Vec4f>(i, j)[1];
			bumpIn[2] = _Bump.at<cv::Vec4f>(i, j)[2];
			bumpIn[3] = _Bump.at<cv::Vec4f>(i, j)[3];

			_FaceSegment.at<cv::Vec4b>(i, j)[0] = 255;
			_FaceSegment.at<cv::Vec4b>(i, j)[1] = 0;
			_FaceSegment.at<cv::Vec4b>(i, j)[2] = 0;
			_FaceSegment.at<cv::Vec4b>(i, j)[3] = 0;

			if (bumpIn[2] == -1.0f) {
				_VMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[3] = 0.0f;

				_Bump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_Bump.at<cv::Vec4f>(i, j)[1] = 0.0f;
				_Bump.at<cv::Vec4f>(i, j)[2] = 0.0f;

				_RGBMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_RGBMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
				_RGBMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
				continue;
			}

			if (bumpIn[3] == -1.0f && bumpIn[1] == 0.0f) {
				_VMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
				continue;
			}

			RGBIn[0] = _RGBMapBump.at<cv::Vec4f>(i, j)[0];
			RGBIn[1] = _RGBMapBump.at<cv::Vec4f>(i, j)[1];
			RGBIn[2] = _RGBMapBump.at<cv::Vec4f>(i, j)[2];

			/*labelIn[0] = _LabelsMask.at<cv::Vec4b>(i, j)[0];
			labelIn[1] = _LabelsMask.at<cv::Vec4b>(i, j)[1];
			labelIn[2] = _LabelsMask.at<cv::Vec4b>(i, j)[2];
			labelIn[0] = labelIn[0] / 255;
			labelIn[1] = labelIn[0] / 255;
			labelIn[2] = labelIn[2] / 255;
			int flag = labelIn[2] + labelIn[1] * (2 + 2 * labelIn[2]) + labelIn[0] * (3 + labelIn[1]);*/

			pt[0] = _VerticesBS[6 * tid];
			pt[1] = _VerticesBS[6 * tid + 1];
			pt[2] = _VerticesBS[6 * tid + 2];
			nmle[0] = _VerticesBS[6 * tid + 3];
			nmle[1] = _VerticesBS[6 * tid + 4];
			nmle[2] = _VerticesBS[6 * tid + 5];

			for (int k_BS = 1; k_BS < NB_BS; k_BS++) {
				// This blended normal is not really a normal since it may be not normalized
				nmle[0] = nmle[0] + _VerticesBS[k_BS * 6 * BumpHeight*BumpWidth + 6 * tid + 3] * _BlendshapeCoeff[k_BS];
				nmle[1] = nmle[1] + _VerticesBS[k_BS * 6 * BumpHeight*BumpWidth + 6 * tid + 4] * _BlendshapeCoeff[k_BS];
				nmle[2] = nmle[2] + _VerticesBS[k_BS * 6 * BumpHeight*BumpWidth + 6 * tid + 5] * _BlendshapeCoeff[k_BS];

				pt[0] = pt[0] + _VerticesBS[k_BS * 6 * BumpHeight*BumpWidth + 6 * tid] * _BlendshapeCoeff[k_BS];
				pt[1] = pt[1] + _VerticesBS[k_BS * 6 * BumpHeight*BumpWidth + 6 * tid + 1] * _BlendshapeCoeff[k_BS];
				pt[2] = pt[2] + _VerticesBS[k_BS * 6 * BumpHeight*BumpWidth + 6 * tid + 2] * _BlendshapeCoeff[k_BS];
			}

			pt_T[0] = pt[0] * _Pose[0] + pt[1] * _Pose[4] + pt[2] * _Pose[8] + _Pose[12];
			pt_T[1] = pt[0] * _Pose[1] + pt[1] * _Pose[5] + pt[2] * _Pose[9] + _Pose[13];
			pt_T[2] = pt[0] * _Pose[2] + pt[1] * _Pose[6] + pt[2] * _Pose[10] + _Pose[14];

			nmle_T[0] = nmle[0] * _Pose[0] + nmle[1] * _Pose[4] + nmle[2] * _Pose[8];
			nmle_T[1] = nmle[0] * _Pose[1] + nmle[1] * _Pose[5] + nmle[2] * _Pose[9];
			nmle_T[2] = nmle[0] * _Pose[2] + nmle[1] * _Pose[6] + nmle[2] * _Pose[10];

			bum_val = bumpIn[0];
			d = bum_val / fact_BP;
			maskIn = bumpIn[1];

			VMapInn[0] = pt[0] + d*nmle[0];
			VMapInn[1] = pt[1] + d*nmle[1];
			VMapInn[2] = pt[2] + d*nmle[2];
/*
			if (bumpIn[3] == -1.0f && bumpIn[1] == 0.0f) {
				_VMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
				continue;
			}*/

			if (nmle_T[2] > 0.0f || maskIn == 30.0f) {
				if (maskIn > 0.0f) {
					_VMapBump.at<cv::Vec4f>(i, j)[0] = VMapInn[0];
					_VMapBump.at<cv::Vec4f>(i, j)[1] = VMapInn[1];
					_VMapBump.at<cv::Vec4f>(i, j)[2] = VMapInn[2];
				}
				else {
					_VMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
					_VMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
					_VMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
				}
				continue;
			}

			// Project onto VMap and test visibility
			s[0] = float(min(cDepthWidth - 1, max(0, int(round(((pt_T[0] + d*nmle_T[0]) / fabs(pt_T[2] + d*nmle_T[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2])))));
			s[1] = float(min(cDepthHeight - 1, max(0, int(round(((pt_T[1] + d*nmle_T[1]) / fabs(pt_T[2] + d*nmle_T[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3])))));

			ptIn[0] = _VMap.at<cv::Vec4f>(int(s[1]), int(s[0]))[0];
			ptIn[1] = _VMap.at<cv::Vec4f>(int(s[1]), int(s[0]))[1];
			ptIn[2] = _VMap.at<cv::Vec4f>(int(s[1]), int(s[0]))[2];
			ptIn[3] = _VMap.at<cv::Vec4f>(int(s[1]), int(s[0]))[3];

			/*if (ptIn[3] > 0.0f) { // Invisible point at projection
				_FaceSegment.at<cv::Vec4b>(i, j)[0] = 0;
				_FaceSegment.at<cv::Vec4b>(i, j)[1] = 0;
				_FaceSegment.at<cv::Vec4b>(i, j)[2] = 0;
				_FaceSegment.at<cv::Vec4b>(i, j)[3] = 0;
				//write_imagef(VMapBumpD, coords, VMapInn);
				continue;
			}*/

			NMapIn[0] = _NMapBump.at<cv::Vec4f>(i, j)[0];
			NMapIn[1] = _NMapBump.at<cv::Vec4f>(i, j)[1];
			NMapIn[2] = _NMapBump.at<cv::Vec4f>(i, j)[2];
			NMapIn[3] = _NMapBump.at<cv::Vec4f>(i, j)[3];
			
			Tnmle[0] = NMapIn[0] * _Pose[0] + NMapIn[1] * _Pose[4] + NMapIn[2] * _Pose[8];
			Tnmle[1] = NMapIn[0] * _Pose[1] + NMapIn[1] * _Pose[5] + NMapIn[2] * _Pose[9];
			Tnmle[2] = NMapIn[0] * _Pose[2] + NMapIn[1] * _Pose[6] + NMapIn[2] * _Pose[10];
			if (Tnmle[0] == 0.0f && Tnmle[1] == 0.0f && Tnmle[2] == 0.0f) {
				Tnmle[0] = nmle_T[0];
				Tnmle[1] = nmle_T[1];
				Tnmle[2] = nmle_T[2];
			}

			min_dist = 1000000000.0f;
			best_state = -1.0f;
			fact_curr = round(maskIn) == 0.0f ? 1.0f : min(5.0f, round(maskIn) + 1.0f);

			//summit 1
			d = (bum_val - (range / fact_curr)) / fact_BP;
			pos[0] = pt_T[0] + d*nmle_T[0];
			pos[1] = pt_T[1] + d*nmle_T[1];
			pos[2] = pt_T[2] + d*nmle_T[2];

			// Project the point onto the depth image
			s1[0] = float(min(cDepthWidth - 1, max(0, int(round((pos[0] / fabs(pos[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2])))));
			s1[1] = float(min(cDepthHeight - 1, max(0, int(round((pos[1] / fabs(pos[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3])))));

			//summit 2
			d = (bum_val + (range / fact_curr)) / fact_BP;
			pos[0] = pt_T[0] + d*nmle_T[0];
			pos[1] = pt_T[1] + d*nmle_T[1];
			pos[2] = pt_T[2] + d*nmle_T[2];

			// Project the point onto the depth image
			s2[0] = float(min(cDepthWidth - 1, max(0, int(round((pos[0] / fabs(pos[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2])))));
			s2[1] = float(min(cDepthHeight - 1, max(0, int(round((pos[1] / fabs(pos[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3])))));

			length = sqrt((s1[0] - s2[0])*(s1[0] - s2[0]) + (s1[1] - s2[1])*(s1[1] - s2[1]));

			dir[0] = (s2[0] - s1[0]) / length;
			dir[1] = (s2[1] - s1[1]) / length;

			d = bum_val / fact_BP;
			pos[0] = pt_T[0] + d*nmle_T[0];
			pos[1] = pt_T[1] + d*nmle_T[1];
			pos[2] = pt_T[2] + d*nmle_T[2];


			thresh_dist = round(maskIn) == 0.0f ? 0.06f : 0.01f;

			for (float lambda = 0.0f; lambda <= length; lambda += 1.0f) {
				k = int(round(s1[1] + lambda*dir[1]));
				l = int(round(s1[0] + lambda*dir[0]));

				if (k < 0 || k > cDepthHeight - 1 || l < 0 || l > cDepthWidth - 1)
					continue;

				ll = max(0, (int)k - size);
				ul = min(cDepthHeight, (int)k + size + 1);
				lr = max(0, (int)l - size);
				ur = min(cDepthWidth, (int)l + size + 1);

				for (int kk = ll; kk < ul; kk++) {
					for (int lk = lr; lk < ur; lk++) {
						ptIn[0] = _VMap.at<cv::Vec4f>(kk, lk)[0];
						ptIn[1] = _VMap.at<cv::Vec4f>(kk, lk)[1];
						ptIn[2] = _VMap.at<cv::Vec4f>(kk, lk)[2];
						ptIn[3] = _VMap.at<cv::Vec4f>(kk, lk)[3];

						nmleIn[0] = _NMap.at<cv::Vec4f>(kk, lk)[0];
						nmleIn[1] = _NMap.at<cv::Vec4f>(kk, lk)[1];
						nmleIn[2] = _NMap.at<cv::Vec4f>(kk, lk)[2];
						nmleIn[3] = _NMap.at<cv::Vec4f>(kk, lk)[3];

						if (nmleIn[0] == 0.0f && nmleIn[1] == 0.0f && nmleIn[2] == 0.0f)
							continue;

						//compute distance of point to the normal
						u_vect[0] = ptIn[0] - pt_T[0];
						u_vect[1] = ptIn[1] - pt_T[1];
						u_vect[2] = ptIn[2] - pt_T[2];

						proj = u_vect[0] * nmle_T[0] + u_vect[1] * nmle_T[1] + u_vect[2] * nmle_T[2];
						v_vect[0] = u_vect[0] - proj * nmle_T[0];
						v_vect[1] = u_vect[1] - proj * nmle_T[1];
						v_vect[2] = u_vect[2] - proj * nmle_T[2];
						dist = sqrt((ptIn[0] - pos[0]) * (ptIn[0] - pos[0]) + (ptIn[1] - pos[1]) * (ptIn[1] - pos[1]) + (ptIn[2] - pos[2]) * (ptIn[2] - pos[2]));
						dist_to_nmle = sqrt(v_vect[0] * v_vect[0] + v_vect[1] * v_vect[1] + v_vect[2] * v_vect[2]);
						dist_angle = Tnmle[0] * nmleIn[0] + Tnmle[1] * nmleIn[1] + Tnmle[2] * nmleIn[2];
						//bool valid = (flag == 0) || (flag == int(ptIn[3]));

						if (dist_to_nmle < min_dist && dist_angle > 0.7f && /*valid &&*/ dist < thresh_dist) {
							min_dist = dist_to_nmle;
							best_state = proj * fact_BP;
						}
					}
				}
			}

			VMapInn[0] = pt[0] + d*nmle[0];
			VMapInn[1] = pt[1] + d*nmle[1];
			VMapInn[2] = pt[2] + d*nmle[2];
			if (best_state == -1.0f || min_dist > 0.01f) {

				//Test for visibility violation
				p_indx[0] = min(cDepthWidth - 1, max(0, int(round((VMapInn[0] / fabs(VMapInn[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2]))));
				p_indx[1] = min(cDepthHeight - 1, max(0, int(round((VMapInn[1] / fabs(VMapInn[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3]))));
				ptIn[0] = _VMap.at<cv::Vec4f>(int(p_indx[1]), int(p_indx[0]))[0];
				ptIn[1] = _VMap.at<cv::Vec4f>(int(p_indx[1]), int(p_indx[0]))[1];
				ptIn[2] = _VMap.at<cv::Vec4f>(int(p_indx[1]), int(p_indx[0]))[2];
				ptIn[3] = _VMap.at<cv::Vec4f>(int(p_indx[1]), int(p_indx[0]))[3];

				if (ptIn[2] > (VMapInn[2] + 0.05f)) { //visibility violation
					if (maskIn < 1.0f) {
						_VMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
						_VMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
						_VMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
						_VMapBump.at<cv::Vec4f>(i, j)[3] = 0.0f;

						_Bump.at<cv::Vec4f>(i, j)[0] = 0.0f;
						_Bump.at<cv::Vec4f>(i, j)[1] = 0.0f;
						_Bump.at<cv::Vec4f>(i, j)[2] = bumpIn[2];
						_Bump.at<cv::Vec4f>(i, j)[3] = bumpIn[3];
					}
					else {
						_VMapBump.at<cv::Vec4f>(i, j)[0] = VMapInn[0];
						_VMapBump.at<cv::Vec4f>(i, j)[1] = VMapInn[1];
						_VMapBump.at<cv::Vec4f>(i, j)[2] = VMapInn[2];
						_VMapBump.at<cv::Vec4f>(i, j)[3] = VMapInn[3];
					
						_Bump.at<cv::Vec4f>(i, j)[0] = bumpIn[0];
						_Bump.at<cv::Vec4f>(i, j)[1] = maskIn - 1.0f;
						_Bump.at<cv::Vec4f>(i, j)[2] = bumpIn[2];
						_Bump.at<cv::Vec4f>(i, j)[3] = bumpIn[3];
					}
					continue;
				}

				if (maskIn > 0.0f) {
					_VMapBump.at<cv::Vec4f>(i, j)[0] = VMapInn[0];
					_VMapBump.at<cv::Vec4f>(i, j)[1] = VMapInn[1];
					_VMapBump.at<cv::Vec4f>(i, j)[2] = VMapInn[2];
					_VMapBump.at<cv::Vec4f>(i, j)[3] = VMapInn[3];
				}
				else {
					_VMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
					_VMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
					_VMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
					_VMapBump.at<cv::Vec4f>(i, j)[3] = 0.0f;
				}
				continue;
			}

			weight = 1.0f; //0.003f/(0.0001f + min_dist*min_dist); //maskIn == 0.0f ? 0.1f : Tnmle[2];
			//weight = weight*weight;
			new_bump = (weight*best_state + bum_val*maskIn) / (maskIn + weight);
			if (maskIn < 2000.0f) {
				_Bump.at<cv::Vec4f>(i, j)[0] = new_bump;


				// IMPORTANT PARAMETER
				// VALUE (SMALL) updating the shape
				// 
				_Bump.at<cv::Vec4f>(i, j)[1] = min(30.0f, maskIn + weight);	// 100
				_Bump.at<cv::Vec4f>(i, j)[2] = bumpIn[2];
				_Bump.at<cv::Vec4f>(i, j)[3] = 1.0f;
			}
			else {
				new_bump = bum_val;
			}

			//Get color
			d = new_bump / fact_BP;
			p1[0] = pt_T[0] + d*nmle_T[0];
			p1[1] = pt_T[1] + d*nmle_T[1];
			p1[2] = pt_T[2] + d*nmle_T[2];

			p_indx[0] = min(cDepthWidth - 1, max(0, int(round((p1[0] / fabs(p1[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2]))));
			p_indx[1] = min(cDepthHeight - 1, max(0, int(round((p1[1] / fabs(p1[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3]))));

			pixelRGB[0] = _RGBMap.at<cv::Vec4b>(int(p_indx[1]), int(p_indx[0]))[0];
			pixelRGB[1] = _RGBMap.at<cv::Vec4b>(int(p_indx[1]), int(p_indx[0]))[1];
			pixelRGB[2] = _RGBMap.at<cv::Vec4b>(int(p_indx[1]), int(p_indx[0]))[2];
			pixelRGB[3] = _RGBMap.at<cv::Vec4b>(int(p_indx[1]), int(p_indx[0]))[3];

			ptIn[0] = _VMap.at<cv::Vec4f>(int(p_indx[1]), int(p_indx[0]))[0];
			ptIn[1] = _VMap.at<cv::Vec4f>(int(p_indx[1]), int(p_indx[0]))[1];
			ptIn[2] = _VMap.at<cv::Vec4f>(int(p_indx[1]), int(p_indx[0]))[2];
			ptIn[3] = _VMap.at<cv::Vec4f>(int(p_indx[1]), int(p_indx[0]))[3];

			RGBMapIn[0] = _RGBMapBump.at<cv::Vec4f>(i, j)[0];
			RGBMapIn[1] = _RGBMapBump.at<cv::Vec4f>(i, j)[1];
			RGBMapIn[2] = _RGBMapBump.at<cv::Vec4f>(i, j)[2];
			RGBMapIn[3] = _RGBMapBump.at<cv::Vec4f>(i, j)[3];
			
			if ((ptIn[0] != 0.0f || ptIn[1] != 0.0f || ptIn[2] != 0.0f) && maskIn < 2000.0f) {
				_RGBMapBump.at<cv::Vec4f>(i, j)[0] = (weight*float(pixelRGB[2]) + RGBMapIn[0] * maskIn) / (maskIn + weight);
				_RGBMapBump.at<cv::Vec4f>(i, j)[1] = (weight*float(pixelRGB[1]) + RGBMapIn[1] * maskIn) / (maskIn + weight);
				_RGBMapBump.at<cv::Vec4f>(i, j)[2] = (weight*float(pixelRGB[0]) + RGBMapIn[2] * maskIn) / (maskIn + weight);
				_RGBMapBump.at<cv::Vec4f>(i, j)[3] = 1.0f;
			}

			VMapInn[0] = pt[0] + d*nmle[0];
			VMapInn[1] = pt[1] + d*nmle[1];
			VMapInn[2] = pt[2] + d*nmle[2];
			if (maskIn + weight > 0.0) {
				_VMapBump.at<cv::Vec4f>(i, j)[0] = VMapInn[0];
				_VMapBump.at<cv::Vec4f>(i, j)[1] = VMapInn[1];
				_VMapBump.at<cv::Vec4f>(i, j)[2] = VMapInn[2];
				_VMapBump.at<cv::Vec4f>(i, j)[3] = VMapInn[3];
			}
			else {
				_VMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
				_VMapBump.at<cv::Vec4f>(i, j)[3] = 0.0f;
			}
		}
	}

	/********Median Filter**********/
	MedianFilter(x, y, width, height);


	/**************Update VMap and NMap*******************/
	//cv::imshow("_RGBMapBump", _RGBMapBump);
	//cv::waitKey(1);
	int n_x = x == 0 ? x + 1 : x;
	int n_y = y == 0 ? y + 1 : y;
	int n_w = width == BumpWidth ? width - 1 : width;
	int n_h = height == BumpHeight ? height - 1 : height;
	ComputeNormalesApprox(n_x, n_y, n_w, n_h);
}

void FaceCap::ComputeNormalesOptimized(int x, int y, int width, int height) {
	// Compute Normal map
	cv::Mat kernel = cv::Mat::zeros(3, 3, CV_32F);
	kernel.at<float>(1, 0) = 1.0f;
	kernel.at<float>(1, 2) = -1.0f;
	filter2D(_VMapBump, _grad_x_bump, -1, kernel);
	kernel = cv::Mat::zeros(3, 3, CV_32F);
	kernel.at<float>(0, 1) = -1.0f;
	kernel.at<float>(2, 1) = 1.0f;
	filter2D(_VMapBump, _grad_y_bump, -1, kernel);

	std::vector<cv::Mat> channelsX(4);
	split(_grad_x_bump, channelsX);
	std::vector<cv::Mat> channelsY(4);
	split(_grad_y_bump, channelsY);

	cv::Mat xMat = channelsX[1].mul(channelsY[2]) - channelsX[2].mul(channelsY[1]);
	cv::Mat yMat = channelsX[2].mul(channelsY[0]) - channelsX[0].mul(channelsY[2]);
	cv::Mat zMat = channelsX[0].mul(channelsY[1]) - channelsX[1].mul(channelsY[0]);
	cv::Mat mag_f(xMat.mul(xMat) + yMat.mul(yMat) + zMat.mul(zMat));
	cv::sqrt(mag_f, mag_f);
	xMat = xMat.mul(1 / mag_f);
	yMat = yMat.mul(1 / mag_f);
	zMat = zMat.mul(1 / mag_f);

	std::vector<cv::Mat> matricesN = { xMat, yMat, zMat, _ones_bump };
	cv::merge(matricesN, _NMapBump);
}

void FaceCap::ComputeNormalesApprox(int x, int y, int width, int height) {
	float p1[4];
	float p2[4];
	float n_p[3];
	float norm_n;
	for (int i = y; i < height; i++) {
		for (int j = x; j < width; j++) {

			p1[0] = _VMapBump.at<cv::Vec4f>(i + 1, j)[0] - _VMapBump.at<cv::Vec4f>(i - 1, j)[0];
			p1[1] = _VMapBump.at<cv::Vec4f>(i + 1, j)[1] - _VMapBump.at<cv::Vec4f>(i - 1, j)[1];
			p1[2] = _VMapBump.at<cv::Vec4f>(i + 1, j)[2] - _VMapBump.at<cv::Vec4f>(i - 1, j)[2];

			p2[0] = _VMapBump.at<cv::Vec4f>(i, j + 1)[0] - _VMapBump.at<cv::Vec4f>(i, j - 1)[0];
			p2[1] = _VMapBump.at<cv::Vec4f>(i, j + 1)[1] - _VMapBump.at<cv::Vec4f>(i, j - 1)[1];
			p2[2] = _VMapBump.at<cv::Vec4f>(i, j + 1)[2] - _VMapBump.at<cv::Vec4f>(i, j - 1)[2];

			if (_VMapBump.at<cv::Vec4f>(i + 1, j)[2] != 0.0f && _VMapBump.at<cv::Vec4f>(i - 1, j)[2] != 0.0f &&
				_VMapBump.at<cv::Vec4f>(i, j + 1)[2] != 0.0f && _VMapBump.at<cv::Vec4f>(i, j - 1)[2] != 0.0f) {
				n_p[0] = p1[1]*p2[2] - p1[2]*p2[1];
				n_p[1] = p1[2]*p2[0] - p1[0]*p2[2];
				n_p[2] = p1[0]*p2[1] - p1[1]*p2[0];

				norm_n = (n_p[0] * n_p[0] + n_p[1] * n_p[1] + n_p[2] * n_p[2]);

				if (norm_n != 0.0f) {
					_NMapBump.at<cv::Vec4f>(i, j)[0] = n_p[0] / sqrt(norm_n);
					_NMapBump.at<cv::Vec4f>(i, j)[1] = n_p[1] / sqrt(norm_n);
					_NMapBump.at<cv::Vec4f>(i, j)[2] = n_p[2] / sqrt(norm_n);
					_NMapBump.at<cv::Vec4f>(i, j)[3] = 0.0f;
				}
				else {
					_NMapBump.at<cv::Vec4f>(i, j)[0] = 0.0f;
					_NMapBump.at<cv::Vec4f>(i, j)[1] = 0.0f;
					_NMapBump.at<cv::Vec4f>(i, j)[2] = 0.0f;
					_NMapBump.at<cv::Vec4f>(i, j)[3] = 0.0f;
				}
			}
		}
	}
}

void FaceCap::ComputeNormales(int x, int y, int width, int height) {
	float p1[4];
	float p2[4];
	float p3[4];
	float n_p[3];
	float n_p1[3];
	float n_p2[3];
	float n_p3[3];
	float n_p4[3];
	float norm_n;
	for (int i = y; i < height; i++) {
		for (int j = x; j < width; j++) {
			p1[0] = _VMapBump.at<cv::Vec4f>(i, j)[0];
			p1[1] = _VMapBump.at<cv::Vec4f>(i, j)[1];
			p1[2] = _VMapBump.at<cv::Vec4f>(i, j)[2];
			float NOut[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

			if (i < 1 || j < 1 || (p1[0] == 0.0f && p1[1] == 0.0f && p1[2] == 0.0f)) {
				_NMapBump.at<cv::Vec4f>(i, j)[0] = NOut[0];
				_NMapBump.at<cv::Vec4f>(i, j)[1] = NOut[1];
				_NMapBump.at<cv::Vec4f>(i, j)[2] = NOut[2];
				_NMapBump.at<cv::Vec4f>(i, j)[3] = NOut[3];
				continue;
			}

			unsigned short n_tot = 0;

			n_p1[0] = 0.0; n_p1[1] = 0.0; n_p1[2] = 0.0;
			n_p2[0] = 0.0; n_p2[1] = 0.0; n_p2[2] = 0.0;
			n_p3[0] = 0.0; n_p3[1] = 0.0; n_p3[2] = 0.0;
			n_p4[0] = 0.0; n_p4[1] = 0.0; n_p4[2] = 0.0;

			////////////////////////// Triangle 1 /////////////////////////////////
			p2[0] = _VMapBump.at<cv::Vec4f>(i + 1, j)[0];
			p2[1] = _VMapBump.at<cv::Vec4f>(i + 1, j)[1];
			p2[2] = _VMapBump.at<cv::Vec4f>(i + 1, j)[2];

			p3[0] = _VMapBump.at<cv::Vec4f>(i, j + 1)[0];
			p3[1] = _VMapBump.at<cv::Vec4f>(i, j + 1)[1];
			p3[2] = _VMapBump.at<cv::Vec4f>(i, j + 1)[2];

			if (p2[2] != 0.0f && p3[2] != 0.0f) {
				n_p1[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p1[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p1[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p1[0] * n_p1[0] + n_p1[1] * n_p1[1] + n_p1[2] * n_p1[2]);

				if (norm_n != 0.0f) {
					n_p1[0] = n_p1[0] / sqrt(norm_n);
					n_p1[1] = n_p1[1] / sqrt(norm_n);
					n_p1[2] = n_p1[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 2 /////////////////////////////////
			p2[0] = _VMapBump.at<cv::Vec4f>(i, j + 1)[0];
			p2[1] = _VMapBump.at<cv::Vec4f>(i, j + 1)[1];
			p2[2] = _VMapBump.at<cv::Vec4f>(i, j + 1)[2];

			p3[0] = _VMapBump.at<cv::Vec4f>(i - 1, j)[0];
			p3[1] = _VMapBump.at<cv::Vec4f>(i - 1, j)[1];
			p3[2] = _VMapBump.at<cv::Vec4f>(i - 1, j)[2];

			if (p2[2] != 0.0f && p3[2] != 0.0f) {
				n_p2[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p2[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p2[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p2[0] * n_p2[0] + n_p2[1] * n_p2[1] + n_p2[2] * n_p2[2]);

				if (norm_n != 0.0f) {
					n_p2[0] = n_p2[0] / sqrt(norm_n);
					n_p2[1] = n_p2[1] / sqrt(norm_n);
					n_p2[2] = n_p2[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 3 /////////////////////////////////
			p2[0] = _VMapBump.at<cv::Vec4f>(i - 1, j)[0];
			p2[1] = _VMapBump.at<cv::Vec4f>(i - 1, j)[1];
			p2[2] = _VMapBump.at<cv::Vec4f>(i - 1, j)[2];

			p3[0] = _VMapBump.at<cv::Vec4f>(i, j - 1)[0];
			p3[1] = _VMapBump.at<cv::Vec4f>(i, j - 1)[1];
			p3[2] = _VMapBump.at<cv::Vec4f>(i, j - 1)[2];

			if (p2[2] != 0.0f && p3[2] != 0.0f) {
				n_p3[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p3[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p3[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p3[0] * n_p3[0] + n_p3[1] * n_p3[1] + n_p3[2] * n_p3[2]);

				if (norm_n != 0.0f) {
					n_p3[0] = n_p3[0] / sqrt(norm_n);
					n_p3[1] = n_p3[1] / sqrt(norm_n);
					n_p3[2] = n_p3[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			////////////////////////// Triangle 4 /////////////////////////////////
			p2[0] = _VMapBump.at<cv::Vec4f>(i, j - 1)[0];
			p2[1] = _VMapBump.at<cv::Vec4f>(i, j - 1)[1];
			p2[2] = _VMapBump.at<cv::Vec4f>(i, j - 1)[2];

			p3[0] = _VMapBump.at<cv::Vec4f>(i + 1, j)[0];
			p3[1] = _VMapBump.at<cv::Vec4f>(i + 1, j)[1];
			p3[2] = _VMapBump.at<cv::Vec4f>(i + 1, j)[2];

			if (p2[2] != 0.0f && p3[2] != 0.0f) {
				n_p4[0] = (p2[1] - p1[1])*(p3[2] - p1[2]) - (p2[2] - p1[2])*(p3[1] - p1[1]);
				n_p4[1] = (p2[2] - p1[2])*(p3[0] - p1[0]) - (p2[0] - p1[0])*(p3[2] - p1[2]);
				n_p4[2] = (p2[0] - p1[0])*(p3[1] - p1[1]) - (p2[1] - p1[1])*(p3[0] - p1[0]);

				norm_n = (n_p4[0] * n_p4[0] + n_p4[1] * n_p4[1] + n_p4[2] * n_p4[2]);

				if (norm_n != 0.0f) {
					n_p4[0] = n_p4[0] / sqrt(norm_n);
					n_p4[1] = n_p4[1] / sqrt(norm_n);
					n_p4[2] = n_p4[2] / sqrt(norm_n);

					n_tot++;
				}
			}

			if (n_tot == 0) {
				_NMapBump.at<cv::Vec4f>(i, j)[0] = NOut[0];
				_NMapBump.at<cv::Vec4f>(i, j)[1] = NOut[1];
				_NMapBump.at<cv::Vec4f>(i, j)[2] = NOut[2];
				_NMapBump.at<cv::Vec4f>(i, j)[3] = NOut[3];
				continue;
			}

			n_p[0] = (n_p1[0] + n_p2[0] + n_p3[0] + n_p4[0]) / ((float)n_tot);
			n_p[1] = (n_p1[1] + n_p2[1] + n_p3[1] + n_p4[1]) / ((float)n_tot);
			n_p[2] = (n_p1[2] + n_p2[2] + n_p3[2] + n_p4[2]) / ((float)n_tot);

			norm_n = sqrt(n_p[0] * n_p[0] + n_p[1] * n_p[1] + n_p[2] * n_p[2]);

			if (norm_n != 0) {
				NOut[0] = n_p[0] / norm_n;
				NOut[1] = n_p[1] / norm_n;
				NOut[2] = n_p[2] / norm_n;
				_NMapBump.at<cv::Vec4f>(i, j)[0] = NOut[0];
				_NMapBump.at<cv::Vec4f>(i, j)[1] = NOut[1];
				_NMapBump.at<cv::Vec4f>(i, j)[2] = NOut[2];
				_NMapBump.at<cv::Vec4f>(i, j)[3] = NOut[3];
			}
			else {
				_NMapBump.at<cv::Vec4f>(i, j)[0] = NOut[0];
				_NMapBump.at<cv::Vec4f>(i, j)[1] = NOut[1];
				_NMapBump.at<cv::Vec4f>(i, j)[2] = NOut[2];
				_NMapBump.at<cv::Vec4f>(i, j)[3] = NOut[3];
			}
		}
	}
}

void FaceCap::ComputeLabels(MyMesh *TheMesh) {

	//for (int i = 0; i < 3818; i++) {
	//	TheMesh->_vertices[FrontIndices[i]]->_BackPoint = true;
		//cout << "i: " << i << " " << FrontIndices[i] << endl;
	//}

	cv::Mat img0(BumpHeight, BumpWidth, CV_16UC3);
	cv::Mat img1(BumpHeight, BumpWidth, CV_16UC3);
	float weights[3];
	Face *CurrFace;
	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			float lab_val = -1.0f;
			unsigned short iii = 0;
			for (vector<Face *>::iterator it = TheMesh->_triangles.begin(); it != TheMesh->_triangles.end(); it++) {
				if (Utility::IsInTriangle(TheMesh, (*it), i, j)) {
					lab_val = float(iii);
					break;
				}
				iii++;
			}

			if (lab_val > -1.0f) {
				CurrFace = TheMesh->_triangles[int(lab_val)];
				TextUV s1 = TextUV(TheMesh->_uvs[CurrFace->_t1]->_u*float(BumpHeight), TheMesh->_uvs[CurrFace->_t1]->_v*float(BumpWidth));
				TextUV s2 = TextUV(TheMesh->_uvs[CurrFace->_t2]->_u*float(BumpHeight), TheMesh->_uvs[CurrFace->_t2]->_v*float(BumpWidth));
				TextUV s3 = TextUV(TheMesh->_uvs[CurrFace->_t3]->_u*float(BumpHeight), TheMesh->_uvs[CurrFace->_t3]->_v*float(BumpWidth));
				Utility::getWeightsB(weights, float(i), float(j), s1, s2, s3);

				img0.at<cv::Vec3w>(i, j)[0] = unsigned short(65535.0f * weights[0] / (weights[0] + weights[1] + weights[2]));
				img0.at<cv::Vec3w>(i, j)[1] = unsigned short(65535.0f * weights[1] / (weights[0] + weights[1] + weights[2]));
				img0.at<cv::Vec3w>(i, j)[2] = unsigned short(65535.0f * weights[2] / (weights[0] + weights[1] + weights[2]));
				img1.at<cv::Vec3w>(i, j)[0] = unsigned short(lab_val + 1.0f);
				img1.at<cv::Vec3w>(i, j)[1] = 0;
				img1.at<cv::Vec3w>(i, j)[2] = 0;
				//bool ok = TheMesh->_vertices[CurrFace->_v1]->_BackPoint && TheMesh->_vertices[CurrFace->_v2]->_BackPoint && TheMesh->_vertices[CurrFace->_v3]->_BackPoint;
				//if (ok)
				//	img1.at<cv::Vec3w>(i, j)[0] = 65535;
			}
			else {
				img0.at<cv::Vec3w>(i, j)[0] = 0;
				img0.at<cv::Vec3w>(i, j)[1] = 0;
				img0.at<cv::Vec3w>(i, j)[2] = 0;
				img1.at<cv::Vec3w>(i, j)[0] = 0;
				img1.at<cv::Vec3w>(i, j)[1] = 0;
				img1.at<cv::Vec3w>(i, j)[2] = 0;
			}
		}
	}
	//cv::imshow("img0", img0);
	cv::imwrite("Weights-240.png", img0);
	cv::imwrite("Labels-240.png", img1);
	//cv::imwrite("Front.png", img1);
	cv::waitKey(1);

}

bool FaceCap::searchLandMark(int l_idx, float *n, float *d, float *s)
{
	// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev
	float ncurr[4];
	float ncurr_cp[3];
	float nprev[4];
	float vcurr[4];
	float vcurr_cp[3];
	float vprev[4];
	float distThres = 0.005f;
	float angleThres = 0.6f;

	int idx_i = _landmarksBump.at<int>(0, l_idx);
	int idx_j = _landmarksBump.at<int>(1, l_idx);

	ncurr[0] = _NMapBump.at<cv::Vec4f>(idx_i, idx_j)[0];
	ncurr[1] = _NMapBump.at<cv::Vec4f>(idx_i, idx_j)[1];
	ncurr[2] = _NMapBump.at<cv::Vec4f>(idx_i, idx_j)[2];

	if ((ncurr[0] == 0.0f && ncurr[1] == 0.0f && ncurr[2] == 0.0f) || _FaceSegment.at<cv::Vec3b>(idx_i, idx_j)[0] == 0)
		return false;

	vcurr[0] = _VMapBump.at<cv::Vec4f>(idx_i, idx_j)[0];
	vcurr[1] = _VMapBump.at<cv::Vec4f>(idx_i, idx_j)[1];
	vcurr[2] = _VMapBump.at<cv::Vec4f>(idx_i, idx_j)[2];
	vcurr[3] = _VMapBump.at<cv::Vec4f>(idx_i, idx_j)[3];

	vcurr_cp[0] = _Pose[0] * vcurr[0] + _Pose[4] * vcurr[1] + _Pose[8] * vcurr[2] + _Pose[12]; //Rcurr is row major
	vcurr_cp[1] = _Pose[1] * vcurr[0] + _Pose[5] * vcurr[1] + _Pose[9] * vcurr[2] + _Pose[13];
	vcurr_cp[2] = _Pose[2] * vcurr[0] + _Pose[6] * vcurr[1] + _Pose[10] * vcurr[2] + _Pose[14];

	ncurr_cp[0] = _Pose[0] * ncurr[0] + _Pose[4] * ncurr[1] + _Pose[8] * ncurr[2]; //Rcurr is row major
	ncurr_cp[1] = _Pose[1] * ncurr[0] + _Pose[5] * ncurr[1] + _Pose[9] * ncurr[2];
	ncurr_cp[2] = _Pose[2] * ncurr[0] + _Pose[6] * ncurr[1] + _Pose[10] * ncurr[2];

	int p_u = int(round(_landmarks.at<float>(0, l_idx)));
	int p_v = int(round(_landmarks.at<float>(1, l_idx)));

	if (p_u == 0 && p_v == 0)
		return false;

	nprev[0] = _NMap.at<cv::Vec4f>(p_v, p_u)[0];
	nprev[1] = _NMap.at<cv::Vec4f>(p_v, p_u)[1];
	nprev[2] = _NMap.at<cv::Vec4f>(p_v, p_u)[2];

	if (nprev[0] == 0.0f && nprev[1] == 0.0f && nprev[2] == 0.0f)
		return false;

	vprev[0] = _VMap.at<cv::Vec4f>(p_v, p_u)[0];
	vprev[1] = _VMap.at<cv::Vec4f>(p_v, p_u)[1];
	vprev[2] = _VMap.at<cv::Vec4f>(p_v, p_u)[2];
	vprev[3] = _VMap.at<cv::Vec4f>(p_v, p_u)[3];

	float dist = sqrt((vprev[0] - vcurr_cp[0])*(vprev[0] - vcurr_cp[0]) + (vprev[1] - vcurr_cp[1])*(vprev[1] - vcurr_cp[1]) + (vprev[2] - vcurr_cp[2])*(vprev[2] - vcurr_cp[2]));
	if (dist > distThres)
		return false;

	/*float angle = ncurr_cp[0] * nprev[0] + ncurr_cp[1] * nprev[1] + ncurr_cp[2] * nprev[2];
	if (angle < angleThres)
		return false;*/

	n[0] = ncurr_cp[0]; n[1] = ncurr_cp[1]; n[2] = ncurr_cp[2];
	d[0] = vprev[0]; d[1] = vprev[1]; d[2] = vprev[2];
	s[0] = vcurr_cp[0]; s[1] = vcurr_cp[1]; s[2] = vcurr_cp[2]; s[3] = vprev[2];

	return true;
}

bool FaceCap::searchGauss(int i, int j, float *n, float *d, float *s) {
	// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev

	float ncurr[3];
	float ncurr_cp[3];
	float nprev[3];
	float vcurr[4];
	float vcurr_cp[3];
	float vprev[4];
	int p_indx[2];
	float distThres = 0.05f;
	float angleThres = 0.76f;

	ncurr[0] = _NMapBump.at<cv::Vec4f>(i, j)[0];
	ncurr[1] = _NMapBump.at<cv::Vec4f>(i, j)[1];
	ncurr[2] = _NMapBump.at<cv::Vec4f>(i, j)[2];

	if (ncurr[0] == 0.0f && ncurr[1] == 0.0f && ncurr[2] == 0.0f)
		return false;

	vcurr[0] = _VMapBump.at<cv::Vec4f>(i, j)[0];
	vcurr[1] = _VMapBump.at<cv::Vec4f>(i, j)[1];
	vcurr[2] = _VMapBump.at<cv::Vec4f>(i, j)[2];
	vcurr[3] = _VMapBump.at<cv::Vec4f>(i, j)[3];

	vcurr_cp[0] = _Pose[0] * vcurr[0] + _Pose[4] * vcurr[1] + _Pose[8] * vcurr[2] + _Pose[12]; //Rcurr is row major
	vcurr_cp[1] = _Pose[1] * vcurr[0] + _Pose[5] * vcurr[1] + _Pose[9] * vcurr[2] + _Pose[13];
	vcurr_cp[2] = _Pose[2] * vcurr[0] + _Pose[6] * vcurr[1] + _Pose[10] * vcurr[2] + _Pose[14];

	ncurr_cp[0] = _Pose[0] * ncurr[0] + _Pose[4] * ncurr[1] + _Pose[8] * ncurr[2]; //Rcurr is row major
	ncurr_cp[1] = _Pose[1] * ncurr[0] + _Pose[5] * ncurr[1] + _Pose[9] * ncurr[2];
	ncurr_cp[2] = _Pose[2] * ncurr[0] + _Pose[6] * ncurr[1] + _Pose[10] * ncurr[2];
	
	p_indx[0] = min(cDepthWidth - 1, max(0, int(round((vcurr_cp[0] / fabs(vcurr_cp[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2]))));
	p_indx[1] = min(cDepthHeight - 1, max(0, int(round((vcurr_cp[1] / fabs(vcurr_cp[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3]))));

	nprev[0] = _NMap.at<cv::Vec4f>(p_indx[1], p_indx[0])[0];
	nprev[1] = _NMap.at<cv::Vec4f>(p_indx[1], p_indx[0])[1];
	nprev[2] = _NMap.at<cv::Vec4f>(p_indx[1], p_indx[0])[2];

	if (nprev[0] == 0.0f && nprev[1] == 0.0f && nprev[2] == 0.0f)
		return false;

	vprev[0] = _VMap.at<cv::Vec4f>(p_indx[1], p_indx[0])[0];
	vprev[1] = _VMap.at<cv::Vec4f>(p_indx[1], p_indx[0])[1];
	vprev[2] = _VMap.at<cv::Vec4f>(p_indx[1], p_indx[0])[2];
	vprev[3] = _VMap.at<cv::Vec4f>(p_indx[1], p_indx[0])[3];

	float dist = sqrt((vprev[0] - vcurr_cp[0])*(vprev[0] - vcurr_cp[0]) + (vprev[1] - vcurr_cp[1])*(vprev[1] - vcurr_cp[1]) + (vprev[2] - vcurr_cp[2])*(vprev[2] - vcurr_cp[2]));
	if (dist > distThres)
		return false;

	float angle = ncurr_cp[0] * nprev[0] + ncurr_cp[1] * nprev[1] + ncurr_cp[2] * nprev[2];
	if (angle < angleThres)
		return false;

	n[0] = ncurr_cp[0]; n[1] = ncurr_cp[1]; n[2] = ncurr_cp[2];
	d[0] = vprev[0]; d[1] = vprev[1]; d[2] = vprev[2];
	s[0] = vcurr_cp[0]; s[1] = vcurr_cp[1]; s[2] = vcurr_cp[2]; s[3] = vprev[2];
	s[4] = max(1.0f, vcurr[3] / 10.0f);

	return true;

}

void FaceCap::Register(vector<MyMesh *> Blendshape) {

	int iter = 0;
	bool converged = false;

	float sum, prev_sum, count;
	float lambda = 1.0;

	Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
	Eigen::Matrix<double, 6, 1> b;
	int shift = 0;
	double det;

	Eigen::Matrix<double, 6, 1> result;
	double q[4];
	double norm;

	double tmp[3][3];

	Eigen::Matrix3f Rinc;
	Eigen::Vector3f tinc;

	for (int lvl = _lvl - 1; lvl > -1; lvl--) {

		int fact = int(pow((float)2.0, (int)lvl));

		iter = 1;
		converged = iter > _max_iter[lvl];

		Eigen::Vector3f Translation_prev;
		Eigen::Matrix3f Rotation_prev;
		prev_sum = 0.0f;

		while (!converged) {
			/*** Compute correspondences ***/
			for (int l = 0; l < 29; l++)      
				_outbuffGICP[l] = 0.0;

			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					_Pose[4 * j + i] = float(_Rotation_inv(i, j));
				}
				_Pose[12 + i] = float(_Translation_inv(i));
			}

			_nbMatches = 0;

			/********************GICP Part****************************/
			float n[3], d[3], s[5];
			bool found_coresp = false;
			float weight = 1.0f;

			unsigned char visibility;

			///// Transform VMap and NMap with current pose


			for (int i = 0; i < BumpHeight; i+=fact) {
				for (int j = 0; j < BumpWidth; j+=fact) {
					visibility = _FaceSegment.at<cv::Vec4b>(i,j)[0];
					found_coresp = false;
					if (visibility == 255) {
						found_coresp = searchGauss(i, j, n, d, s);
						weight = s[4];
					}

					if (!found_coresp)
						continue;

					float row[7];
					float rowY[7];
					float rowZ[7];
					float JD[18];
					float JRot[18];
					float min_dist = 0.0;
					int indx_buff;
					int indx_buffY;
					int indx_buffZ;

					// row [0 -> 5] = A^t = [skew(s) | Id(3,3)]^t*n

					JD[0] = 1.0; JD[3] = 0.0; JD[6] = 0.0;	JD[9] = 0.0; JD[12] = d[2];	JD[15] = -d[1];
					JD[1] = 0.0; JD[4] = 1.0; JD[7] = 0.0;	JD[10] = -d[2]; JD[13] = 0.0; JD[16] = d[0];
					JD[2] = 0.0; JD[5] = 0.0; JD[8] = 1.0;	JD[11] = d[1];	JD[14] = -d[0]; JD[17] = 0.0;

					row[0] = weight*n[0];
					row[1] = weight*n[1];
					row[2] = weight*n[2];
					row[3] = weight*(-d[2] * n[1] + d[1] * n[2]);
					row[4] = weight*(d[2] * n[0] - d[0] * n[2]);
					row[5] = weight*(-d[1] * n[0] + d[0] * n[1]);

					row[6] = weight*(n[0] * (d[0] - s[0]) + n[1] * (d[1] - s[1]) + n[2] * (d[2] - s[2]));

					min_dist = sqrt((s[0] - d[0])*(s[0] - d[0]) + (s[1] - d[1])*(s[1] - d[1]) + (s[2] - d[2])*(s[2] - d[2]));

					_nbMatches++;

					shift = 0;
					for (int k = 0; k < 6; k++) {
						for (int l = k; l < 7; l++)          // cols + b
						{
							_outbuffGICP[shift] = _outbuffGICP[shift] + double(row[k] * row[l]);
							shift++;
						}
					}
					_outbuffGICP[27] = _outbuffGICP[27] + min_dist*min_dist;
				}
			}

			// Build correspondences for landmarks
			for (int i = 0; i < 43; i++) {
				found_coresp = false;
				found_coresp = searchLandMark(i, n, d, s);
				weight = 10.0f;

				if (!found_coresp)
					continue;

				float row[7];
				float rowY[7];
				float rowZ[7];
				float JD[18];
				float JRot[18];
				float min_dist = 0.0;
				int indx_buff;
				int indx_buffY;
				int indx_buffZ;

				// row [0 -> 5] = A^t = [skew(s) | Id(3,3)]^t*n

				JD[0] = 1.0; JD[3] = 0.0; JD[6] = 0.0;	JD[9] = 0.0;		JD[12] = d[2];	JD[15] = -d[1];
				JD[1] = 0.0; JD[4] = 1.0; JD[7] = 0.0;	JD[10] = -d[2]; JD[13] = 0.0;		JD[16] = d[0];
				JD[2] = 0.0; JD[5] = 0.0; JD[8] = 1.0;	JD[11] = d[1];	JD[14] = -d[0]; JD[17] = 0.0;

				// Landmark
				row[0] = weight*(JD[0]);
				row[1] = weight*(JD[3]);
				row[2] = weight*(JD[6]);
				row[3] = weight*(JD[9]);
				row[4] = weight*(JD[12]);
				row[5] = weight*(JD[15]);
				row[6] = -weight*(s[0] - d[0]);

				rowY[0] = weight*(JD[1]);
				rowY[1] = weight*(JD[4]);
				rowY[2] = weight*(JD[7]);
				rowY[3] = weight*(JD[10]);
				rowY[4] = weight*(JD[13]);
				rowY[5] = weight*(JD[16]);
				rowY[6] = -weight*(s[1] - d[1]);

				rowZ[0] = weight*(JD[2]);
				rowZ[1] = weight*(JD[5]);
				rowZ[2] = weight*(JD[8]);
				rowZ[3] = weight*(JD[11]);
				rowZ[4] = weight*(JD[14]);
				rowZ[5] = weight*(JD[17]);
				rowZ[6] = -weight*(s[2] - d[2]);

				shift = 0;
				for (int k = 0; k < 6; k++) {
					for (int l = k; l < 7; l++)          // cols + b
					{
						_outbuffGICP[shift] = _outbuffGICP[shift] + double(row[k] * row[l]) 
																	+ double(rowY[k] * rowY[l]) 
																	+ double(rowZ[k] * rowZ[l]);
						shift++;
					}
				}
				_outbuffGICP[27] = _outbuffGICP[27] + (s[0] - d[0])*(s[0] - d[0]) 
													+ (s[1] - d[1])*(s[1] - d[1]) 
													+ (s[2] - d[2])*(s[2] - d[2]);
			}

			/***********************End GICP part*************************/

			//cout << "NbMatches: " << _nbMatches << endl;
			if (_nbMatches < 100)
				break;

			sum = _outbuffGICP[27];
			sum = sum / float(_nbMatches);

			//cout << "Sum: " << sum << endl;

			shift = 0;
			for (int i = 0; i < 6; ++i) {  //rows
				for (int j = i; j < 7; ++j)    // cols + b
				{
					double value = double(_outbuffGICP[shift++]);
					if (j == 6)       // vector b
						b(i) = value;
					else
						A(j, i) = A(i, j) = value;
				}
				I(i, i) = A(i, i);
			}

			det = A.determinant();

			if (fabs(det) < 1e-15 || det != det)
			{
				if (det != det) std::cout << "qnan" << endl;
				std::cout << "det null" << endl;
				break;
			}

			if (prev_sum != 0.0f) {
				if (sum > prev_sum) {
					_Translation_inv = Translation_prev;
					_Rotation_inv = Rotation_prev;
					lambda = lambda / 1.5f;
					iter++;
					converged = iter > _max_iter[lvl];
					continue;
				}
				else {
					if (sum < prev_sum) {
						Translation_prev = _Translation_inv;
						Rotation_prev = _Rotation_inv;
						prev_sum = sum;
					}
				}
			}
			else {
				//prev_sum = sum;
				Translation_prev = _Translation_inv;
				Rotation_prev = _Rotation_inv;
			}

			Eigen::Matrix<double, 6, 1> delta_qsi = -A.inverse() * b;

			// Update transformation matrix
			Eigen::Matrix4d delta_transfo = Exponential(delta_qsi).inverse();

			Rinc(0, 0) = float(delta_transfo(0, 0)); Rinc(0, 1) = float(delta_transfo(0, 1)); Rinc(0, 2) = float(delta_transfo(0, 2));
			Rinc(1, 0) = float(delta_transfo(1, 0)); Rinc(1, 1) = float(delta_transfo(1, 1)); Rinc(1, 2) = float(delta_transfo(1, 2));
			Rinc(2, 0) = float(delta_transfo(2, 0)); Rinc(2, 1) = float(delta_transfo(2, 1)); Rinc(2, 2) = float(delta_transfo(2, 2));
			tinc(0) = float(delta_transfo(0, 3)); tinc(1) = float(delta_transfo(1, 3)); tinc(2) = float(delta_transfo(2, 3));

			_Translation_inv = Rinc * _Translation_inv + tinc;
			_Rotation_inv = Rinc * _Rotation_inv;
			iter++;

			if (iter > _max_iter[lvl] || ((Rinc - Eigen::Matrix3f::Identity()).norm() < 1.0e-6 && tinc.norm() < 1.0e-6)) {
				converged = true;
			}

		}
	}

	if (VERBOSE)
		std::cout << "Rot: " << _Rotation_inv << " Translation: " << _Translation_inv;
	_Rotation = _Rotation_inv.inverse();
	_Translation = -_Rotation * _Translation_inv;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Pose[4 * i + j] = _Rotation_inv(j, i);
		}
		_Pose[12 + i] = _Translation_inv(i);
	}

	_TranslationWindow.push_back(_Translation);
	_RotationWindow.push_back(_Rotation);

	if (_TranslationWindow.size() > 10) {
		_TranslationWindow.erase(_TranslationWindow.begin());
		_RotationWindow.erase(_RotationWindow.begin());
	}


	if (save_data) {
		ofstream  filestr;

		string filename = string(dest_name) + string("\\Animation\\Pose") + to_string(_idx) + string(".txt");
		filestr.open(filename, fstream::out);
		while (!filestr.is_open()) {
			cout << "Could not open MappingList" << endl;
			return;
		}

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				filestr << _Pose[4 * i + j] << endl;
			}
		}

		filestr.close();
	}
}

bool FaceCap::searchLandMark_PR(vector<MyMesh *> Blendshape, int l_idx, float *tmpX, float *tmpY, float *tmpZ) {
	// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev
	float w = 8.0f;
	float vprev[4];

	int p_u = int(round(_landmarks.at<float>(0, l_idx)));
	int p_v = int(round(_landmarks.at<float>(1, l_idx)));

	if ((p_u == 0 && p_v == 0) || p_u < 0 || p_v < 0)
		return false;

	vprev[0] = _VMap.at<cv::Vec4f>(p_v, p_u)[0];
	vprev[1] = _VMap.at<cv::Vec4f>(p_v, p_u)[1];
	vprev[2] = _VMap.at<cv::Vec4f>(p_v, p_u)[2];
	vprev[3] = _VMap.at<cv::Vec4f>(p_v, p_u)[3];

	if (vprev[3] == 0.0f)
		w = 2.0f;

	float pt_T[3] = { 0.0f, 0.0f, 0.0f };
	float pt[3] = { 0.0f, 0.0f, 0.0f };
	float nmle[3] = { 0.0f, 0.0f, 0.0f };
	float nmleTmp[3] = { 0.0f, 0.0f, 0.0f };

	MyPoint *LANDMARK_mesh = Blendshape[0]->Landmark(l_idx);
	pt_T[0] = LANDMARK_mesh->_x;
	pt_T[1] = LANDMARK_mesh->_y;
	pt_T[2] = LANDMARK_mesh->_z;

	for (int k = 1; k < NB_BS; k++) {
		MyPoint *LANDMARK_curr = Blendshape[k]->Landmark(l_idx);
		//(f((bj-bo) + d(nj-n0))
		tmpX[k - 1] = (LANDMARK_curr->_x - LANDMARK_mesh->_x) * _Pose[0] + (LANDMARK_curr->_y - LANDMARK_mesh->_y) * _Pose[4] + (LANDMARK_curr->_z - LANDMARK_mesh->_z) * _Pose[8];
		tmpY[k - 1] = (LANDMARK_curr->_x - LANDMARK_mesh->_x) * _Pose[1] + (LANDMARK_curr->_y - LANDMARK_mesh->_y) * _Pose[5] + (LANDMARK_curr->_z - LANDMARK_mesh->_z) * _Pose[9];
		tmpZ[k - 1] = (LANDMARK_curr->_x - LANDMARK_mesh->_x) * _Pose[2] + (LANDMARK_curr->_y - LANDMARK_mesh->_y) * _Pose[6] + (LANDMARK_curr->_z - LANDMARK_mesh->_z) * _Pose[10];

		pt_T[0] = pt_T[0] + _BlendshapeCoeff[k] * (LANDMARK_curr->_x - LANDMARK_mesh->_x);
		pt_T[1] = pt_T[1] + _BlendshapeCoeff[k] * (LANDMARK_curr->_y - LANDMARK_mesh->_y);
		pt_T[2] = pt_T[2] + _BlendshapeCoeff[k] * (LANDMARK_curr->_z - LANDMARK_mesh->_z);

		tmpX[k - 1] = w*(tmpX[k - 1]);
		tmpY[k - 1] = w*(tmpY[k - 1]);
		tmpZ[k - 1] = w*(tmpZ[k - 1]);

	}

	pt[0] = pt_T[0] * _Pose[0] + pt_T[1] * _Pose[4] + pt_T[2] * _Pose[8] + _Pose[12];
	pt[1] = pt_T[0] * _Pose[1] + pt_T[1] * _Pose[5] + pt_T[2] * _Pose[9] + _Pose[13];
	pt[2] = pt_T[0] * _Pose[2] + pt_T[1] * _Pose[6] + pt_T[2] * _Pose[10] + _Pose[14];

	float dist = sqrt((pt[0] - vprev[0])*(pt[0] - vprev[0]) + (pt[1] - vprev[1])*(pt[1] - vprev[1]) + (pt[2] - vprev[2])*(pt[2] - vprev[2]));
	if (dist > 0.02f)
		return false;

	tmpX[NB_BS - 1] = -w*(pt[0] - vprev[0]);
	tmpY[NB_BS - 1] = -w*(pt[1] - vprev[1]);
	tmpZ[NB_BS - 1] = -w*(pt[2] - vprev[2]);

	return true;
}

bool FaceCap::search(int tid, int i, int j, float *tmpX) {
	// Rcurr and Tcurr should be the transformation that align VMap to VMap_prev
	float bumpIn[4];
	float nBump[4];
	float nprev[4];
	float vprev[4];
	int p_indx[2];
	float distThres = 0.01f;
	float angleThres = 0.6f;

	bumpIn[0] = _Bump.at<cv::Vec4f>(i, j)[0];
	bumpIn[1] = _Bump.at<cv::Vec4f>(i, j)[1];
	bumpIn[2] = _Bump.at<cv::Vec4f>(i, j)[2];
	bumpIn[3] = _Bump.at<cv::Vec4f>(i, j)[3];

	float w = 1.0f;
	nBump[0] = _NMapBump.at<cv::Vec4f>(i, j)[0];
	nBump[1] = _NMapBump.at<cv::Vec4f>(i, j)[1];
	nBump[2] = _NMapBump.at<cv::Vec4f>(i, j)[2];

	if (bumpIn[1] == 0.0f || _LabelsMask.at<cv::Vec4b>(i, j)[3] == 0 || (nBump[0] == 0.0f && nBump[1] == 0.0f && nBump[2] == 0.0f))
		return false;
	
	float nmleBump[3]; 
	nmleBump[0] = _Pose[0] * nBump[0] + _Pose[4] * nBump[1] + _Pose[8] * nBump[2]; //Rcurr is row major
	nmleBump[1] = _Pose[1] * nBump[0] + _Pose[5] * nBump[1] + _Pose[9] * nBump[2];
	nmleBump[2] = _Pose[2] * nBump[0] + _Pose[6] * nBump[1] + _Pose[10] * nBump[2];

	float nmle[3] = { 0.0f, 0.0f, 0.0f };
	float pt[3] = { 0.0f, 0.0f, 0.0f };
	float pt_ref[3] = { 0.0f, 0.0f, 0.0f };
	float pt_TB[27][3];
	float nmleTmp[3] = { 0.0f, 0.0f, 0.0f };
	float ptTmp[3] = { 0.0f, 0.0f, 0.0f };
	float d = bumpIn[0] / 1000.0f;

	pt[0] = _VerticesBS[6 * tid];
	pt[1] = _VerticesBS[6 * tid + 1];
	pt[2] = _VerticesBS[6 * tid + 2];
	nmleTmp[0] = _VerticesBS[6 * tid + 3];
	nmleTmp[1] = _VerticesBS[6 * tid + 4];
	nmleTmp[2] = _VerticesBS[6 * tid + 5];

	ptTmp[0] = pt[0] + d*nmleTmp[0];
	ptTmp[1] = pt[1] + d*nmleTmp[1];
	ptTmp[2] = pt[2] + d*nmleTmp[2];

	nmle[0] = nmleTmp[0] * _Pose[0] + nmleTmp[1] * _Pose[4] + nmleTmp[2] * _Pose[8];
	nmle[1] = nmleTmp[0] * _Pose[1] + nmleTmp[1] * _Pose[5] + nmleTmp[2] * _Pose[9];
	nmle[2] = nmleTmp[0] * _Pose[2] + nmleTmp[1] * _Pose[6] + nmleTmp[2] * _Pose[10];

	if (nmleBump[2] > 0.0f)
		return false;
	
	pt[0] = ptTmp[0] * _Pose[0] + ptTmp[1] * _Pose[4] + ptTmp[2] * _Pose[8];
	pt[1] = ptTmp[0] * _Pose[1] + ptTmp[1] * _Pose[5] + ptTmp[2] * _Pose[9];
	pt[2] = ptTmp[0] * _Pose[2] + ptTmp[1] * _Pose[6] + ptTmp[2] * _Pose[10];
	pt_ref[0] = pt[0] + _Pose[12];
	pt_ref[1] = pt[1] + _Pose[13];
	pt_ref[2] = pt[2] + _Pose[14];

	for (int k = 1; k < NB_BS; k++) {
		// This blended normal is not really a normal since it may be not normalized
		nmleTmp[0] = _VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 3];
		nmleTmp[1] = _VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 4];
		nmleTmp[2] = _VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 5];

		ptTmp[0] = _VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid];
		ptTmp[1] = _VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 1];
		ptTmp[2] = _VerticesBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 2];

		ptTmp[0] = ptTmp[0] + d*nmleTmp[0];
		ptTmp[1] = ptTmp[1] + d*nmleTmp[1];
		ptTmp[2] = ptTmp[2] + d*nmleTmp[2];

		pt_TB[k - 1][0] = ptTmp[0] * _Pose[0] + ptTmp[1] * _Pose[4] + ptTmp[2] * _Pose[8];
		pt_TB[k - 1][1] = ptTmp[0] * _Pose[1] + ptTmp[1] * _Pose[5] + ptTmp[2] * _Pose[9];
		pt_TB[k - 1][2] = ptTmp[0] * _Pose[2] + ptTmp[1] * _Pose[6] + ptTmp[2] * _Pose[10];

		pt[0] = pt[0] + _BlendshapeCoeff[k] * pt_TB[k - 1][0];
		pt[1] = pt[1] + _BlendshapeCoeff[k] * pt_TB[k - 1][1];
		pt[2] = pt[2] + _BlendshapeCoeff[k] * pt_TB[k - 1][2];
	}

	pt[0] = pt[0] + _Pose[12];
	pt[1] = pt[1] + _Pose[13];
	pt[2] = pt[2] + _Pose[14];

	p_indx[0] = min(cDepthWidth - 1, max(0, int(round((pt[0] / fabs(pt[2]))*_IntrinsicRGB[0] + _IntrinsicRGB[2]))));
	p_indx[1] = min(cDepthHeight - 1, max(0, int(round((pt[1] / fabs(pt[2]))*_IntrinsicRGB[1] + _IntrinsicRGB[3]))));

	int size = 1;
	int li = max(p_indx[1] - size, 0);
	int ui = min(p_indx[1] + size + 1, cDepthHeight);
	int lj = max(p_indx[0] - size, 0);
	int uj = min(p_indx[0] + size + 1, cDepthWidth);
	float dist;
	float min_dist = 1000.0f;
	float best_n[4];
	float best_v[4];

	for (int u = li; u < ui; u++) {
		for (int v = lj; v < uj; v++) {

			nprev[0] = _NMap.at<cv::Vec4f>(u, v)[0];
			nprev[1] = _NMap.at<cv::Vec4f>(u, v)[1];
			nprev[2] = _NMap.at<cv::Vec4f>(u, v)[2];

			if (nprev[0] == 0.0f && nprev[1] == 0.0f && nprev[2] == 0.0f)
				continue;

			vprev[0] = _VMap.at<cv::Vec4f>(u, v)[0];
			vprev[1] = _VMap.at<cv::Vec4f>(u, v)[1];
			vprev[2] = _VMap.at<cv::Vec4f>(u, v)[2];
			vprev[3] = _VMap.at<cv::Vec4f>(u, v)[3];

			dist = sqrt((vprev[0] - pt[0])*(vprev[0] - pt[0]) + (vprev[1] - pt[1])*(vprev[1] - pt[1]) + (vprev[2] - pt[2])*(vprev[2] - pt[2]));
			float dist_angle = nmleBump[0] * nprev[9] + nmleBump[1] * nprev[1] + nmleBump[2] * nprev[2];

			if (dist < min_dist && dist_angle > angleThres) {
				min_dist = dist;
				best_n[0] = nprev[0]; best_n[1] = nprev[1]; best_n[2] = nprev[2];
				best_v[0] = vprev[0]; best_v[1] = vprev[1]; best_v[2] = vprev[2]; best_v[3] = vprev[3];
			}
		}
	}

	if (min_dist > distThres)
		return false;
	
	for (int k = 0; k < NB_BS - 1; k++) {
		tmpX[k] = w*(best_n[0]*pt_TB[k][0] + best_n[1]*pt_TB[k][1] + best_n[2]*pt_TB[k][2]);
	}

	tmpX[NB_BS - 1] = -w*(best_n[0] * (pt[0] - best_v[0]) + best_n[1] * (pt[1] - best_v[1]) + best_n[2] * (pt[2] - best_v[2]));

	return true;

}

void FaceCap::EstimateBlendShapeCoefficientsPR(vector<MyMesh *> Blendshape) {

	Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(NB_BS - 1, NB_BS - 1);
	Eigen::VectorXd cc = Eigen::VectorXd(NB_BS - 1);
	Eigen::VectorXd x0 = Eigen::VectorXd(NB_BS - 1);
	Eigen::VectorXd xres = Eigen::VectorXd(NB_BS - 1);
	Eigen::VectorXd lb = Eigen::VectorXd(NB_BS - 1);
	Eigen::VectorXd ub = Eigen::VectorXd(NB_BS - 1);

	if (_idx_curr < 10) 
		return;

	int size_tables = ((NB_BS)*(NB_BS + 1)) / 2;

	/* 1. Compute matrix B **************************/
	// Component from the geometry

	bool converged = false;
	int max_iter = 3;
	int iter = 0;
	int shift;
	double value;
	int fact = 1;

	int length;
	int length_out;
	int length2;
	int length_out2;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Pose[4 * j + i] = float(_Rotation_inv(i, j));
		}
		_Pose[12 + i] = float(_Translation_inv(i));
	}


	float BInit[NB_BS];
	for (int i = 1; i < NB_BS; i++)
		BInit[i] = _BlendshapeCoeff[i];

	for (int lvl = _lvl - 1; lvl > -1; lvl--) {

		int fact = int(pow((float)2.0, (int)lvl));

		iter = 1;
		converged = iter > _max_iterPR[lvl];

		while (!converged) {
			_nbMatches = 0;

			/*************************Build the system****************************/
			for (int l = 0; l < size_tables; l++)
				_outbuffReduce[l] = 0.0;

			unsigned char visibility;
			bool found_coresp = false;
			float weight = 1.0;

			float row[NB_BS];
			float rowY[NB_BS];
			float rowZ[NB_BS];
			
			for (int i = 0; i < BumpHeight; i += fact) {
				for (int j = 0; j < BumpWidth; j += fact) {

					int tid = i * BumpWidth + j;

					//visibility = _FaceSegment.at<cv::Vec4b>(i, j)[0];
					found_coresp = false;
					//if (visibility == 255) {
						found_coresp = search(tid, i, j, row);
					//}

					if (found_coresp) {

						shift = 0;
						for (int k = 0; k < NB_BS - 1; k++) {
							for (int l = k; l < NB_BS; l++)          // cols + b
							{
								_outbuffReduce[shift] = _outbuffReduce[shift] + double(row[k] * row[l]);
								shift++;
							}
							_BCPU(k, _nbMatches) = row[k];
						}
						_BCPU(NB_BS - 1, _nbMatches) = row[NB_BS - 1];
						_nbMatches++;
					}
				}
			}

			// Build correspondences for landmarks
			for (int i = 0; i < 43; i++) {
				found_coresp = false;
				found_coresp = searchLandMark_PR(Blendshape, i, row, rowY, rowZ);
				shift = 0;
				for (int k = 0; k < NB_BS - 1; k++) {
					for (int l = k; l < NB_BS; l++)          // cols + b
					{
						_outbuffReduce[shift] = _outbuffReduce[shift] + double(row[k] * row[l])
																		+ double(rowY[k] * rowY[l])
																		+ double(rowZ[k] * rowZ[l]);
						shift++;
					}
					_BCPU(k, _nbMatches) = row[k];
					_BCPU(k, _nbMatches+1) = rowY[k];
					_BCPU(k, _nbMatches+2) = rowZ[k];
				}
				_BCPU(NB_BS - 1, _nbMatches) = row[NB_BS - 1];
				_BCPU(NB_BS - 1, _nbMatches+1) = rowY[NB_BS - 1];
				_BCPU(NB_BS - 1, _nbMatches+2) = rowZ[NB_BS - 1];
				_nbMatches+=3;
			}

			// Add smoothness constraints 1
			for (int i = 0; i < NB_BS-1; i++) {

				for (int k = 0; k < NB_BS - 1; k++) {
					row[k] = 0.0f;
				}
				float lambda = 0.2f;
				row[i] = lambda;
				row[NB_BS - 1] = -lambda * (_BlendshapeCoeff[i + 1]);

				shift = 0;
				for (int k = 0; k < NB_BS - 1; k++) {
					for (int l = k; l < NB_BS; l++)          // cols + b
					{
						_outbuffReduce[shift] = _outbuffReduce[shift] + double(row[k] * row[l]);
						shift++;
					}
					_BCPU(k, _nbMatches) = row[k];
				}
				_BCPU(NB_BS - 1, _nbMatches) = row[NB_BS - 1];
				_nbMatches++;
			}

			// Add smoothness constraints 2
			for (int i = 0; i < NB_BS-1; i++) {

				for (int k = 0; k < NB_BS - 1; k++) {
					row[k] = 0.0f;
				}
				float lambda = 0.2f;
				row[i] = lambda;
				row[NB_BS - 1] = lambda * (BInit[i + 1] - _BlendshapeCoeff[i + 1]);

				shift = 0;
				for (int k = 0; k < NB_BS - 1; k++) {
					for (int l = k; l < NB_BS; l++)          // cols + b
					{
						_outbuffReduce[shift] = _outbuffReduce[shift] + double(row[k] * row[l]);
						shift++;
					}
					_BCPU(k, _nbMatches) = row[k];
				}
				_BCPU(NB_BS - 1, _nbMatches) = row[NB_BS - 1];
				_nbMatches++;
			}

			//cout << "nb_matches: " << _nbMatches << endl;

			/***2. Compute pseud inverse*************************************************/
			// Compute BTB

			shift = 0;
			for (int i = 0; i < NB_BS - 1; i++)
			{
				for (int j = i; j < NB_BS; j++)
				{
					value = double(_outbuffReduce[shift++]);
					if (j == NB_BS - 1)       // vector b
						cc(i) = value;
					else
						Q(j, i) = Q(i, j) = value;
				}
			}

			//determinant
			double det = Q.determinant();

			if (det == 0.0/*fabs(det) < 1e-15*/ || det != det)
			{
				if (det != det) std::cout << "qnan" << endl;
				std::cout << "det null" << endl;
				//cout << "_BlendshapeCoeff: " << endl;
				//for (int i = 1; i < NB_BS; i++) {
				//	cout << _BlendshapeCoeff[i] << endl;
				//}
				return;
			}

			Eigen::MatrixXd Q_inv = Q.inverse();

			for (int i = 0; i < NB_BS - 1; i++) {
				for (int j = 0; j < NB_BS - 1; j++) {
					_Qinv[i * (NB_BS - 1) + j] = float(Q_inv(i, j));
				}
			}

			/// Compute matrix PseudoInverse*B and PseudoInverse*cc
			for (int i = 0; i < NB_BS - 1; i++) {
				float val_2 = 0.0f;
				for (int j = 0; j < _nbMatches; j++) {
					float val = 0.0f;
					for (int l = 0; l < NB_BS - 1; l++) {
						val += Q_inv(i, l)*_BCPU(l, j);
					}
					val_2 += val*_BCPU(NB_BS - 1, j);
				}

				// Compute I - Pseudo * B
				x0(i) = val_2;
				lb(i) = double(0.0f - _BlendshapeCoeff[i + 1]);
				ub(i) = double(1.0f - _BlendshapeCoeff[i + 1]);
			}

			/*************************END Build system****************************/

			xres = Utility::ParallelRelaxation(Q_inv, x0, lb, ub);

			iter++;

			float residual = 0.0f;
			for (int i = 1; i < NB_BS; i++) {
				residual += fabs(_BlendshapeCoeff[i] - float(xres(i - 1)));
				_BlendshapeCoeff[i] = _BlendshapeCoeff[i] + float(xres(i - 1));
				if (_BlendshapeCoeff[i] < 0.0)
					_BlendshapeCoeff[i] = 0.0;
				if (_BlendshapeCoeff[i] > 1.0)
					_BlendshapeCoeff[i] = 1.0;
			}
			converged = (iter > _max_iterPR[lvl]/* || residual < 1.0e-5*/);

		}
	}
}

void FaceCap::EstimateBlendShapeCoefficientsLandmarks(vector<MyMesh *> Blendshape) {
	if (_idx_curr < 30)
		return;

	// Rotate the reference mesh and compute translation
	MyMesh * RefMesh = Blendshape[0];

	float *VBS = (float *) malloc(NB_BS * 6 * BumpHeight * BumpWidth*sizeof(float));

	/// ESTIMATE RIGID ALIGNMENT ////////////

	Eigen::Matrix4d Transfo = Eigen::Matrix4d::Identity();
	Eigen::Matrix4d Transfo_prev = Eigen::Matrix4d::Identity();
	Eigen::Matrix<double, 3, NB_BS, Eigen::RowMajor> Jac;
	Eigen::Matrix<double, NB_BS - 1, NB_BS - 1, Eigen::RowMajor> A;
	Eigen::Matrix<double, NB_BS - 1, 1> b;
	Eigen::Matrix<double, NB_BS - 1, 1> result;

	float BlendshapeInit[NB_BS];
	float BlendshapePrev[NB_BS];
	for (int i = 0; i < NB_BS; i++) {
		BlendshapeInit[i] = _BlendshapeCoeff[i];
		BlendshapePrev[i] = _BlendshapeCoeff[i];
	}

	float Mat[434];
	float pt_T[3];
	int iter = 0;
	int _max_iter = 100;
	bool converged = false;
	double lambda = 1.0;
	float prev_res = 1.0e10;

	while (!converged) {
		float residual = 0.0f;
		for (int i = 0; i < 434; i++)
			Mat[i] = 0.0f;

		int count_matches = 0;
		for (int i = 0; i < 43; i++) {

			MyPoint *LANDMARKPT = new MyPoint(_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0],
				_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1],
				_VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2]);
			if (LANDMARKPT->_x == 0.0f && LANDMARKPT->_y == 0.0f && LANDMARKPT->_z == 0.0f) {
				//cout << "Landmark " << i << " NULL" << endl;
				delete LANDMARKPT;
				continue;
			}
			count_matches++;

			//float pt[3];
			//float pt_T[3];
			//pt_T[0] = RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_x;
			//pt_T[1] = RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_y;
			//pt_T[2] = RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_z;

			//pt[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2);
			//pt[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2);
			//pt[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2);

			//float w_l = 100.0f;
			//float pt_diff[3];
			//for (int k = 1; k < NB_BS; k++) {
			//	MyMesh * Mesh = Blendshape[k];
			//	pt_T[0] = Mesh->_vertices[FACIAL_LANDMARKS[i]]->_x - RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_x;
			//	pt_T[1] = Mesh->_vertices[FACIAL_LANDMARKS[i]]->_y - RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_y;
			//	pt_T[2] = Mesh->_vertices[FACIAL_LANDMARKS[i]]->_z - RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_z;

			//	pt_diff[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2);
			//	pt_diff[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2);
			//	pt_diff[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2);

			//	Jac(0, k - 1) = w_l*(pt_diff[0]);
			//	Jac(1, k - 1) = w_l*(pt_diff[1]);
			//	Jac(2, k - 1) = w_l*(pt_diff[2]);

			//	//cout << "Jac(0, k - 1) " << Jac(0, k - 1) << ", " << Jac(1, k - 1) << ", " << Jac(2, k - 1) << endl;

			//	pt[0] = pt[0] + _BlendshapeCoeff[k] * pt_diff[0];
			//	pt[1] = pt[1] + _BlendshapeCoeff[k] * pt_diff[1];
			//	pt[2] = pt[2] + _BlendshapeCoeff[k] * pt_diff[2];
			//}

			//pt[0] = pt[0] + _Translation_inv(0);
			//pt[1] = pt[1] + _Translation_inv(1);
			//pt[2] = pt[2] + _Translation_inv(2);

			int u = _landmarksBump.at<int>(0, i);
			int v = _landmarksBump.at<int>(1, i);

			if (u < 0 || u > BumpHeight - 1 || v < 0 || v > BumpWidth - 1) {
				delete LANDMARKPT;
				cout << "Out of bound" << endl;
				continue;
			}

			if (_Bump.at<cv::Vec4f>(u, v)[1] == 0.0f) {
				delete LANDMARKPT;
				cout << "Mask null" << endl;
				continue;
			}

			int tid = u*BumpWidth + v;
			float pt[3];
			float nmleTmp[3];
			pt[0] = VBS[6 * tid];
			pt[1] = VBS[6 * tid + 1];
			pt[2] = VBS[6 * tid + 2];
			nmleTmp[0] = VBS[6 * tid + 3];
			nmleTmp[1] = VBS[6 * tid + 4];
			nmleTmp[2] = VBS[6 * tid + 5];

			float pt_T[3];
			float d = _Bump.at<cv::Vec4f>(u, v)[0]/1000.0f;
			pt_T[0] = pt[0] + d*nmleTmp[0];
			pt_T[1] = pt[1] + d*nmleTmp[1];
			pt_T[2] = pt[2] + d*nmleTmp[2];

			float nmle[3];
			nmle[0] = nmleTmp[0] * _Rotation_inv(0, 0) + nmleTmp[1] * _Rotation_inv(0, 1) + nmleTmp[2] * _Rotation_inv(0, 2);
			nmle[1] = nmleTmp[0] * _Rotation_inv(1, 0) + nmleTmp[1] * _Rotation_inv(1, 1) + nmleTmp[2] * _Rotation_inv(1, 2);
			nmle[2] = nmleTmp[0] * _Rotation_inv(2, 0) + nmleTmp[1] * _Rotation_inv(2, 1) + nmleTmp[2] * _Rotation_inv(2, 2);
			// Test if normal oriented backward
			if (nmle[2] > 0.0f) {
				delete LANDMARKPT;
				cout << "nmle[2] > 0.0f" << endl;
				continue;
			}

			pt[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2);
			pt[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2);
			pt[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2);

			float w_l = 100.0f;
			float pt_diff[3];
			for (int k = 1; k < NB_BS; k++) {
				//(f((bj-bo) + d(nj-n0))
				pt_T[0] = VBS[k * 6 * BumpHeight*BumpWidth + 6 * tid] + d * VBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 3];
				pt_T[1] = VBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 1] + d * VBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 4];
				pt_T[2] = VBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 2] + d * VBS[k * 6 * BumpHeight*BumpWidth + 6 * tid + 5];

				pt_diff[0] = pt_T[0] * _Rotation_inv(0, 0) + pt_T[1] * _Rotation_inv(0, 1) + pt_T[2] * _Rotation_inv(0, 2);
				pt_diff[1] = pt_T[0] * _Rotation_inv(1, 0) + pt_T[1] * _Rotation_inv(1, 1) + pt_T[2] * _Rotation_inv(1, 2);
				pt_diff[2] = pt_T[0] * _Rotation_inv(2, 0) + pt_T[1] * _Rotation_inv(2, 1) + pt_T[2] * _Rotation_inv(2, 2);

				Jac(0, k - 1) = w_l*(pt_diff[0]);
				Jac(1, k - 1) = w_l*(pt_diff[1]);
				Jac(2, k - 1) = w_l*(pt_diff[2]);

				pt[0] = pt[0] + _BlendshapeCoeff[k] * pt_diff[0];
				pt[1] = pt[1] + _BlendshapeCoeff[k] * pt_diff[1];
				pt[2] = pt[2] + _BlendshapeCoeff[k] * pt_diff[2];
			}

			pt[0] = pt[0] + _Translation_inv(0);
			pt[1] = pt[1] + _Translation_inv(1);
			pt[2] = pt[2] + _Translation_inv(2);

			float dist = sqrt((pt[0] - LANDMARKPT->_x)*(pt[0] - LANDMARKPT->_x) + (pt[1] - LANDMARKPT->_y)*(pt[1] - LANDMARKPT->_y) + (pt[2] - LANDMARKPT->_z)*(pt[2] - LANDMARKPT->_z));
			if (dist > 0.05f) {
				delete LANDMARKPT;
				cout << "dist > 0.05f" << endl;
				continue;
			}

			Jac(0, NB_BS - 1) = w_l*(LANDMARKPT->_x - pt[0]);
			Jac(1, NB_BS - 1) = w_l*(LANDMARKPT->_y - pt[1]);
			Jac(2, NB_BS - 1) = w_l*(LANDMARKPT->_z - pt[2]);

			int shift = 0;
			for (int k = 0; k < NB_BS-1; ++k)        //rows
			{
				for (int l = k; l < NB_BS; ++l)          // cols + b
				{
					Mat[shift] = Mat[shift] + Jac(0, k)*Jac(0, l) + Jac(1, k)*Jac(1, l) + Jac(2, k)*Jac(2, l);
					shift++;
				}
			}
			residual += Jac(0, NB_BS - 1)*Jac(0, NB_BS - 1) + Jac(1, NB_BS - 1)*Jac(1, NB_BS - 1) + Jac(2, NB_BS - 1)*Jac(2, NB_BS - 1);

			delete LANDMARKPT;

		}

		// Add regularization terms
		// min L2 norm
		float w = 0.5f;
		for (int i = 1; i < NB_BS; i++) {
			for (int k = 1; k < NB_BS; k++) {
				Jac(0, k - 1) = 0.0f;
				Jac(1, k - 1) = 0.0f;
				Jac(2, k - 1) = 0.0f;
			}
			Jac(0, i - 1) = w*1.0f;
			Jac(1, i - 1) = w*1.0f;
			Jac(2, i - 1) = w*1.0f;
			Jac(0, NB_BS - 1) = -w*_BlendshapeCoeff[i];
			Jac(1, NB_BS - 1) = -w*_BlendshapeCoeff[i];
			Jac(2, NB_BS - 1) = -w*_BlendshapeCoeff[i];

			int shift = 0;
			for (int k = 0; k < NB_BS - 1; ++k)        //rows
			{
				for (int l = k; l < NB_BS; ++l)          // cols + b
				{
					Mat[shift] = Mat[shift] + Jac(0, k)*Jac(0, l) + Jac(1, k)*Jac(1, l) + Jac(2, k)*Jac(2, l);
					shift++;
				}
			}
			residual += Jac(0, NB_BS - 1)*Jac(0, NB_BS - 1) + Jac(1, NB_BS - 1)*Jac(1, NB_BS - 1) + Jac(2, NB_BS - 1)*Jac(2, NB_BS - 1);
		}

		//smooth
		w = 0.1f;
		for (int i = 1; i < NB_BS; i++) {
			for (int k = 1; k < NB_BS; k++) {
				Jac(0, k - 1) = 0.0f;
				Jac(1, k - 1) = 0.0f;
				Jac(2, k - 1) = 0.0f;
			}
			Jac(0, i - 1) = w*1.0f;
			Jac(1, i - 1) = w*1.0f;
			Jac(2, i - 1) = w*1.0f;
			Jac(0, NB_BS - 1) = w*(BlendshapeInit[i] - _BlendshapeCoeff[i]);
			Jac(1, NB_BS - 1) = w*(BlendshapeInit[i] - _BlendshapeCoeff[i]);
			Jac(2, NB_BS - 1) = w*(BlendshapeInit[i] - _BlendshapeCoeff[i]);

			int shift = 0;
			for (int k = 0; k < NB_BS - 1; ++k)        //rows
			{
				for (int l = k; l < NB_BS; ++l)          // cols + b
				{
					Mat[shift] = Mat[shift] + Jac(0, k)*Jac(0, l) + Jac(1, k)*Jac(1, l) + Jac(2, k)*Jac(2, l);
					shift++;
				}
			}
			residual += Jac(0, NB_BS - 1)*Jac(0, NB_BS - 1) + Jac(1, NB_BS - 1)*Jac(1, NB_BS - 1) + Jac(2, NB_BS - 1)*Jac(2, NB_BS - 1);
		}

		//std::cout << "Residual: " << residual << std::endl;

		if (count_matches < 3) {
			free(VBS);
			return;
		}

		if (residual > prev_res) {
			lambda = lambda / 2.0;
			for (int i = 0; i < NB_BS; i++) {
				_BlendshapeCoeff[i] = BlendshapePrev[i];
			}
			continue;
		}
		else {
			prev_res = residual;
			for (int i = 0; i < NB_BS; i++) {
				BlendshapePrev[i] = _BlendshapeCoeff[i];
			}
		}

		int shift = 0;
		for (int i = 0; i < NB_BS - 1; ++i) {  //rows
			for (int j = i; j < NB_BS; ++j)    // cols + b
			{
				double value = double(Mat[shift++]);
				if (j == NB_BS - 1)       // vector b
					b(i) = value;
				else
					A(j, i) = A(i, j) = value;
			}
		}

		//cout << A << endl;
		//checking nullspace
		float det = A.determinant();

		if (fabs(det) < 1e-15 || det != det)
		{
			if (det != det) std::cout << "qnan" << endl;
			std::cout << "det null" << endl;
			free(VBS);
			return;
		}

		result = lambda * A.inverse() * b;

		float rms = 0.0f;
		for (int i = 0; i < NB_BS - 1; ++i) {
			_BlendshapeCoeff[i + 1] = std::max(0.0f, std::min(1.0f, (float) (_BlendshapeCoeff[i + 1] + result(i))));
			rms += (_BlendshapeCoeff[i + 1] - BlendshapePrev[i + 1]) * (_BlendshapeCoeff[i + 1] - BlendshapePrev[i + 1]);
		}

		iter++;

		if (iter > _max_iter || sqrt(rms) < 1.0e-10) {
			converged = true;
		}

	}
	free(VBS);

	/*std::cout << "iter: " << iter << ", _BlendshapeCoeff: " << std::endl;
	for (int i = 0; i < NB_BS - 1; ++i) {
		std::cout << ", " << _BlendshapeCoeff[i + 1];
	}
	std::cout << " " << endl;*/

	//int tt;
	//cin >> tt;

}

void FaceCap::SetLandmarks(MyMesh *RefMesh) {

	_landmarkOK = true;

	float ptLM[3];
	//for (int i = 31; i < 43; i++) {
	//	if (_landmarks.cols < i + 1)
	//		continue;

	//	ptLM[0] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[0];
	//	ptLM[1] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[1];
	//	ptLM[2] = _VMap.at<cv::Vec4f>(Myround(_landmarks.at<float>(1, i))*cDepthWidth + Myround(_landmarks.at<float>(0, i)))[2];

	//	if (ptLM[0] == 0.0f && ptLM[1] == 0.0f && ptLM[2] == 0.0f) {
	//		cout << "No landmark VMAP!" << endl;
	//		continue;
	//	}

	//	// Search for closest point in the bump image
	//	float min_dist = 1.0e6;
	//	int best_i, best_j;
	//	float pt[3];
	//	for (int u = 0; u < BumpHeight; u ++) {
	//		for (int v = 0; v < BumpWidth; v ++) {
	//			if (_Bump.at<cv::Vec4f>(u, v)[1] == 0.0f)
	//				continue;

	//			pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(0, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(0, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(0, 2) + _Translation_inv(0);
	//			pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(1, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(1, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(1, 2) + _Translation_inv(1);
	//			pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[0] * _Rotation_inv(2, 0) + _VMapBump.at<cv::Vec4f>(u, v)[1] * _Rotation_inv(2, 1) + _VMapBump.at<cv::Vec4f>(u, v)[2] * _Rotation_inv(2, 2) + _Translation_inv(2);

	//			/*pt[0] = _VMapBump.at<cv::Vec4f>(u, v)[0];
	//			pt[1] = _VMapBump.at<cv::Vec4f>(u, v)[1];
	//			pt[2] = _VMapBump.at<cv::Vec4f>(u, v)[2];*/
	//			float dist = sqrt((pt[0] - ptLM[0])*(pt[0] - ptLM[0]) + (pt[1] - ptLM[1])*(pt[1] - ptLM[1]) + (pt[2] - ptLM[2])*(pt[2] - ptLM[2]));
	//			if (dist < min_dist) {
	//				min_dist = dist;
	//				best_i = u;
	//				best_j = v;
	//			}
	//		}
	//	}

	//	if (min_dist < 0.005) {
	//		_landmarksBump.at<int>(0, i) = best_i;
	//		_landmarksBump.at<int>(1, i) = best_j;
	//		//cout << "best_i: " << best_i << "; best_j: " << best_j << endl;
	//	}
	//	else {
	//		_landmarksBump.at<int>(0, i) = -1;
	//		_landmarksBump.at<int>(1, i) = -1;
	//		//cout << "No landmark!" << endl;
	//		_landmarkOK = false;
	//	}

	//}

	for (int i = 0; i < 43; i++) {
		int best_i = Myround(RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_u*float(BumpHeight));
		int best_j = Myround(RefMesh->_vertices[FACIAL_LANDMARKS[i]]->_v*float(BumpWidth));
		if (i == 22) {
			best_i = 110;
			best_j = 169;
		}
		if (i == 25) {
			best_i = 130;
			best_j = 169;
		}

		/*if (i == 38) {
			best_i = 128;
			best_j = 122;
		}

		if (i == 39) {
			best_i = 126;
			best_j = 122;
		}

		if (i == 40) {
			best_i = 120;
			best_j = 122;
		}

		if (i == 41) {
			best_i = 114;
			best_j = 122;
		}

		if (i == 42) {
			best_i = 110;
			best_j = 122;
		}*/

		_landmarksBump.at<int>(0, i) = best_i;
		_landmarksBump.at<int>(1, i) = best_j;
	}
}

void FaceCap::LoadAnimatedModel() {
	// Load the Bump and RGB images.
	float bumpval;
	cv::Mat img = cv::imread(string("Bump.png"), CV_LOAD_IMAGE_UNCHANGED);
	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			if (img.at<cv::Vec3b>(i, j)[0] == 255 && img.at<cv::Vec3b>(i, j)[1] == 255 && img.at<cv::Vec3b>(i, j)[2] == 255) {
				_Bump.at<cv::Vec4f>(i, j)[0] = 0.0f;
				_Bump.at<cv::Vec4f>(i, j)[1] = 0.0f;
				continue;
			}

			if (img.at<cv::Vec3b>(i, j)[2] == 0) {
				//bumpval = ((((float(img.at<cv::Vec3b>(i, j)[1]) / 2.0f) / 255.0f)*2000.0f) - 1000.0f) / 50.0f;
				bumpval = ((((float(img.at<cv::Vec3b>(i, j)[1]) / 2.0f) / 255.0f)*6000.0f) - 3000.0f) / 50.0f;
			}
			else {
				//bumpval = (((((float(img.at<cv::Vec3b>(i, j)[2]) / 2.0f) + 128.0f) / 255.0f)*2000.0f) - 1000.0f) / 50.0f;
				bumpval = (((((float(img.at<cv::Vec3b>(i, j)[2]) / 2.0f) + 128.0f) / 255.0f)*6000.0f) - 3000.0f) / 50.0f;
			}
			_Bump.at<cv::Vec4f>(i, j)[0] = bumpval;
			_Bump.at<cv::Vec4f>(i, j)[1] = 1.0f;
		}
	}

	//Load RGB image
	img = cv::imread(string("BumpRGB.png"), CV_LOAD_IMAGE_UNCHANGED);
	for (int i = 0; i < BumpHeight; i++) {
		for (int j = 0; j < BumpWidth; j++) {
			_RGBMapBump.at<cv::Vec4f>(i, j)[2] = float(img.at<cv::Vec3b>(i, j)[0]);
			_RGBMapBump.at<cv::Vec4f>(i, j)[1] = float(img.at<cv::Vec3b>(i, j)[1]);
			_RGBMapBump.at<cv::Vec4f>(i, j)[0] = float(img.at<cv::Vec3b>(i, j)[2]);
			_RGBMapBump.at<cv::Vec4f>(i, j)[3] = 1.0f;
		}
	}
}

void FaceCap::LoadCoeffPose(char *path) {
	ifstream  filestr;
	char tmpline[256];

	string filename = string(path) + string("\\BSCoeff") + to_string(_idx) + string(".txt");
	filestr.open(filename, fstream::in);
	while (!filestr.is_open()) {
		cout << "Could not open " << filename << endl;
		_idx++;
		return;
	}

	for (int i = 0; i < NB_BS; i++) {
		filestr.getline(tmpline, 256);
		float tmpval;
		sscanf_s(tmpline, "%f", &tmpval);
		_BlendshapeCoeff[i] = tmpval;
		//cout << _BlendshapeCoeff[i] << endl;
	}

	filestr.close();

	filename = string(path) + string("\\Pose") + to_string(_idx) + string(".txt");
	filestr.open(filename, fstream::in);
	while (!filestr.is_open()) {
		cout << "Could not open " << filename << endl;
		_idx++;
		return;
	}

	for (int i = 0; i < 16; i++) {
		filestr.getline(tmpline, 256);
		sscanf_s(tmpline, "%f", &_Pose[i]);
	}

	filestr.close();
	_idx++;
	_idx_curr++;

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			_Rotation_inv(j, i) = _Pose[4 * i + j];
		}
		_Translation_inv(i) = _Pose[12 + i];
	}

	cout << "Param loaded" << endl;
}

void FaceCap::SuperpixelSegmentation() {
	for (int i = 0; i < _FaceL.height; i++) {
		for (int j = 0; j < _FaceL.width; j++) {
			((unsigned char *)_pbuff)[4 * (i*_FaceL.width + j)] = _imgC.at<cv::Vec4b>(_FaceL.y + i, _FaceL.x + j)[3];
			((unsigned char *)_pbuff)[4 * (i*_FaceL.width + j) + 1] = _imgC.at<cv::Vec4b>(_FaceL.y + i, _FaceL.x + j)[0];
			((unsigned char *)_pbuff)[4 * (i*_FaceL.width + j) + 2] = _imgC.at<cv::Vec4b>(_FaceL.y + i, _FaceL.x + j)[1];
			((unsigned char *)_pbuff)[4 * (i*_FaceL.width + j) + 3] = _imgC.at<cv::Vec4b>(_FaceL.y + i, _FaceL.x + j)[2];
		}
	}

	_klabelsIn = (int *)malloc(_FaceL.height*_FaceL.width*sizeof(int));
	//----------------------------------
	// Initialize parameters
	//----------------------------------
	int k = NB_SUP_PIXELS;//Desired number of superpixels.
	double m = 30;//Compactness factor. use a value ranging from 10 to 40 depending on your needs. Default is 10
	int numlabels(0);
	//----------------------------------
	// Perform SLIC on the image buffer
	//----------------------------------
	_segment.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(_pbuff, _FaceL.width, _FaceL.height, _klabelsIn, numlabels, k, m);
	// Alternately one can also use the function DoSuperpixelSegmentation_ForGivenStepSize() for a desired superpixel size
	//----------------------------------
	// Save the labels to a text file
	//----------------------------------
	//_segment.SaveSuperpixelLabels(klabels, BumpWidth, BumpHeight, filename, savepath);
	//----------------------------------
	// Draw boundaries around segments
	//----------------------------------
	//_segment.DrawContoursAroundSegments(_pbuff, _klabelsIn, _FaceL.width, _FaceL.height, 0xff0000);

	/*cv::Mat tmp = cv::Mat(_FaceL.height, _FaceL.width, CV_8UC3);
	for (int i = 0; i < _FaceL.height; i++) {
		for (int j = 0; j < _FaceL.width; j++) {
			//tmp.at<cv::Vec3b>(i, j)[3] = ((unsigned char *)_pbuff)[4 * (i*_FaceL.width + j)];
			tmp.at<cv::Vec3b>(i, j)[0] = ((unsigned char *)_pbuff)[4 * (i*_FaceL.width + j) + 1];
			tmp.at<cv::Vec3b>(i, j)[1] = ((unsigned char *)_pbuff)[4 * (i*_FaceL.width + j) + 2];
			tmp.at<cv::Vec3b>(i, j)[2] = ((unsigned char *)_pbuff)[4 * (i*_FaceL.width + j) + 3];
		}
	}

	cv::imwrite("superpixels.png", tmp);
	cv::imshow("super pixels", tmp);
	cv::waitKey(1);*/
}
