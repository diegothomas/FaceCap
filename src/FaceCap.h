#ifndef __FACECAP_H
#define __FACECAP_H

#include <KLibrary\SensorManager.h>

class FaceCap {

public:

	FaceCap();
	~FaceCap();
		
	bool StartSensor();		//Start sensor	

	void Draw(bool color = true, bool bump = true);
	void DrawBlendedMesh(vector<MyMesh *> Blendshape);
	void DrawQuad(int color = 0, bool bump = true);
	void DrawVBO(bool color = true);
	void DrawRect(bool color = true);
	void DrawLandmark(int i, vector<MyMesh *> Blendshape);

	// Load next RGB-D image
	int Update();
	void Pop();

	int LoadToSave(int k);
	int SaveData(int k);

	bool Compute3DDataCPU(int idx);
	
	// Compute normals for each pixel of the depth image
	void ComputeNormalesDepth();

	// Detect or track facial landmarks using supervised descent method
	void DetectFeatures(cv::CascadeClassifier *face_cascade, bool draw = true);

	// Detect or track facial landmarks using supervised descent method
	//void ReDetectFeatures(cv::CascadeClassifier *face_cascade, bool draw = true);

	// Re-scale all blendshapes to match user landmarks
	bool Rescale(vector<MyMesh *> Blendshape);

	// Roughly align all blendshapes to the detected facial landmarks
	bool AlignToFace(vector<MyMesh *> Blendshape, bool inverted);

	void DataProc();

	bool ElasticRegistrationFull(vector<MyMesh *> Blendshape);

	// Perform elastic registration to match facial features
	void ElasticRegistration(vector<MyMesh *> Blendshape);

	// Transfer expression deformations to all blendshapes
	void ComputeAffineTransfo(vector<MyMesh *> Blendshape);

	// Update Bump image from input RGB-D image
	void MedianFilter(int x, int y, int width, int height);
	void GenerateBump(vector<MyMesh *> Blendshape, int x, int y, int width, int height);

	void ComputeNormalesOptimized(int x, int y, int width, int height);
	void ComputeNormalesApprox(int x, int y, int width, int height);
	void ComputeNormales(int x, int y, int width, int height);

	void ComputeLabels(MyMesh *TheMesh);

	bool searchLandMark(int l_idx, float *n, float *d, float *s);
	bool searchGauss(int i, int j, float *n, float *d, float *s);

	//Rigid registration of input RGB-D frame to 3D model
	void Register(vector<MyMesh *> Blendshape);

	bool searchLandMark_PR(vector<MyMesh *> Blendshape, int l_idx, float *tmpX, float *tmpY, float *tmpZ);
	bool search(int tid, int i, int j, float *tmpX);

	// Estimate animation blendshape coefficients
	void EstimateBlendShapeCoefficientsPR(vector<MyMesh *> Blendshape);
	// Estimate animation blendshape coefficients
	void EstimateBlendShapeCoefficientsLandmarks(vector<MyMesh *> Blendshape);

	// Set landmark indices to fit initial RGB-D image
	void SetLandmarks(MyMesh *RefMesh);

	void LoadAnimatedModel();

	void LoadCoeffPose(char *path);

	void SuperpixelSegmentation();

	int _idx_curr;

	Eigen::Matrix4d _TransfoD2RGB;
	float _IntrinsicRGB[11];

	bool _bSensor;
	bool _landmarkOK;

	cv::Mat _depth_in[2];
	cv::Mat _color_in[10];
	cv::Mat _VMap_in[10];
	cv::Mat _NMap_in[10];
	int _idx_thread[2];

	//Class to read from sensor
	SensorManager* _sensorManager;

	int *TABLE_I;
	int *TABLE_J;

	GLuint _frame_buf;
	GLuint _textureId;
	GLuint _DepthRendererId;

	////////////// Blendshape memory ////////////////

	FaceGPU * _triangles;
	Point3DGPU * _verticesList;

	////////////////////////////////////////////////////

private:

	// 3D input data
	MyPoint **_vertices;
	cv::Mat _depthIn;
	cv::Mat _VMap;
	cv::Mat _NMap;
	cv::Mat _RGBMap;
	cv::Mat _segmentedIn;
	int *_CoordMapping;
	int *_CoordMappingD2RGB;

	int *_CoordMappingIn;
	int *_CoordMappingInD2RGB;
	cv::Mat _imgD;
	cv::Mat _imgC;
	cv::Mat _imgS;
	cv::Mat _pts;
	cv::Mat _ptsRe;

	cv::Mat _VMapBump;
	cv::Mat _NMapBump;
	cv::Mat _RGBMapBump;
	cv::Mat _WeightMap;

	cv::Mat _Bump; // deviation in milimeters
	cv::Mat _BumpSwap;

	cv::Mat _FaceSegment;

	cv::Mat _x_raw;
	cv::Mat _y_raw;
	cv::Mat _ones;
	cv::Mat _grad_x;
	cv::Mat _grad_y;
	cv::Mat _ones_bump;
	cv::Mat _grad_x_bump;
	cv::Mat _grad_y_bump;

	GLuint _VBO;
	GLuint _Index;

	// RGB-D data
	queue<int *> _CoordMaps;
	queue<int *> _CoordMapsD2RGB;
	queue<cv::Mat> _depth;
	queue<cv::Mat> _color;
	queue<cv::Mat> _segmented_color;
	queue<int*> _klabelsSet;

	cv::Mat _LabelsMask;

	unsigned int* _pbuff;
	int* _klabels;
	int* _klabelsIn;

	// Bump image data
	Point3DGPU *_verticesBump;
	float *_VerticesBS;
	BYTE *_RGB;
	//float *_WeightMap;
	float *_Vtx[49];

	double *_outbuff;
	double *_outbuffJTJ;
	double *_outbuffGICP;
	float *_outbuffReduce;
	float *_outbuffResolved;
	Eigen::MatrixXd _BCPU;

	// Parameters for facial landmark detection/tracking
	unique_ptr<SDM> _sdm;
	unique_ptr<HPE> _hpe;

	cv::Mat _prevPts;
	cv::Mat _prevPtsRe;
	cv::Mat _landmarks;
	cv::Mat _landmarks_prev;
	cv::Mat _landmarksBump;
	cv::Mat _landmarks_in[10];
	bool _restart;
	bool _restartRe;
	float _minScore;
	float _minScoreRe;
	queue<cv::Mat> _ptsQ;
	queue<cv::Rect> _ptsRect;
	facio::HeadPose _hp;

	SLIC _segment;

	cv::Rect _pRect; 
	cv::Rect _FaceL;

	vector<vector<Eigen::Matrix3f>> _TransfoExpression;
	vector<SpMat> _MatList1;
	vector<SpMat> _MatList2;

	Eigen::Vector3f _Translation;
	Eigen::Matrix3f _Rotation;

	Eigen::Vector3f _Translation_inv;
	Eigen::Matrix3f _Rotation_inv;

	vector<Eigen::Vector3f> _TranslationWindow;
	vector<Eigen::Matrix3f> _RotationWindow;
	vector<float> _BSCoeff[NB_BS];

	float _BlendshapeCoeff[49];

	float _intrinsic[11];
	float _Pose[16];
	float *_Qinv;
	int _nbMatches;

	// some flags
	bool _draw;
	int _idx;
	char *_path;

	int _lvl;
	int _max_iter[3];
	int _max_iterPR[3];

public:

	// inline function
	inline void SetParam(float *Calib, char *path, bool bSensor) {
		_path = path;
		_draw = true;
		_bSensor = bSensor;

		for (int i = 0; i < 11; i++)
			_intrinsic[i] = Calib[i];

		for (int i = 0; i < cDepthHeight; i++) {
			for (int j = 0; j < cDepthWidth; j++) {
				_x_raw.at<float>(i, j) = (float(j) - _IntrinsicRGB[2]) / _IntrinsicRGB[0];
				_y_raw.at<float>(i, j) = (float(i) - _IntrinsicRGB[3]) / _IntrinsicRGB[1];
			}
		}
	}

	inline void SetCoeff(int anim_indx) {
		/*if (_idx < 5)
		return;*/
		_Rotation_inv = Eigen::Matrix3f::Identity();
		_Translation_inv = Eigen::Vector3f::Zero();

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				_Pose[4 * i + j] = _Rotation_inv(j, i);
			}
			_Pose[12 + i] = _Translation_inv(i);
		}

		for (int i = 0; i < NB_BS; i++)
			_BlendshapeCoeff[i] = 0.0f;
		_BlendshapeCoeff[anim_indx] = 1.0f;
	}

	inline float *getBump() { return (float *)_Bump.data; }

	inline Point3DGPU *getVerticesBump() { return _verticesBump; }

	inline bool Push() {
		if (_depth.size() == FRAME_BUFFER_SIZE) {
			return false;
		}

		_CoordMaps.push(_CoordMappingIn);
		_CoordMapsD2RGB.push(_CoordMappingInD2RGB);
		_depth.push(_imgD);
		_color.push(_imgC);
		_segmented_color.push(_imgS);
		_ptsRect.push(_pRect);
		_klabelsSet.push(_klabelsIn);

		if (!_restart) {
			_ptsQ.push(_pts);
		}
		else {
			_ptsQ.push(cv::Mat());
		}

		return true;
	}

	inline void SetInput(int indx) {
		_color_in[indx].copyTo(_RGBMap);
		_VMap_in[indx].copyTo(_VMap);
		_NMap_in[indx].copyTo(_NMap);
		_landmarks_in[indx].copyTo(_landmarks);
	}

};


#endif
