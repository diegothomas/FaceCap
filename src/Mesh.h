#ifndef __MESH_H
#define __MESH_H

typedef Eigen::Triplet<double> TrplType;

//3D point with index in the mesh, texture coordinates, normal vector and neighboorhing points indices.
template <typename T>
struct Point3D {
	//Position
	T _x;
	T _y;
	T _z;

	// Transformed Position when deformation is happening
	T _Tx;
	T _Ty;
	T _Tz;

	//Normal vector
	T _Nx;
	T _Ny;
	T _Nz;

	// Transformed Normal vector when deformation is happening
	T _TNx;
	T _TNy;
	T _TNz;

	//Color vector
	T _R;
	T _G;
	T _B;

	//texture coordinates
	float _u;
	float _v;

	unsigned char _flag;

	// Index
	int _indx;
	bool _BackPoint;

	// Neighboors indeices
	vector<int> _neighboors;

	Point3D() : _BackPoint(false), _flag(0), _u(-1.0f), _v(-1.0f), _x(0.0f), _y(0.0f), _z(0.0f), _Nx(0.0f), _Ny(0.0f), _Nz(0.0f), 
		_Tx(0.0f), _Ty(0.0f), _Tz(0.0f), _TNx(0.0f), _TNy(0.0f), _TNz(0.0f), _R(0.0f), _G(0.0f), _B(0.0f) {};
	Point3D(T x, T y, T z) : _x(x), _y(y), _z(z), _BackPoint(false), _flag(0), _u(-1.0f), _v(-1.0f)  {};
	Point3D(T x, T y, T z, int indx) : _x(x), _y(y), _z(z), _indx(indx), _BackPoint(false), _flag(0), _u(-1.0f), _v(-1.0f)   {};
	inline Point3D CrossProduct(Point3D a) {
		return Point3D(_y*a._z - _z*a._y, -_x*a._z + _z*a._x, _x*a._y - _y*a._x);
	}
	inline T DotProduct(Point3D a) {
		return _x*a._x + _y*a._y + _z*a._z;
	}
};

struct Point3DGPU {
	//Position
	float _x;
	float _y;
	float _z;

	// Transformed Position when deformation is happening
	float _Tx;
	float _Ty;
	float _Tz;

	//Normal vector
	float _Nx;
	float _Ny;
	float _Nz;

	// Transformed Normal vector when deformation is happening
	float _TNx;
	float _TNy;
	float _TNz;

	//Color vector
	float _R;
	float _G;
	float _B;

	//texture coordinates
	float _u;
	float _v;

	unsigned char _flag;

	// Index
	int _indx;
	bool _BackPoint;

	Point3DGPU() : _BackPoint(false), _flag(0), _u(-1.0f), _v(-1.0f), _x(0.0f), _y(0.0f), _z(0.0f), _Nx(0.0f), _Ny(0.0f), _Nz(0.0f),
		_Tx(0.0f), _Ty(0.0f), _Tz(0.0f), _TNx(0.0f), _TNy(0.0f), _TNz(0.0f), _R(0.0f), _G(0.0f), _B(0.0f) {};
	Point3DGPU(float x, float y, float z) : _x(x), _y(y), _z(z), _BackPoint(false), _flag(0), _u(-1.0f), _v(-1.0f)  {};
	Point3DGPU(float x, float y, float z, int indx) : _x(x), _y(y), _z(z), _indx(indx), _BackPoint(false), _flag(0), _u(-1.0f), _v(-1.0f)   {};
	inline Point3DGPU CrossProduct(Point3DGPU a) {
		return Point3DGPU(_y*a._z - _z*a._y, -_x*a._z + _z*a._x, _x*a._y - _y*a._x);
	}
	inline float DotProduct(Point3DGPU a) {
		return _x*a._x + _y*a._y + _z*a._z;
	}
};

//Texture coordinate
struct TextUV {
	float _u;
	float _v;

	TextUV(float u, float v) : _u(u), _v(v) {};
	inline TextUV operator-(TextUV a) {
		return TextUV(_u - a._u, _v - a._v);
	}
	inline TextUV operator+(TextUV a) {
		return TextUV(_u + a._u, _v + a._v);
	}
	inline TextUV operator*(float a) {
		return TextUV(_u*a, _v*a);
	}
};

//Triangular face
struct Face {
	int _v1;
	int _t1;
	int _n1;

	int _v2;
	int _t2;
	int _n2;

	int _v3;
	int _t3;
	int _n3;

	Face(int v1, int t1, int n1, int v2, int t2, int n2, int v3, int t3, int n3) : _v1(v1), _t1(t1), _n1(n1), _v2(v2), _t2(t2), _n2(n2), _v3(v3), _t3(t3), _n3(n3) {};
};

//Triangular face For GPU computations
struct FaceGPU {
	int _v1;
	int _v2;
	int _v3;

	FaceGPU(int v1, int v2, int v3) : _v1(v1), _v2(v2), _v3(v3) {};
};

template <typename T>
class Mesh { 
private:
	vector<T *> _quaternions;

public:
	vector<Point3D<T> *> _vertices;
	vector<Face *> _triangles;
	vector<TextUV *> _uvs;

	FaceGPU * _trianglesList;
	Point3DGPU * _verticesList;

	vector<int> _indxList;

	Mesh(Point3DGPU * _verticesMem, FaceGPU * _trianglesMem) {
		_vertices.clear(); 
		_triangles.clear();
		_uvs.clear();

		_trianglesList = _trianglesMem;
		_verticesList = _verticesMem;
	};
	~Mesh() {
		int nbQuat = _quaternions.size();
		for (int i = 0; i < nbQuat; i++) {
			std::free(_quaternions.back());
			_quaternions.pop_back();
		}
		while (!_vertices.empty()) {
			std::free(_vertices.back());
			_vertices.pop_back();
		}
		while (!_triangles.empty()) {
			std::free(_triangles.back());
			_triangles.pop_back();
		}
		while (!_uvs.empty()) {
			std::free(_uvs.back());
			_uvs.pop_back();
		}
	};

	// Load mesh from file
	void Load(string filename, bool get_neighboors = false);
	void LoadS(string filename);

	void Write(string filename);

	void Draw(float *Bump);

	void DrawLandmark(int i);

	Point3D<T> *Landmark(int i) { return _vertices[FACIAL_LANDMARKS[i]]; };

	void Scale(T factor);

	void Transform(Eigen::Matrix4d Transfo);

	void Rotate(const cv::Mat& rot);

	void Translate(const cv::Point3f& xyz);

	void AffectToTVal();

	void AffectToTVector(Eigen::VectorXd *xres);

	void AffectToTVectorT(Eigen::VectorXd *xres);

	void ComputeTgtPlane();

	void PopulateMatrix(vector<TrplType> *tripletList, Eigen::VectorXd *b1, int offfset);

	int sizeV();

	void Deform(Eigen::VectorXd *boV);

	void Map(Mesh<float> *M1);

	void Modify(Mesh<float> *M1);

	/*****Inline funtions*****/
	inline int size() { return _vertices.size(); };

};

// Load mesh from file
template<typename T>
void Mesh<T>::Load(string filename, bool get_neighboors) {
	cout << "Loading " << filename << endl;
	string line;
	char tmpline[200];
	ifstream  filestr;
	
	filestr.open(filename, fstream::in);
	while (!filestr.is_open()) {
		cout << "Could not open " << filename << endl;
		return;
	}

	// load mesh
	float x, y, z;
	int v1, t1, n1, v2, t2, n2, v3, t3, n3;
	int j = 0, t = 0, nbVertices;
	char c[200];
	char c2[200];
	while (filestr.good())          // loop while extraction from file is possible
	{
		filestr.getline(tmpline, 256);
		sscanf_s(tmpline, "%s", c, _countof(c));

		if (strcmp(c, "#") == 0) {
			sscanf_s(tmpline, "%s %s %d", c, _countof(c), c2, _countof(c2), &nbVertices);

			if (strcmp(c2, "Vertices:") == 0) {
				_vertices.reserve(nbVertices);
				//cout << "nb Vertices: " << nbVertices << endl;
			}
			else if (strcmp(c2, "Faces:") == 0) {
				_triangles.reserve(nbVertices);
				//cout << "nb Faces: " << nbVertices << endl;
			}
			else if (strcmp(c2, "Uvs:") == 0) {
				_uvs.reserve(nbVertices);
				//cout << "nb Faces: " << nbVertices << endl;
			}
		}

		if (strcmp(c, "v") == 0) {
			sscanf_s(tmpline, "%s %f %f %f", c, _countof(c), &x, &y, &z);
			_verticesList[j]._x = x;
			_verticesList[j]._y = -y;
			_verticesList[j]._z = -z;
			_verticesList[j]._indx = j;


			Point3D<T> *currpt = new Point3D<T>(x, -y, -z, j);

			filestr.getline(tmpline, 256);
			sscanf_s(tmpline, "%s", c, _countof(c));
			if (strcmp(c, "vn") == 0) {
				sscanf_s(tmpline, "%s %f %f %f", c, _countof(c), &x, &y, &z);
				if (x == 0.0f && y == 0.0f && z == 0.0f)
					cout << "Error normal null" << endl;
				float norm = sqrt(x*x + y*y + z*z);
				_verticesList[j]._Nx = x / norm;
				_verticesList[j]._Ny = -y / norm;
				_verticesList[j]._Nz = -z / norm;
				currpt->_Nx = -x / norm;
				currpt->_Ny = -y / norm;
				currpt->_Nz = -z / norm;
			}
			else {
				cout << "error reading mesh file" << endl;
			}

			_vertices.push_back(currpt);
			j++;
		}


		if (strcmp(c, "vt") == 0) {
			sscanf_s(tmpline, "%s %f %f", c, _countof(c), &x, &y);
			TextUV *currUV = new TextUV(x, y);
			_uvs.push_back(currUV);
		}

		if (strcmp(c, "f") == 0) {
			//if (!get_neighboors)
			//	sscanf_s(tmpline, "%s %d/%d/%d %d/%d/%d %d/%d/%d", c, _countof(c), &v1, &t1, &n1, &v2, &t2, &n2, &v3, &t3, &n3);
			//else
				sscanf_s(tmpline, "%s %d/%d %d/%d %d/%d", c, _countof(c), &v1, &t1, &v2, &t2, &v3, &t3);
				n1 = 0;
				n2 = 0;
				n3 = 0;
			if (get_neighboors) {
				_trianglesList[t]._v1 = v1 - 1;
				_trianglesList[t]._v2 = v2 - 1;
				_trianglesList[t]._v3 = v3 - 1;


				_verticesList[v1 - 1]._u = _uvs[t1 - 1]->_u;
				_verticesList[v1 - 1]._v = _uvs[t1 - 1]->_v;

				_verticesList[v2 - 1]._u = _uvs[t2 - 1]->_u;
				_verticesList[v2 - 1]._v = _uvs[t2 - 1]->_v;

				_verticesList[v3 - 1]._u = _uvs[t3 - 1]->_u;
				_verticesList[v3 - 1]._v = _uvs[t3 - 1]->_v;
			}

			_triangles.push_back(new Face(v1 - 1, t1 - 1, n1 - 1, v2 - 1, t2 - 1, n2 - 1, v3 - 1, t3 - 1, n3 - 1));
			t++;
		}
	}
	filestr.close();


	/**********Compute neighboorhing information for the reference blendshape******************/
	if (get_neighboors) {
		float thresh = 10.0;
		_quaternions.reserve(_vertices.size());
		int nbQuat = _vertices.size();
		for (int i = 0; i < nbQuat; i++) {
			_quaternions.push_back(new T[9]);
		}

		/////// TEST FOR FRONT POINTS
		/*ofstream filestrout;
		filestrout.open("front.txt", fstream::out);
		while (!filestrout.is_open()) {
			cout << "Could not open " << filename << endl;
			return;
		}
		vector<int> frontpts;*/

		for (vector<Face *>::iterator it = _triangles.begin(); it != _triangles.end(); it++) {
			/*if (_vertices[(*it)->_v1]->_u != -1.0f && (_vertices[(*it)->_v1]->_u != _uvs[(*it)->_t1]->_u || _vertices[(*it)->_v1]->_v != _uvs[(*it)->_t1]->_v)) {
				cout << "Landmark: " << (*it)->_v1 << endl;
				cout << "u: " << _vertices[(*it)->_v1]->_u << "u: " << _vertices[(*it)->_v1]->_v << endl;
				cout << "u*: " << _uvs[(*it)->_t1]->_u << "u*: " << _uvs[(*it)->_t1]->_v << endl;
			}*/
			_vertices[(*it)->_v1]->_u = _uvs[(*it)->_t1]->_u;
			_vertices[(*it)->_v1]->_v = _uvs[(*it)->_t1]->_v;

			/*bool valid1 = true;
			bool valid2 = true;
			bool valid3 = true;
			for (vector<int>::iterator itpt = frontpts.begin(); itpt != frontpts.end(); itpt++) {
				if ((*it)->_v1 == (*itpt))
					valid1 = false;
				if ((*it)->_v2 == (*itpt))
					valid2 = false;
				if ((*it)->_v3 == (*itpt))
					valid3 = false;
			}

			if (valid1) {
				frontpts.push_back((*it)->_v1);
				filestrout << (*it)->_v1 << ", ";
			}
			if (valid2) {
				frontpts.push_back((*it)->_v2);
				filestrout << (*it)->_v2 << ", ";
			}
			if (valid3) {
				frontpts.push_back((*it)->_v3);
				filestrout << (*it)->_v3 << ", ";
			}*/

			//Affect texture values to Landmark points
			bool is_landmark  = false;
			for (int i = 0; i < 43; i++) {
				if (FACIAL_LANDMARKS[i] == (*it)->_v1)
					is_landmark = true;
			}
			if (is_landmark) {
				if (_vertices[(*it)->_v1]->_u != -1.0f && (_vertices[(*it)->_v1]->_u != _uvs[(*it)->_t1]->_u || _vertices[(*it)->_v1]->_v != _uvs[(*it)->_t1]->_v)) {
					cout << "Landmark: " << (*it)->_v1 << endl;
					cout << "u: " << _vertices[(*it)->_v1]->_u << "u: " << _vertices[(*it)->_v1]->_v << endl;
					cout << "u*: " << _uvs[(*it)->_t1]->_u << "u*: " << _uvs[(*it)->_t1]->_v << endl;
				}
				_vertices[(*it)->_v1]->_u = _uvs[(*it)->_t1]->_u;
				_vertices[(*it)->_v1]->_v = _uvs[(*it)->_t1]->_v;
			}
			is_landmark = false;
			for (int i = 0; i < 43; i++) {
				if (FACIAL_LANDMARKS[i] == (*it)->_v2)
					is_landmark = true;
			}
			if (is_landmark) {
				if (_vertices[(*it)->_v2]->_u != -1.0f && (_vertices[(*it)->_v2]->_u != _uvs[(*it)->_t2]->_u || _vertices[(*it)->_v2]->_v != _uvs[(*it)->_t2]->_v)) {
					cout << "Landmark: " << (*it)->_v2 << endl;
					cout << "u: " << _vertices[(*it)->_v2]->_u << "u: " << _vertices[(*it)->_v2]->_v << endl;
					cout << "u*: " << _uvs[(*it)->_t2]->_u << "u*: " << _uvs[(*it)->_t2]->_v << endl;
				}
				_vertices[(*it)->_v2]->_u = _uvs[(*it)->_t2]->_u;
				_vertices[(*it)->_v2]->_v = _uvs[(*it)->_t2]->_v;
			}
			is_landmark = false;
			for (int i = 0; i < 43; i++) {
				if (FACIAL_LANDMARKS[i] == (*it)->_v3)
					is_landmark = true;
			}
			if (is_landmark) {
				if (_vertices[(*it)->_v3]->_u != -1.0f && (_vertices[(*it)->_v3]->_u != _uvs[(*it)->_t3]->_u || _vertices[(*it)->_v3]->_v != _uvs[(*it)->_t3]->_v)) {
					cout << "Landmark: " << (*it)->_v3 << endl;
					cout << "u: " << _vertices[(*it)->_v3]->_u << "u: " << _vertices[(*it)->_v3]->_v << endl;
					cout << "u*: " << _uvs[(*it)->_t3]->_u << "u*: " << _uvs[(*it)->_t3]->_v << endl;
				}
				_vertices[(*it)->_v3]->_u = _uvs[(*it)->_t3]->_u;
				_vertices[(*it)->_v3]->_v = _uvs[(*it)->_t3]->_v;
			}

			bool add_ok1 = true;
			bool add_ok2 = true;
			bool add_ok3 = true;
			for (vector<int>::iterator it2 = _vertices[(*it)->_v1]->_neighboors.begin(); it2 != _vertices[(*it)->_v1]->_neighboors.end(); it2++) {
				if ((*it)->_v2 == (*it2))
					add_ok2 = false;
				if ((*it)->_v3 == (*it2))
					add_ok3 = false;
			}
			if (add_ok2)
				_vertices[(*it)->_v1]->_neighboors.push_back((*it)->_v2);
			if (add_ok3)
				_vertices[(*it)->_v1]->_neighboors.push_back((*it)->_v3);

			add_ok1 = true;
			add_ok3 = true;
			for (vector<int>::iterator it2 = _vertices[(*it)->_v2]->_neighboors.begin(); it2 != _vertices[(*it)->_v2]->_neighboors.end(); it2++) {
				if ((*it)->_v1 == (*it2))
					add_ok1 = false;
				if ((*it)->_v3 == (*it2))
					add_ok3 = false;
			}
			if (add_ok1)
				_vertices[(*it)->_v2]->_neighboors.push_back((*it)->_v1);
			if (add_ok3)
				_vertices[(*it)->_v2]->_neighboors.push_back((*it)->_v3);

			add_ok1 = true;
			add_ok2 = true;
			for (vector<int>::iterator it2 = _vertices[(*it)->_v3]->_neighboors.begin(); it2 != _vertices[(*it)->_v3]->_neighboors.end(); it2++) {
				if ((*it)->_v1 == (*it2))
					add_ok1 = false;
				if ((*it)->_v2 == (*it2))
					add_ok2 = false;
			}
			if (add_ok1)
				_vertices[(*it)->_v3]->_neighboors.push_back((*it)->_v1);
			if (add_ok2)
				_vertices[(*it)->_v3]->_neighboors.push_back((*it)->_v2);
		}
		//filestrout.close();
	}

	// set back point values to true
	for (int i = 0; i < 393; i++) {
		_vertices[BackIndices[i]]->_BackPoint = true;
	}
}

template<typename T>
void Mesh<T>::LoadS(string filename) {
	cout << "Loading " << filename << endl;
	string line;
	char tmpline[200];
	ifstream  filestr;

	filestr.open(filename, fstream::in);
	while (!filestr.is_open()) {
		cout << "Could not open " << filename << endl;
		return;
	}

	// load mesh
	float x, y, z;
	int v1, t1, n1, v2, t2, n2, v3, t3, n3;
	int j = 0, t = 0, nbVertices;
	char c[200];
	char c2[200];
	while (filestr.good())          // loop while extraction from file is possible
	{
		filestr.getline(tmpline, 256);
		sscanf_s(tmpline, "%s", c, _countof(c));

		if (strcmp(c, "#") == 0) {
			sscanf_s(tmpline, "%s %s %d", c, _countof(c), c2, _countof(c2), &nbVertices);

			if (strcmp(c2, "Vertices:") == 0) {
				_vertices.reserve(nbVertices);
				//cout << "nb Vertices: " << nbVertices << endl;
			}
			else if (strcmp(c2, "Faces:") == 0) {
				_triangles.reserve(nbVertices);
				//cout << "nb Faces: " << nbVertices << endl;
			}
			else if (strcmp(c2, "Uvs:") == 0) {
				_uvs.reserve(nbVertices);
				//cout << "nb Faces: " << nbVertices << endl;
			}
		}

		if (strcmp(c, "vn") == 0) {
			sscanf_s(tmpline, "%s %f %f %f", c, _countof(c), &x, &y, &z);
			_verticesList[j]._Nx = x;
			_verticesList[j]._Ny = y;
			_verticesList[j]._Nz = z;
			_verticesList[j]._indx = j;


			Point3D<T> *currpt = new Point3D<T>(x, y, z, j);
			currpt->_Nx = x;
			currpt->_Ny = y;
			currpt->_Nz = z;

			filestr.getline(tmpline, 256);
			sscanf_s(tmpline, "%s", c, _countof(c));
			if (strcmp(c, "v") == 0) {
				sscanf_s(tmpline, "%s %f %f %f", c, _countof(c), &x, &y, &z);
				_verticesList[j]._x = x;
				_verticesList[j]._y = y;
				_verticesList[j]._z = z;
				currpt->_x = x;
				currpt->_y = y;
				currpt->_z = z;
			}
			else {
				cout << "error reading mesh file" << endl;
			}

			_vertices.push_back(currpt);
			j++;
		}


		if (strcmp(c, "vt") == 0) {
			sscanf_s(tmpline, "%s %f %f", c, _countof(c), &x, &y);
			TextUV *currUV = new TextUV(x, y);
			_uvs.push_back(currUV);
		}

		if (strcmp(c, "f") == 0) {
			sscanf_s(tmpline, "%s %d/%d/%d %d/%d/%d %d/%d/%d", c, _countof(c), &v1, &t1, &n1, &v2, &t2, &n2, &v3, &t3, &n3);
			_trianglesList[t]._v1 = v1 - 1;
			_trianglesList[t]._v2 = v2 - 1;
			_trianglesList[t]._v3 = v3 - 1;

			_verticesList[v1 - 1]._u = _uvs[t1 - 1]->_u;
			_verticesList[v1 - 1]._v = _uvs[t1 - 1]->_v;

			_verticesList[v2 - 1]._u = _uvs[t2 - 1]->_u;
			_verticesList[v2 - 1]._v = _uvs[t2 - 1]->_v;

			_verticesList[v3 - 1]._u = _uvs[t3 - 1]->_u;
			_verticesList[v3 - 1]._v = _uvs[t3 - 1]->_v;

			_triangles.push_back(new Face(v1 - 1, t1 - 1, n1 - 1, v2 - 1, t2 - 1, n2 - 1, v3 - 1, t3 - 1, n3 - 1));
			t++;
		}
	}
	filestr.close();

	_verticesList[2702]._u = 115.0f / float(BumpHeight);
	_verticesList[2702]._v = 305.0f / float(BumpWidth);

	_verticesList[437]._u = 365.0f / float(BumpHeight);
	_verticesList[437]._v = 305.0f/ float(BumpWidth);

	for (vector<Face *>::iterator it = _triangles.begin(); it != _triangles.end(); it++) {
		if ((*it)->_v1 == 2702) {
			_uvs[(*it)->_t1]->_u = 115.0f / float(BumpHeight);
			_uvs[(*it)->_t1]->_v = 305.0f / float(BumpHeight);
		}

		if ((*it)->_v2 == 2702) {
			_uvs[(*it)->_t2]->_u = 115.0f / float(BumpHeight);
			_uvs[(*it)->_t2]->_v = 305.0f / float(BumpHeight);
		}

		if ((*it)->_v3 == 2702) {
			_uvs[(*it)->_t3]->_u = 115.0f / float(BumpHeight);
			_uvs[(*it)->_t3]->_v = 305.0f / float(BumpHeight);
		}

		if ((*it)->_v1 == 437) {
			_uvs[(*it)->_t1]->_u = 365.0f / float(BumpHeight);
			_uvs[(*it)->_t1]->_v = 305.0f / float(BumpHeight);
		}

		if ((*it)->_v2 == 437) {
			_uvs[(*it)->_t2]->_u = 365.0f / float(BumpHeight);
			_uvs[(*it)->_t2]->_v = 305.0f / float(BumpHeight);
		}

		if ((*it)->_v3 == 437) {
			_uvs[(*it)->_t3]->_u = 365.0f / float(BumpHeight);
			_uvs[(*it)->_t3]->_v = 305.0f / float(BumpHeight);
		}
	}

}

template<typename T>
void Mesh<T>::Write(string filename) {

	ofstream  filestr;

	filestr.open(filename, fstream::out);
	while (!filestr.is_open()) {
		cout << "Could not open MappingList" << endl;
		return;
	}

	filestr << "#### \n# \n# OBJ File Generated by Diego \n# \n#### \n# Object " << filename << endl;

	filestr << "# Vertices: " << _vertices.size() << endl;
	filestr << "# Faces: " << _triangles.size() << endl;
	filestr << "# Uvs: " << _uvs.size() << endl;

	for (vector<Point3D<T> *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		filestr << "v " << (*it)->_x << " " << (*it)->_y << " " << (*it)->_z << endl;
		filestr << "vn " << (*it)->_Nx << " " << (*it)->_Ny << " " << (*it)->_Nz << endl;
	}

	filestr << "# " << _vertices.size() << " vertices, " << _vertices.size() << " vertices normals \n \n \n" << endl;

	for (vector<TextUV *>::iterator it = _uvs.begin(); it != _uvs.end(); it++) {
		filestr << "vt " << (*it)->_u << " " << (*it)->_v << endl;
	}

	for (vector<Face *>::iterator it = _triangles.begin(); it != _triangles.end(); it++) {
		filestr << "f " << (*it)->_v1 + 1 << "/" << (*it)->_t1 + 1 << " "
			<< (*it)->_v2 + 1 << "/" << (*it)->_t2 + 1 << " " 
			<< (*it)->_v3 + 1 << "/" << (*it)->_t3 + 1 << " " << endl;
	}

	filestr << "# End of File" << endl;

	filestr.close();
}

// Scale mesh from file
template<typename T>
void Mesh<T>::Scale(T factor) {
	int i = 0;
	for (vector<MyPoint *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		(*it)->_x = (*it)->_x * factor;
		(*it)->_y = (*it)->_y * factor;
		(*it)->_z = (*it)->_z * factor;
		_verticesList[i]._x = _verticesList[i]._x * factor;
		_verticesList[i]._y = _verticesList[i]._y * factor;
		_verticesList[i]._z = _verticesList[i]._z * factor;

		i++;
	}
}

// Transform the mesh with input Transformation
template<typename T>
void Mesh<T>::Transform(Eigen::Matrix4d Transfo) {
	int i = 0;
	T tmp[3];
	for (vector<MyPoint *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {

		tmp[0] = (*it)->_x * Transfo(0, 0) + (*it)->_y * Transfo(0, 1) + (*it)->_z * Transfo(0, 2) + Transfo(0, 3);
		tmp[1] = (*it)->_x * Transfo(1, 0) + (*it)->_y * Transfo(1, 1) + (*it)->_z * Transfo(1, 2) + Transfo(1, 3);
		tmp[2] = (*it)->_x * Transfo(2, 0) + (*it)->_y * Transfo(2, 1) + (*it)->_z * Transfo(2, 2) + Transfo(2, 3);

		(*it)->_x = tmp[0];
		(*it)->_y = tmp[1];
		(*it)->_z = tmp[2];
		_verticesList[i]._x = tmp[0];
		_verticesList[i]._y = tmp[1];
		_verticesList[i]._z = tmp[2];

		tmp[0] = (*it)->_Nx * Transfo(0, 0) + (*it)->_Ny * Transfo(0, 1) + (*it)->_Nz * Transfo(0, 2);
		tmp[1] = (*it)->_Nx * Transfo(1, 0) + (*it)->_Ny * Transfo(1, 1) + (*it)->_Nz * Transfo(1, 2);
		tmp[2] = (*it)->_Nx * Transfo(2, 0) + (*it)->_Ny * Transfo(2, 1) + (*it)->_Nz * Transfo(2, 2);
		(*it)->_Nx = tmp[0];
		(*it)->_Ny = tmp[1];
		(*it)->_Nz = tmp[2];
		_verticesList[i]._Nx = tmp[0];
		_verticesList[i]._Ny = tmp[1];
		_verticesList[i]._Nz = tmp[2];
		i++;
	}
}

// Rotate the mesh with input rotation
template<typename T>
void Mesh<T>::Rotate(const cv::Mat& rot) {
	int i = 0;
	T tmp[3];
	for (vector<MyPoint *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		tmp[0] = (*it)->_x * rot.at<float>(0, 0) + (*it)->_y * rot.at<float>(0, 1) + (*it)->_z * rot.at<float>(0, 2);
		tmp[1] = (*it)->_x * rot.at<float>(1, 0) + (*it)->_y * rot.at<float>(1, 1) + (*it)->_z * rot.at<float>(1, 2);
		tmp[2] = (*it)->_x * rot.at<float>(2, 0) + (*it)->_y * rot.at<float>(2, 1) + (*it)->_z * rot.at<float>(2, 2);
		(*it)->_x = -tmp[0];
		(*it)->_y = -tmp[1];
		(*it)->_z = -tmp[2];
		_verticesList[i]._x = -tmp[0];
		_verticesList[i]._y = -tmp[1];
		_verticesList[i]._z = -tmp[2];

		tmp[0] = (*it)->_Nx * rot.at<float>(0, 0) + (*it)->_Ny * rot.at<float>(0, 1) + (*it)->_Nz * rot.at<float>(0, 2);
		tmp[1] = (*it)->_Nx * rot.at<float>(1, 0) + (*it)->_Ny * rot.at<float>(1, 1) + (*it)->_Nz * rot.at<float>(1, 2);
		tmp[2] = (*it)->_Nx * rot.at<float>(2, 0) + (*it)->_Ny * rot.at<float>(2, 1) + (*it)->_Nz * rot.at<float>(2, 2);
		(*it)->_Nx = -tmp[0];
		(*it)->_Ny = -tmp[1];
		(*it)->_Nz = -tmp[2];
		_verticesList[i]._Nx = -tmp[0];
		_verticesList[i]._Ny = -tmp[1];
		_verticesList[i]._Nz = -tmp[2];
		i++;
	}
}

// Translate the mesh with input translation vector
template<typename T>
void Mesh<T>::Translate(const cv::Point3f& xyz) {
	int i = 0;
	for (vector<MyPoint *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		(*it)->_x = (*it)->_x + xyz.x;
		(*it)->_y = (*it)->_y + xyz.y;
		(*it)->_z = (*it)->_z + xyz.z;
		_verticesList[i]._x = _verticesList[i]._x + xyz.x;
		_verticesList[i]._y = _verticesList[i]._y + xyz.y;
		_verticesList[i]._z = _verticesList[i]._z + xyz.z;
		i++;
	}
}

template<typename T>
void Mesh<T>::Draw(float *Bump) {
	glColor4f(1.0, 1.0, 1.0, 1.0);
	glBegin(GL_TRIANGLES);

	float tmpPt[3];
	MyPoint *s1, *s2, *s3;
	for (vector<Face *>::iterator it = _triangles.begin(); it != _triangles.end(); it++) {
		s1 = _vertices[(*it)->_v1];
		s2 = _vertices[(*it)->_v2];
		s3 = _vertices[(*it)->_v3];


		int indx_i = Myround(_verticesList[(*it)->_v1]._u*float(BumpHeight));
		int indx_j = Myround(_verticesList[(*it)->_v1]._v*float(BumpWidth));
		//if (Bump[(4 * (indx_i*BumpWidth + indx_j)) + 1] == 0.0)
		//	continue;

		indx_i = Myround(_verticesList[(*it)->_v2]._u*float(BumpHeight));
		indx_j = Myround(_verticesList[(*it)->_v2]._v*float(BumpWidth));
		//if (Bump[(4 * (indx_i*BumpWidth + indx_j)) + 1] == 0.0)
		//	continue;

		indx_i = Myround(_verticesList[(*it)->_v3]._u*float(BumpHeight));
		indx_j = Myround(_verticesList[(*it)->_v3]._v*float(BumpWidth));
		//if (Bump[(4 * (indx_i*BumpWidth + indx_j)) + 1] == 0.0)
		//	continue;
	
		indx_i = Myround(_verticesList[(*it)->_v1]._u*float(BumpHeight));
		indx_j = Myround(_verticesList[(*it)->_v1]._v*float(BumpWidth));
		float d = 0.0f;// Bump[4 * (indx_i*BumpWidth + indx_j)] / 1000.0f;
		tmpPt[0] = _verticesList[(*it)->_v1]._x + d*_verticesList[(*it)->_v1]._Nx;
		tmpPt[1] = _verticesList[(*it)->_v1]._y + d*_verticesList[(*it)->_v1]._Ny;
		tmpPt[2] = _verticesList[(*it)->_v1]._z + d*_verticesList[(*it)->_v1]._Nz;

		glNormal3f(s1->_Nx, s1->_Ny, s1->_Nz);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);

		indx_i = Myround(_verticesList[(*it)->_v2]._u*float(BumpHeight));
		indx_j = Myround(_verticesList[(*it)->_v2]._v*float(BumpWidth));
		d = 0.0f;//Bump[4 * (indx_i*BumpWidth + indx_j)] / 1000.0f;
		tmpPt[0] = _verticesList[(*it)->_v2]._x + d*_verticesList[(*it)->_v2]._Nx;
		tmpPt[1] = _verticesList[(*it)->_v2]._y + d*_verticesList[(*it)->_v2]._Ny;
		tmpPt[2] = _verticesList[(*it)->_v2]._z + d*_verticesList[(*it)->_v2]._Nz;
		glNormal3f(s2->_Nx, s2->_Ny, s2->_Nz);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);

		indx_i = Myround(_verticesList[(*it)->_v3]._u*float(BumpHeight));
		indx_j = Myround(_verticesList[(*it)->_v3]._v*float(BumpWidth));
		d = 0.0f;//Bump[4 * (indx_i*BumpWidth + indx_j)] / 1000.0f;
		tmpPt[0] = _verticesList[(*it)->_v3]._x + d*_verticesList[(*it)->_v3]._Nx;
		tmpPt[1] = _verticesList[(*it)->_v3]._y + d*_verticesList[(*it)->_v3]._Ny;
		tmpPt[2] = _verticesList[(*it)->_v3]._z + d*_verticesList[(*it)->_v3]._Nz;
		glNormal3f(s3->_Nx, s3->_Ny, s3->_Nz);
		glVertex3f(tmpPt[0], tmpPt[1], tmpPt[2]);
	}

	glEnd();
}

template<typename T>
void Mesh<T>::DrawLandmark(int i){

	MyPoint *Feat = _vertices[FACIAL_LANDMARKS[i]];

	glBegin(GL_POINTS);
	glColor4f(0.0, 1.0, 0.0, 1.0);

	glVertex3f(Feat->_x, Feat->_y, Feat->_z);

	glEnd();
}

template<typename T>
void Mesh<T>::AffectToTVal() {
	for (vector<Point3D<T> *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		(*it)->_Tx = (*it)->_x;
		(*it)->_Ty = (*it)->_y;
		(*it)->_Tz = (*it)->_z;
		(*it)->_TNx = (*it)->_Nx;
		(*it)->_TNy = (*it)->_Ny;
		(*it)->_TNz = (*it)->_Nz;
	}
}

template<typename T>
void Mesh<T>::AffectToTVector(Eigen::VectorXd *xres) {
	int indx = 0;
	for (vector<Point3D<T> *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		(*it)->_x = T((*xres)(3 * indx));
		(*it)->_y = T((*xres)(3 * indx + 1));
		(*it)->_z = T((*xres)(3 * indx + 2));

		(*it)->_Nx = 0.0f;
		(*it)->_Ny = 0.0f;
		(*it)->_Nz = 0.0f;

		_verticesList[indx]._x = T((*xres)(3 * indx));
		_verticesList[indx]._y = T((*xres)(3 * indx + 1));
		_verticesList[indx]._z = T((*xres)(3 * indx + 2));
		_verticesList[indx]._Nx = 0.0f;
		_verticesList[indx]._Ny = 0.0f;
		_verticesList[indx]._Nz = 0.0f;

		indx++;
	}

	/***************Compute normals*******************/

	T v1[3];
	T v2[3];
	T nmle[3];
	T nrm;
	Face *CurrFace;
	for (vector<Face *>::iterator it = _triangles.begin(); it != _triangles.end(); it++) {
		CurrFace = (*it);

		// Compute normal and weight of the face
		v1[0] = _vertices[CurrFace->_v2]->_x - _vertices[CurrFace->_v1]->_x;
		v1[1] = _vertices[CurrFace->_v2]->_y - _vertices[CurrFace->_v1]->_y;
		v1[2] = _vertices[CurrFace->_v2]->_z - _vertices[CurrFace->_v1]->_z;

		v2[0] = _vertices[CurrFace->_v3]->_x - _vertices[CurrFace->_v1]->_x;
		v2[1] = _vertices[CurrFace->_v3]->_y - _vertices[CurrFace->_v1]->_y;
		v2[2] = _vertices[CurrFace->_v3]->_z - _vertices[CurrFace->_v1]->_z;

		nmle[0] = v1[1] * v2[2] - v1[2] * v2[1];
		nmle[1] = -v1[0] * v2[2] + v1[2] * v2[0];
		nmle[2] = v1[0] * v2[1] - v1[1] * v2[0];

		nrm = sqrt(nmle[0] * nmle[0] + nmle[1] * nmle[1] + nmle[2] * nmle[2]);
		nmle[0] = (nmle[0] / nrm);
		nmle[1] = (nmle[1] / nrm);
		nmle[2] = (nmle[2] / nrm);

		_vertices[(*it)->_v1]->_Nx = _vertices[(*it)->_v1]->_Nx + nmle[0];
		_vertices[(*it)->_v1]->_Ny = _vertices[(*it)->_v1]->_Ny + nmle[1];
		_vertices[(*it)->_v1]->_Nz = _vertices[(*it)->_v1]->_Nz + nmle[2];

		_vertices[(*it)->_v2]->_Nx = _vertices[(*it)->_v2]->_Nx + nmle[0];
		_vertices[(*it)->_v2]->_Ny = _vertices[(*it)->_v2]->_Ny + nmle[1];
		_vertices[(*it)->_v2]->_Nz = _vertices[(*it)->_v2]->_Nz + nmle[2];

		_vertices[(*it)->_v3]->_Nx = _vertices[(*it)->_v3]->_Nx + nmle[0];
		_vertices[(*it)->_v3]->_Ny = _vertices[(*it)->_v3]->_Ny + nmle[1];
		_vertices[(*it)->_v3]->_Nz = _vertices[(*it)->_v3]->_Nz + nmle[2];
	}

	indx = 0;
	for (vector<Point3D<T> *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		float nrm = sqrt((*it)->_Nx * (*it)->_Nx + (*it)->_Ny * (*it)->_Ny + (*it)->_Nz * (*it)->_Nz);
		if (nrm != 0.0) {
			(*it)->_Nx = (*it)->_Nx / nrm;
			(*it)->_Ny = (*it)->_Ny / nrm;
			(*it)->_Nz = (*it)->_Nz / nrm; 
			_verticesList[indx]._Nx = (*it)->_Nx;
			_verticesList[indx]._Ny = (*it)->_Ny;
			_verticesList[indx]._Nz = (*it)->_Nz;
		}
		indx++;
	}
}

template<typename T>
void Mesh<T>::AffectToTVectorT(Eigen::VectorXd *xres) {
	int indx = 0;
	for (vector<Point3D<T> *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		(*it)->_Tx = T((*xres)(3 * indx));
		(*it)->_Ty = T((*xres)(3 * indx + 1));
		(*it)->_Tz = T((*xres)(3 * indx + 2));

		(*it)->_TNx = 0.0f;
		(*it)->_TNy = 0.0f;
		(*it)->_TNz = 0.0f;

		_verticesList[indx]._Tx = T((*xres)(3 * indx));
		_verticesList[indx]._Ty = T((*xres)(3 * indx + 1));
		_verticesList[indx]._Tz = T((*xres)(3 * indx + 2));
		_verticesList[indx]._TNx = 0.0f;
		_verticesList[indx]._TNy = 0.0f;
		_verticesList[indx]._TNz = 0.0f;

		indx++;
	}

	/***************Compute normals*******************/

	T v1[3];
	T v2[3];
	T nmle[3];
	T nrm;
	Face *CurrFace;
	for (vector<Face *>::iterator it = _triangles.begin(); it != _triangles.end(); it++) {
		CurrFace = (*it);

		// Compute normal and weight of the face
		v1[0] = _vertices[CurrFace->_v2]->_Tx - _vertices[CurrFace->_v1]->_Tx;
		v1[1] = _vertices[CurrFace->_v2]->_Ty - _vertices[CurrFace->_v1]->_Ty;
		v1[2] = _vertices[CurrFace->_v2]->_Tz - _vertices[CurrFace->_v1]->_Tz;

		v2[0] = _vertices[CurrFace->_v3]->_Tx - _vertices[CurrFace->_v1]->_Tx;
		v2[1] = _vertices[CurrFace->_v3]->_Ty - _vertices[CurrFace->_v1]->_Ty;
		v2[2] = _vertices[CurrFace->_v3]->_Tz - _vertices[CurrFace->_v1]->_Tz;

		nmle[0] = v1[1] * v2[2] - v1[2] * v2[1];
		nmle[1] = -v1[0] * v2[2] + v1[2] * v2[0];
		nmle[2] = v1[0] * v2[1] - v1[1] * v2[0];

		nrm = sqrt(nmle[0] * nmle[0] + nmle[1] * nmle[1] + nmle[2] * nmle[2]);
		nmle[0] = (nmle[0] / nrm);
		nmle[1] = (nmle[1] / nrm);
		nmle[2] = (nmle[2] / nrm);

		_vertices[(*it)->_v1]->_TNx = _vertices[(*it)->_v1]->_TNx + nmle[0];
		_vertices[(*it)->_v1]->_TNy = _vertices[(*it)->_v1]->_TNy + nmle[1];
		_vertices[(*it)->_v1]->_TNz = _vertices[(*it)->_v1]->_TNz + nmle[2];

		_vertices[(*it)->_v2]->_TNx = _vertices[(*it)->_v2]->_TNx + nmle[0];
		_vertices[(*it)->_v2]->_TNy = _vertices[(*it)->_v2]->_TNy + nmle[1];
		_vertices[(*it)->_v2]->_TNz = _vertices[(*it)->_v2]->_TNz + nmle[2];

		_vertices[(*it)->_v3]->_TNx = _vertices[(*it)->_v3]->_TNx + nmle[0];
		_vertices[(*it)->_v3]->_TNy = _vertices[(*it)->_v3]->_TNy + nmle[1];
		_vertices[(*it)->_v3]->_TNz = _vertices[(*it)->_v3]->_TNz + nmle[2];
	}

	for (vector<Point3D<T> *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		float nrm = sqrt((*it)->_TNx * (*it)->_TNx + (*it)->_TNy * (*it)->_TNy + (*it)->_TNz * (*it)->_TNz);
		if (nrm != 0.0) {
			(*it)->_TNx = (*it)->_TNx / nrm;
			(*it)->_TNy = (*it)->_TNy / nrm;
			(*it)->_TNz = (*it)->_TNz / nrm;
		}
	}
}

template<typename T>
void Mesh<T>::ComputeTgtPlane() {

	int indx = 0;
	MyPoint *pt1, *pt2;
	T e1[3], e2[3], e3[3];
	T Te1[3], Te2[3], Te3[3];
	T proj, nrme;
	for (vector<MyPoint *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		pt1 = (*it);
		// Compute Original and transformed tangent basis
		if (pt1->_neighboors.empty())
			cout << "error no voisins" << endl;

		//compute original basis, centered on pt1, with z nmle and oriented in p1->p2
		e3[0] = pt1->_Nx;
		e3[1] = pt1->_Ny;
		e3[2] = pt1->_Nz;

		pt2 = _vertices[pt1->_neighboors[0]];

		e1[0] = pt2->_x - pt1->_x;
		e1[1] = pt2->_y - pt1->_y;
		e1[2] = pt2->_z - pt1->_z;
		proj = e1[0] * e3[0] + e1[1] * e3[1] + e1[2] * e3[2];
		e1[0] = e1[0] - proj*e3[0];
		e1[1] = e1[1] - proj*e3[1];
		e1[2] = e1[2] - proj*e3[2];
		nrme = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
		e1[0] = e1[0] / nrme;
		e1[1] = e1[1] / nrme;
		e1[2] = e1[2] / nrme;

		e2[0] = e3[1] * e1[2] - e3[2] * e1[1];
		e2[1] = -(e3[0] * e1[2] - e3[2] * e1[0]);
		e2[2] = e3[0] * e1[1] - e3[1] * e1[0];

		nrme = sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
		e2[0] = e2[0] / nrme;
		e2[1] = e2[1] / nrme;
		e2[2] = e2[2] / nrme;

		//compute transformed basis, centered on pt1', with z nmle and oriented in p1'->p2'

		Te3[0] = pt1->_TNx;
		Te3[1] = pt1->_TNy;
		Te3[2] = pt1->_TNz;

		Te1[0] = pt2->_Tx - pt1->_Tx;
		Te1[1] = pt2->_Ty - pt1->_Ty;
		Te1[2] = pt2->_Tz - pt1->_Tz;
		proj = Te1[0] * Te3[0] + Te1[1] * Te3[1] + Te1[2] * Te3[2];
		Te1[0] = Te1[0] - proj*Te3[0];
		Te1[1] = Te1[1] - proj*Te3[1];
		Te1[2] = Te1[2] - proj*Te3[2];
		nrme = sqrt(Te1[0] * Te1[0] + Te1[1] * Te1[1] + Te1[2] * Te1[2]);
		Te1[0] = Te1[0] / nrme;
		Te1[1] = Te1[1] / nrme;
		Te1[2] = Te1[2] / nrme;

		Te2[0] = Te3[1] * Te1[2] - Te3[2] * Te1[1];
		Te2[1] = -(Te3[0] * Te1[2] - Te3[2] * Te1[0]);
		Te2[2] = Te3[0] * Te1[1] - Te3[1] * Te1[0];

		nrme = sqrt(Te2[0] * Te2[0] + Te2[1] * Te2[1] + Te2[2] * Te2[2]);
		Te2[0] = Te2[0] / nrme;
		Te2[1] = Te2[1] / nrme;
		Te2[2] = Te2[2] / nrme;

		//Compute Rotation matrix
		Eigen::Matrix3f B1;
		B1(0, 0) = e1[0]; B1(0, 1) = e2[0]; B1(0, 2) = e3[0];
		B1(1, 0) = e1[1]; B1(1, 1) = e2[1]; B1(1, 2) = e3[1];
		B1(2, 0) = e1[2]; B1(2, 1) = e2[2]; B1(2, 2) = e3[2];

		Eigen::Matrix3f TB1;
		TB1(0, 0) = Te1[0]; TB1(0, 1) = Te2[0]; TB1(0, 2) = Te3[0];
		TB1(1, 0) = Te1[1]; TB1(1, 1) = Te2[1]; TB1(1, 2) = Te3[1];
		TB1(2, 0) = Te1[2]; TB1(2, 1) = Te2[2]; TB1(2, 2) = Te3[2];

		Eigen::Matrix3f Rot = TB1 * B1.inverse();

		if (Rot(0, 0) != Rot(0, 0))
			cout << Rot << endl;

		_quaternions[indx][0] = Rot(0, 0); _quaternions[indx][3] = Rot(0, 1); _quaternions[indx][6] = Rot(0, 2);
		_quaternions[indx][1] = Rot(1, 0); _quaternions[indx][4] = Rot(1, 1); _quaternions[indx][7] = Rot(1, 2);
		_quaternions[indx][2] = Rot(2, 0); _quaternions[indx][5] = Rot(2, 1); _quaternions[indx][8] = Rot(2, 2);

		indx++;
	}
}

template<typename T>
void Mesh<T>::PopulateMatrix(vector<TrplType> *tripletList, Eigen::VectorXd *b1, int offfset) {
	//int offfset = 3 * 43;
	int indx = 0;
	int indx2 = 0;
	MyPoint *pt1;
	for (vector<MyPoint *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		pt1 = (*it);

		/// Go through neighboors of all vertice
		for (vector<int>::iterator it2 = (*it)->_neighboors.begin(); it2 != (*it)->_neighboors.end(); it2++) {

			(*tripletList).push_back(TrplType(offfset + 3 * indx2, 3 * indx, -1.0));
			(*tripletList).push_back(TrplType(offfset + 3 * indx2 + 1, 3 * indx + 1, -1.0));
			(*tripletList).push_back(TrplType(offfset + 3 * indx2 + 2, 3 * indx + 2, -1.0));


			(*tripletList).push_back(TrplType(offfset + 3 * indx2, 3 * (*it2), 1.0));
			(*tripletList).push_back(TrplType(offfset + 3 * indx2 + 1, 3 * (*it2) + 1, 1.0));
			(*tripletList).push_back(TrplType(offfset + 3 * indx2 + 2, 3 * (*it2) + 2, 1.0));


			T ptv[3];
			ptv[0] = _vertices[(*it2)]->_x - pt1->_x;
			ptv[1] = _vertices[(*it2)]->_y - pt1->_y;
			ptv[2] = _vertices[(*it2)]->_z - pt1->_z;

			//Rotate with the quaternion
			T Rptv[3];
			Rot<T>(Rptv, ptv, _quaternions[indx]);

			(*b1)(offfset + 3 * indx2) = double(Rptv[0]);
			(*b1)(offfset + 3 * indx2 + 1) = double(Rptv[1]);
			(*b1)(offfset + 3 * indx2 + 2) = double(Rptv[2]);

			indx2++;
		}
		indx++;
	}
}

template<typename T>
int Mesh<T>::sizeV() {
	int res = 0;
	for (vector<MyPoint *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		res += (*it)->_neighboors.size();
	}
	return res;
}

template<typename T>
void Mesh<T>::Deform(Eigen::VectorXd *boV) {
	int indx = 0;
	for (vector<Point3D<T> *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
		(*it)->_x = (*it)->_Tx;
		(*it)->_y = (*it)->_Ty;
		(*it)->_z = (*it)->_Tz;
		(*it)->_Nx = (*it)->_TNx;
		(*it)->_Ny = (*it)->_TNy;
		(*it)->_Nz = (*it)->_TNz; 
		(*boV)(3 * indx) = double((*it)->_x);
		(*boV)(3 * indx + 1) = double((*it)->_y);
		(*boV)(3 * indx + 2) = double((*it)->_z);

		_verticesList[indx]._x = (*it)->_Tx;
		_verticesList[indx]._y = (*it)->_Ty;
		_verticesList[indx]._z = (*it)->_Tz;
		_verticesList[indx]._Nx = (*it)->_TNx;
		_verticesList[indx]._Ny = (*it)->_TNy;
		_verticesList[indx]._Nz = (*it)->_TNz;

		indx++;
	}
}

template<typename T>
void Mesh<T>::Map(Mesh<float> *M1) {

	cout << "nb vertices: " << _vertices.size() << endl;
	ofstream  filestr;

	filestr.open("MappingList.txt", fstream::out);
	while (!filestr.is_open()) {
		cout << "Could not open MappingList" << endl;
		return;
	}

	MyPoint *currPt;
	float dist;
	for (int i = 0; i < 3083; i++) {
		//currPt = _vertices[FACIAL_LANDMARKS[i]];
		currPt = _vertices[BackIndices[i]];
		int indx2 = 0;
		bool found = false;
		for (vector<Point3D<float> *>::iterator it2 = M1->_vertices.begin(); it2 != M1->_vertices.end(); it2++) {
				
			dist = sqrt((currPt->_x - (*it2)->_x)*(currPt->_x - (*it2)->_x) 
				+ (currPt->_y - (*it2)->_y)*(currPt->_y - (*it2)->_y)
				+ (currPt->_z - (*it2)->_z)*(currPt->_z - (*it2)->_z));

			if (dist < 0.001f) {
				filestr << indx2 << ", ";
				//_indxList.push_back(indx1);
				_indxList.push_back(indx2);
				found = true;
				break;
			}
			indx2++;
		}
		if (!found) {
			cout << i << endl;
		}
	}
	filestr << _indxList.size() << ", ";
	int tmp;
	cin >> tmp;
	


	//int indx1 = 0;
	//float dist;
	//for (vector<Point3D<T> *>::iterator it = _vertices.begin(); it != _vertices.end(); it++) {
	//	//Search closest point in M1
	//	int indx2 = 0;
	//	bool found = false;
	//	for (vector<Point3D<float> *>::iterator it2 = M1->_vertices.begin(); it2 != M1->_vertices.end(); it2++) {
	//		
	//		dist = sqrt(((*it)->_x - (*it2)->_x)*((*it)->_x - (*it2)->_x) + ((*it)->_y - (*it2)->_y)*((*it)->_y - (*it2)->_y)
	//			+ ((*it)->_z - (*it2)->_z)*((*it)->_z - (*it2)->_z));

	//		if (dist < 0.001f) {
	//			filestr << indx1 << " " << indx2 << endl;
	//			//_indxList.push_back(indx1);
	//			_indxList.push_back(indx2);
	//			found = true;
	//			break;
	//		}
	//		indx2++;
	//	}
	//	if (!found) {
	//		_indxList.push_back(indx1);
	//		cout << indx1 << endl;
	//	}
	//	indx1++;
	//}

	filestr.close();
}

template<typename T>
void Mesh<T>::Modify(Mesh<float> *M1) {
	/*vector<Point3D<T> *> verticesTmp;

	for (int i = 0; i < _vertices.size(); i++) {
		verticesTmp.push_back(M1->_vertices[_indxList[i]]);
	}
*/
	/*verticesTmp.push_back(_vertices[437]);

	for (int i = 438; i < 2702; i++) {
		verticesTmp.push_back(M1->_vertices[_indxList[i]]);
	}

	verticesTmp.push_back(_vertices[2702]);

	for (int i = 2703; i < _vertices.size(); i++) {
		verticesTmp.push_back(M1->_vertices[_indxList[i]]);
	}*/

	//M1->_vertices.clear();
	//M1->_vertices = verticesTmp;
	M1->_uvs = _uvs;
	M1->_triangles = _triangles;
}


#endif