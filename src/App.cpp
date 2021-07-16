// ConsoleApplication1.cpp : Defines the entry point for the console application.
//

#include "App.h"
#include "FaceCap.h"
#include "Mesh.h"

#define NB_THREADS 1

namespace {

#ifdef __ANDROID__
	//float Calib[11] = { 580.8857, 583.317, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8000.0 }; // depth sensor
#else
	float Calib[11] = { 580.8857, 583.317, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8000.0 }; // Kinect data
																								 //float Calib[11] = { 580.8857, 583.317, 319.5, 239.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 8000.0 }; // Kinect data
#endif
}

GLdouble eyeBallCenterScale = 5.0f;

/*** Camera variables for OpenGL ***/
GLfloat intrinsics[16] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
float Znear = 0.05f;
float Zfar = 10.0f;
GLfloat light_pos[] = { 0.0, 0.0, -2.0, 0.0 }; //{ 1.0, 1.0, 0.0, 0.0 };
GLfloat ambientLight[] = { 0.2f, 0.2f, 0.2f, 1.0f };
GLfloat diffuseLight[] = { 0.8f, 0.8f, 0.8f, 1.0f };

// angle of rotation for the camera direction
float anglex = 0.0f;
float angley = 0.0f;

// actual vector representing the camera's direction
float lx = 0.0f, ly = 0.0f, lz = 1.0f;
float lxStrap = -1.0f, lyStrap = 0.0f, lzStrap = 0.0f;

// XZ position of the camera
float x = 0.0f, y = 0.0f, z = -0.1f; //0.15f;//
float deltaAnglex = 0.0f;
float deltaAngley = 0.0f;
float deltaMove = 0;
float deltaStrap = 0;
int xOrigin = -1;
int yOrigin = -1;

//bool Running = false;
bool stop = false;
clock_t current_time_prod, current_time_csm, current_time_disp;
clock_t last_time_prod, last_time_csm, last_time_disp;
float my_count_prod, my_count_csm, my_count_disp;
float fps_prod, fps_csm, fps_disp;
bool first;
int anim_indx = 0;
bool save_img = false;
bool inverted = true;

FaceCap *faceCap = NULL;
cv::CascadeClassifier face_cascade;
//cv::CascadeClassifier face_cascadeRe;

mutex imageMutex;
mutex bumpMutex;
condition_variable condv;

std::vector<MyMesh *> Blendshape;
std::vector<MyMesh *> eyeBall;

thread t1;
thread t2;
thread t3;
thread t4;
thread t5;
thread t6;
thread t_csm[NB_THREADS*NB_THREADS];
int idim = 0;
bool color = true;
bool bump = true;
bool quad = true;
bool ready_to_bump = false;

bool terminated;
bool isTerminated() { return terminated; }

int INDX_PROD = 0;
int INDX_CSM = 0;

// prototype 
void produce();
void consume();
void stream();
void saveInput(int k);

App::App()
{

}

App::~App()
{

}

bool App::setup()
{
	// setup OpenGL
	if (setupOpenGL() == false) {
		std::cout << "failed to setup OpenGL" << std::endl;
		return false;
	}

	// Load the cascade for facial detection
	string faceDetectionModel("..\\data\\models\\haarcascade_frontalface_alt2.xml");
	if (!face_cascade.load(faceDetectionModel))
	{
		cerr << "error loading cascade\n";
		throw runtime_error("Error loading face detection model.");
	}

	// setup 3D facial data
	setupFacialModel();

	// initialize each variables
	current_time_prod = clock();
	last_time_prod = clock();
	my_count_prod = 0.0;
	current_time_csm = clock();
	last_time_csm = clock();
	my_count_csm = 0.0;
	current_time_disp = clock();
	last_time_disp = clock();
	my_count_disp = 0.0;

	return true;
}

void App::captureOnline(bool bSave, char *path)
{
	// initialize Depth Sensor
	if (initDepthSensor()) {

		//Running = true;
		/*** Live Kinect data ***/
		intrinsics[0] = 2.0*580.8857 / 640.0; intrinsics[5] = 2.0*583.317 / 480.0;
		//intrinsics[0] = 2.0*516.8 / 640.0; intrinsics[5] = 2.0*519.6 / 480.0;
		intrinsics[8] = 0.0;
		intrinsics[9] = 0.0;
		intrinsics[10] = -(Zfar + Znear) / (Zfar - Znear);
		intrinsics[11] = -1.0; intrinsics[14] = -2.0*(Zfar*Znear) / (Zfar - Znear);

		if (!bSave) {
			cout << "ONLINE WITHOUT SAVE" << endl;

			first = true;
			faceCap->SetParam(Calib, "", true);

			t1 = thread(produce);
			t2 = thread(consume);
		}
		else {
			cout << "ONLINE WITH SAVE" << endl;
			faceCap->SetParam(Calib, path, true);

			t1 = thread(stream);
			t2 = thread(saveInput, 0);
			t3 = thread(saveInput, 1);
		}

	}
	else {
		cerr << "Can't find depth sensor" << endl;
	}
}

void App::captureOffline(char *path)
{
	cout << "Offline Process" << endl;

	/*** Offline Kinect data ***/
	//Running = false;

	intrinsics[0] = 2.0*580.8857 / 640.0; intrinsics[5] = 2.0*583.317 / 480.0;
	//intrinsics[0] = 2.0*516.8 / 640.0; intrinsics[5] = 2.0*519.6 / 480.0;
	intrinsics[8] = 0.0;
	intrinsics[9] = 0.0;
	intrinsics[10] = -(Zfar + Znear) / (Zfar - Znear);
	intrinsics[11] = -1.0; intrinsics[14] = -2.0*(Zfar*Znear) / (Zfar - Znear);

	first = true;

	faceCap->SetParam(Calib, path, false);

	t1 = thread(produce);
	t2 = thread(consume);
}

void App::playback()
{
	t1 = thread(stream);
	t2 = thread(saveInput, 0);
	t3 = thread(saveInput, 1);
}

void App::update()
{
	if (deltaMove || deltaStrap)
		computePos(deltaMove, deltaStrap);
}

void App::draw() {

	if (faceCap) {
		display();
	}
	else {
		glutSwapBuffers();
		glutPostRedisplay();
	}
}

void App::exit()
{
	delete faceCap;
	std::exit(1);
}

void App::resize(int width_in, int height_in)
{
	glViewport(0, 0, width_in, height_in);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix
														// Set up camera intrinsics
	glLoadMatrixf(intrinsics);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void App::computePos(float deltaMove, float deltaStrap) {

	x += deltaMove * lx * 0.1f + deltaStrap * lxStrap * 0.1f;
	y += deltaMove * ly * 0.1f + deltaStrap * lyStrap * 0.1f;
	z += deltaMove * lz * 0.1f + deltaStrap * lzStrap * 0.1f;
}

void App::saveImg(int x, int y, int id, char *path) {
	float *image = new float[3 * cDepthWidth*cDepthHeight];
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadBuffer(GL_FRONT);
	glReadPixels(x, y, cDepthWidth, cDepthHeight, GL_RGB, GL_FLOAT, image);

	cv::Mat imagetest(cDepthHeight, cDepthWidth, CV_8UC3);
	for (int i = 0; i < cDepthHeight; i++) {
		for (int j = 0; j < cDepthWidth; j++) {
			imagetest.at<cv::Vec3b>(cDepthHeight - 1 - i, j)[2] = unsigned char(255.0*image[3 * (i*cDepthWidth + j)]);
			imagetest.at<cv::Vec3b>(cDepthHeight - 1 - i, j)[1] = unsigned char(255.0*image[3 * (i*cDepthWidth + j) + 1]);
			imagetest.at<cv::Vec3b>(cDepthHeight - 1 - i, j)[0] = unsigned char(255.0*image[3 * (i*cDepthWidth + j) + 2]);
		}
	}

	char filename[100];
	sprintf_s(filename, "%s%d.png", path, id);
	cv::imwrite(filename, imagetest);

	delete[] image;
	image = 0;
}

void App::display(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix
														// Set up camera intrinsics
	glLoadMatrixf(intrinsics);

	glViewport(cDepthWidth, cDepthHeight, cDepthWidth, cDepthHeight);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Set the camera
	gluLookAt(x, y, z,
		x + lx, y + ly, z + lz,
		0.0f, -1.0f, 0.0f);

	glEnable(GL_LIGHTING);
	GLfloat ambientLightq[] = { 0.2f, 0.2f, 0.2f, 1.0f };
	GLfloat diffuseLightq[] = { 0.4f, 0.4f, 0.4f, 1.0f };
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLightq);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLightq);
	glEnable(GL_LIGHT0);


	glDepthFunc(GL_LEQUAL);
	glDisable(GL_BLEND);


	glViewport(0, 0, cDepthWidth, cDepthHeight); // Normal image
												 //faceCap->DrawVBO(false);
												 //if (quad)
	faceCap->DrawQuad(0, bump);
	//else
	//	faceCap->Draw(color, bump);

	glDepthFunc(GL_LESS);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glViewport(0, 0, cDepthWidth, cDepthHeight);
	if (INDX_CSM > 0) {
		faceCap->DrawRect(true);
	}

	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHTING);

	if (save_img) {

		cout << "start playback" << endl;
		char filename_buff[100];
		sprintf_s(filename_buff, "Seq\\res-Kinect1-f\\Output\\Normal", dest_name);
		saveImg(0, 0, idim, filename_buff);
		cout << "end playback" << endl;
		/*sprintf_s(filename_buff, "Seq\\BLENDSHAPES\\PlayBack\\RGB", dest_name);
		saveimg(cDepthWidth, 0, idim, filename_buff);
		sprintf_s(filename_buff, "Seq\\BLENDSHAPES\\PlayBack\\Geo", dest_name);
		saveimg(2 * cDepthWidth, 0, idim, filename_buff);
		sprintf_s(filename_buff, "Seq\\BLENDSHAPES\\PlayBack\\Original", dest_name);
		saveimg(2 * cDepthWidth, cDepthHeight, idim, filename_buff);*/

		/*sprintf_s(filename_buff, "%s\\Retargeted\\Normal", dest_name);
		saveimg(0, 0, idim, filename_buff);
		sprintf_s(filename_buff, "%s\\Retargeted\\RGB", dest_name);
		saveimg(cDepthWidth, 0, idim, filename_buff);
		sprintf_s(filename_buff, "%s\\Retargeted\\Geo", dest_name);
		saveimg(2 * cDepthWidth, 0, idim, filename_buff);
		sprintf_s(filename_buff, "%s\\Input\\RGB", dest_name);
		saveimg(0, cDepthHeight, idim, filename_buff);
		sprintf_s(filename_buff, "%s\\Input\\Normal", dest_name);
		saveimg(cDepthWidth, cDepthHeight, idim, filename_buff);*/
		idim++;
		save_img = false;
		anim_indx = (anim_indx + 1) % 28;
	}

	my_count_disp++;
	current_time_disp = clock();
	if ((current_time_disp - last_time_disp) / CLOCKS_PER_SEC > 1.0) {
		fps_disp = my_count_disp / ((current_time_disp - last_time_disp) / CLOCKS_PER_SEC);
		last_time_disp = current_time_disp;
		my_count_disp = 0.0;
		cout << "fps display: " << fps_disp << endl;
	}

	glFlush();
	glFinish();
	glutPostRedisplay();
	return;
}

void App::keyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 'c':
		color = !color;
		break;
	case 'b':
		bump = !bump;
		break;
	case 'q':
		quad = !quad;
		break;
	case 'x':
		eyeBallCenterScale -= 0.001f;
		break;
	case 'z':
		eyeBallCenterScale += 0.001f;
		break;
	case 27 /* Esc */:
		terminated = true;
		//if (Running) {
		//	try {
		//		//t1.join();
		//		//t2.join();
		//		//t3.join();
		//	}
		//	catch (exception& e) {
		//		cerr << e.what() << endl;
		//	}
		//}
		delete faceCap;
		std::exit(1);
	}
	// 
}

void App::keyUp(unsigned char key, int x, int y) {
	switch (key) {
	case 'a':
		break;
	default:
		break;
	}
}

void App::pressKey(int key, int xx, int yy) {

	switch (key) {
	case GLUT_KEY_UP: deltaMove = 0.5f / 3.0f/*(fps/30.0)*/; break;
	case GLUT_KEY_DOWN: deltaMove = -0.5f / 3.0f/*(fps/30.0)*/; break;
	case GLUT_KEY_LEFT: deltaStrap = 0.5f / 3.0f/*(fps/30.0)*/; break;
	case GLUT_KEY_RIGHT: deltaStrap = -0.5f / 3.0f/*(fps/30.0)*/; break;
	}
}

void App::releaseKey(int key, int x, int y) {

	switch (key) {
	case GLUT_KEY_LEFT:
	case GLUT_KEY_RIGHT:
	case GLUT_KEY_UP:
	case GLUT_KEY_DOWN: deltaMove = 0; deltaStrap = 0; break;
	}
}

void App::mouseMove(int x, int y) {

	// this will only be true when the left button is down
	if (xOrigin >= 0 || yOrigin >= 0) {

		// update deltaAngle
		deltaAnglex = (x - xOrigin) * 0.001f;
		deltaAngley = (y - yOrigin) * 0.001f;

		// update camera's direction
		lx = sin(anglex + deltaAnglex);
		ly = -cos(anglex + deltaAnglex) * sin(-(angley + deltaAngley));
		lz = cos(anglex + deltaAnglex) * cos(-(angley + deltaAngley));

		// update camera's direction
		lxStrap = -cos(anglex + deltaAnglex);
		lyStrap = -sin(anglex + deltaAnglex) * sin(-(angley + deltaAngley));
		lzStrap = sin(anglex + deltaAnglex) * cos(-(angley + deltaAngley));
	}
}

void App::mouseButton(int button, int state, int x, int y) {

	// only start motion if the left button is pressed
	if (button == GLUT_LEFT_BUTTON) {
		// when the button is released
		if (state == GLUT_UP) {
			anglex += deltaAnglex;
			angley += deltaAngley;
			xOrigin = -1;
			yOrigin = -1;
		}
		else {// state = GLUT_DOWN
			xOrigin = x;
			yOrigin = y;
		}
	}
}

/***** Function to handle right click of Mouse for subwindow 1*****/
bool App::initDepthSensor()
{
	if (!faceCap->StartSensor()) {
		return false;
	};
	return true;
}

void GenerateBumpMng(int x, int y, int width, int height) {
	faceCap->GenerateBump(Blendshape, x, y, width, height);
}

void produce()
{
	while (!isTerminated()) {

		int res = faceCap->Update();

		if (res == 3) { // end of sequence
			delete faceCap;
			exit(1);
		}

		if (res == 2) {
			faceCap->DetectFeatures(&face_cascade);
			//faceCap->SuperpixelSegmentation();
		}

		while ((INDX_PROD % 10) == (INDX_CSM % 10) && INDX_PROD > INDX_CSM) {
			cv::waitKey(1);
		}

		faceCap->Compute3DDataCPU(INDX_PROD % 10);

		INDX_PROD++;

		my_count_prod++;
		current_time_prod = clock();
		if ((current_time_prod - last_time_prod) / CLOCKS_PER_SEC > 1.0) {
			fps_prod = my_count_prod / ((current_time_prod - last_time_prod) / CLOCKS_PER_SEC);
			last_time_prod = current_time_prod;
			my_count_prod = 0.0;
			cout << "fps produce: " << fps_prod << endl;
		}
	}
}

void consume() {
	while (!isTerminated()) {

		while (INDX_CSM == INDX_PROD) {
			cv::waitKey(1);
		}

		assert(INDX_CSM < INDX_PROD);

		faceCap->SetInput(INDX_CSM % 10);

		if (first) {
			// Re-scale all blendshapes to match user
			if (faceCap->Rescale(Blendshape)) {
				if (faceCap->AlignToFace(Blendshape, inverted)) {
					if (faceCap->ElasticRegistrationFull(Blendshape)) {
						faceCap->GenerateBump(Blendshape, 0, 0, BumpWidth, BumpHeight);
						//faceCap->SetLandmarks(Blendshape[0]);
						first = false;
					}
				}
			}
		}
		else {
			//float current_time = clock();

			faceCap->Register(Blendshape);

			faceCap->EstimateBlendShapeCoefficientsPR(Blendshape);
			//faceCap->EstimateBlendShapeCoefficientsLandmarks(Blendshape);

			for (int i = 0; i < NB_THREADS; i++)
				for (int j = 0; j < NB_THREADS; j++)
					t_csm[i*NB_THREADS + j] = thread(GenerateBumpMng, i*(BumpWidth / NB_THREADS),
						j*(BumpHeight / NB_THREADS),
						(i + 1)*(BumpWidth / NB_THREADS),
						(j + 1)*(BumpHeight / NB_THREADS));

			/*if (INDX_CSM % 4 == 0)
			faceCap->GenerateBump(Blendshape, 0, 0, BumpWidth/2, BumpHeight/2);
			else if (INDX_CSM % 4 == 1)
			faceCap->GenerateBump(Blendshape, BumpWidth / 2, 0, BumpWidth, BumpHeight/2);
			else if (INDX_CSM % 4 == 2)
			faceCap->GenerateBump(Blendshape, 0, BumpHeight / 2, BumpWidth/2, BumpHeight);
			else
			faceCap->GenerateBump(Blendshape, BumpWidth / 2, BumpHeight / 2, BumpWidth, BumpHeight);
			*/
			for (int i = 0; i < NB_THREADS*NB_THREADS; i++)
				t_csm[i].join();
		}
		//clFlush(faceCap->_queue_consume);
		//clFinish(faceCap->_queue_consume);

		INDX_CSM++;

		my_count_csm++;
		current_time_csm = clock();
		if ((current_time_csm - last_time_csm) / CLOCKS_PER_SEC > 1.0) {
			fps_csm = my_count_csm / ((current_time_csm - last_time_csm) / CLOCKS_PER_SEC);
			last_time_csm = current_time_csm;
			my_count_csm = 0.0;
			cout << "fps consume: " << fps_csm << endl;
		}
	}
}


void stream()
{
	while (true) {

		int res = faceCap->Update();

		unique_lock<mutex> lck(imageMutex);
		if (condv.wait_for(lck, chrono::milliseconds(1), isTerminated)) break;


		while (true) {
			if (!lck.owns_lock())
				lck.lock();

			if (faceCap->Push() || terminated) {
				lck.unlock();
				break;
			}

			lck.unlock();
		}

		my_count_csm++;
		current_time_csm = clock();
		if ((current_time_csm - last_time_csm) / CLOCKS_PER_SEC > 1.0) {
			fps_csm = my_count_csm / ((current_time_csm - last_time_csm) / CLOCKS_PER_SEC);
			last_time_csm = current_time_csm;
			my_count_csm = 0.0;
			cout << "fps: " << fps_csm << endl;

		}
	}
}

void saveInput(int k) {
	chrono::time_point<std::chrono::system_clock> start_t, end_t;
	while (true) {
		unique_lock<mutex> lck(imageMutex);
		if (condv.wait_for(lck, chrono::milliseconds(1), isTerminated)) break;

		bool oksave = false;
		if (!lck.owns_lock())
			lck.lock();

		if (faceCap->LoadToSave(k) == 1)
			oksave = true;

		faceCap->Pop();
		lck.unlock();

		if (oksave)
			faceCap->SaveData(k);
	}
}

bool App::setupOpenGL()
{
	glewExperimental = TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		//Problem: glewInit failed, something is seriously wrong.
		cout << "glewInit failed, aborting." << endl;
		return false;
	}

	/*** Initialize OpenGL ***/
	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations
														// enable color tracking
	glEnable(GL_COLOR_MATERIAL);
	// set material properties which will be assigned by glColor
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnable(GL_TEXTURE_2D);
	glEnable(GL_BLEND);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_BACK);

	intrinsics[0] = 2.0 * Calib[0] / cDepthWidth;
	intrinsics[5] = 2.0 * Calib[1] / cDepthHeight;
	intrinsics[10] = -(Zfar + Znear) / (Zfar - Znear);
	intrinsics[11] = -1.0;
	intrinsics[14] = -2.0*(Zfar * Znear) / (Zfar - Znear);

	return true;
}

void App::setupFacialModel()
{
	cout << "start: setupFacialModel" << endl;

	faceCap = new FaceCap();

	cout << "load: Neutral Model" << endl;

	MyMesh *TheMesh;
	TheMesh = new MyMesh(&faceCap->_verticesList[0], &faceCap->_triangles[0]);
	TheMesh->Load(string("..\\data\\blendshapes\\MyTemplate\\Neutralm.obj"), true);
	Blendshape.push_back(TheMesh);

	cout << "load: Template Model " << endl;

	// remove 02,03,04,05,06,07,08,09,10,11,12,13,17,18,19,22,26,27,34,35,42
	char filename[100];
	int indx_vtx = 1;
	for (int i = 0; i < 2; i++) {
		TheMesh = new MyMesh(&faceCap->_verticesList[indx_vtx * 4325], &faceCap->_triangles[0]);
		TheMesh->Load(string("..\\data\\blendshapes\\MyTemplate\\0") + to_string(i) + string("m.obj"));
		Blendshape.push_back(TheMesh);
		indx_vtx++;
	}

	for (int i = 14; i < 48; i++) {
		if (i == 17 || i == 18 || i == 19 || i == 22 || i == 26 || i == 27 || i == 34 || i == 35 || i == 42)
			continue;

		TheMesh = new MyMesh(&faceCap->_verticesList[indx_vtx * 4325], &faceCap->_triangles[0]);
		TheMesh->Load(string("..\\data\\blendshapes\\MyTemplate\\") + to_string(i) + string("m.obj"));
		Blendshape.push_back(TheMesh);
		indx_vtx++;
	}
	cout << "Number of expressions : " << Blendshape.size() << endl;

	faceCap->ComputeAffineTransfo(Blendshape);
}