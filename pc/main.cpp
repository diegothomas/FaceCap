
#include <iostream>

#include "../src/stdafx.h"
#include "../src/App.h"
#include <GLFW/glfw3.h>

namespace {
	App app;
	bool bSave;
	char* savePath = "..\\data\\Seq";
}

void reshape(int width_in, int height_in)
{
	app.resize(width_in, height_in);
}

void display(void)
{
	app.update();
	app.draw();
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case '1':
		cout << "CAPTURE ONLINE WITHOUT SAVING" << endl;
		bSave = false;
		app.captureOnline(bSave, "");
		break;
	case '2':
		cout << "CAPTURE ONLINE WITH SAVING" << endl;
		bSave = true;
		app.captureOnline(bSave, savePath);
		break;
	case '3':
		cout << "PLAYBACK WITH SAVED DATA" << endl;
		app.captureOffline(savePath); // start capture with sensor initialization
		break;
		//case 27: // ESC
		//	app.exit();
		//	break;
	default:
		break;
	}
}

int initGL(GLFWwindow*& window)
{
	/* Initialize the library */
	if (!glfwInit())
		return 1;
}

int main(int argc, _TCHAR* argv[]) {

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH /*| GLUT_DOUBLE*/);
	glutInitWindowSize(cDepthWidth, cDepthHeight); // (3 * cDepthWidth, 2 * cDepthHeight);
	GLuint window = glutCreateWindow("FaceCap");

	//GLFWwindow* window;
	//int ret = initGL(window);

	if (!app.setup())
		return 1;

	glutReshapeFunc(reshape);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);

	glutMainLoop();

	return 0;
}
