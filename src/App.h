// ConsoleApplication1.cpp : Defines the entry point for the console application.
//
#pragma once

#include "stdafx.h"

class App
{
public:
	App();
	~App();

	bool setup();
	void update();
	void draw();
	void exit();
	void resize(int width_in, int height_in);

	void captureOnline(bool bSave, char *path = "");
	void captureOffline(char *path);
	void playback();

private:
	bool setupOpenGL();
	void setupFacialModel();

	void computePos(float deltaMove, float deltaStrap);
	void saveImg(int x, int y, int id, char *path);

	void display(void);
	void keyUp(unsigned char key, int x, int y);
	void pressKey(int key, int xx, int yy);
	void releaseKey(int key, int x, int y);
	void mouseMove(int x, int y);
	void mouseButton(int button, int state, int x, int y);
	void keyboard(unsigned char key, int x, int y);

	bool initDepthSensor();
};