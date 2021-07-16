
#ifndef __HW_KINECT_MANAGER_H
#define __HW_KINECT_MANAGER_H

#include "SensorManager.h"
#include "../src/stdafx.h"

class SensorManager::Impl
{
public:
	Impl();
	~Impl();

	bool init();
	void update();

	bool getDepth(unsigned short *dest);
	bool getColor(unsigned char *dest);

	unsigned char* getColorFrame();
	unsigned short* getDepthFrame();
	long* getColorCoord();

	/* Function to map color frame to the depth frame */
	bool MapColorToDepth(unsigned char* colorFrame, unsigned short* depthFrame);

	int getColorWidth();
	int getColorHeight();
	int getDepthWidth();
	int getDepthHeight();

private:
	// Resolution of the streams
	static const NUI_IMAGE_RESOLUTION colorResolution = NUI_IMAGE_RESOLUTION_640x480;
	static const NUI_IMAGE_RESOLUTION depthResolution = NUI_IMAGE_RESOLUTION_640x480;

	// Mapped color coordinates from color frame to depth frame
	LONG*                         colorCoordinates;

	// Event handlers
	HANDLE						  rgbStream;
	HANDLE						  depthStream;
	HANDLE						  NextDepthFrameEvent;
	HANDLE						  NextColorFrameEvent;

	// Variables related to resolution assigned in constructor
	int							  CtoDdiv;
	long int					  cwidth;
	long int					  dwidth;
	long int					  cheight;
	long int					  dheight;

	// Actual sensor connected to the Computer
	INuiSensor*				      sensor;

	// Color and depth frames
	unsigned char*				  colordata;
	unsigned short*				  depthdata;

	/****** Class function declarations *******/
};


#endif


