
#ifndef __HW_ORBEC_MANAGER_H
#define __HW_ORBEC_MANAGER_H

#include "SensorManager.h"

class SensorManager::Impl
{
public:
	Impl();
	~Impl();

	bool init() {}
	void update() {}

	bool getDepth(unsigned short *dest) {}
	bool getColor(unsigned char *dest) {}

	unsigned char* getColorFrame() {}
	unsigned short* getDepthFrame() {}
	long* getColorCoord() {}

	/* Function to map color frame to the depth frame */
	bool MapColorToDepth(unsigned char* colorFrame, unsigned short* depthFrame) {}

	int getColorWidth() {}
	int getColorHeight() {}
	int getDepthWidth() {}
	int getDepthHeight() {}

private:
	int mColorWidth;
	int mColorHeight;
	int mDepthWidth;
	int mDepthHeight;
};


#endif


