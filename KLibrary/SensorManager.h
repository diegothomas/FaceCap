
#ifndef __HW_SENSOR_MANAGER_H
#define __HW_SENSOR_MANAGER_H

#include <memory>

class SensorManager
{
public:

	SensorManager();
	~SensorManager();

	bool init();
	void update();
	bool getDepth(unsigned short *dest);
	bool getColor(unsigned char *dest);
	bool MapColorToDepth(unsigned char* colorFrame, unsigned short* depthFrame);
	
	int getColorWidth();
	int getColorHeight();
	int getDepthWidth();
	int getDepthHeight();

	unsigned char* getColorFrame();
	unsigned short* getDepthFrame();
	long* getColorCoord();

private:

	class Impl;
	std::unique_ptr<Impl> mImpl;
};
#endif


