
#include "SensorManager.h"

#ifdef __ANDROID__
	/*** SDK for Orbec ***/
	#include "OrbecManager.h"
#else
	/*** SDK for Kinect ***/
	#include "KinectManager.h"
#endif

SensorManager::SensorManager() : mImpl(new Impl)
{

}

SensorManager::~SensorManager()
{

}

bool SensorManager::init()
{
	return mImpl->init();
}

void SensorManager::update()
{
	return mImpl->update();
}

unsigned char* SensorManager::getColorFrame()
{
	return mImpl->getColorFrame();
}

unsigned short* SensorManager::getDepthFrame()
{
	return mImpl->getDepthFrame();
}

long* SensorManager::getColorCoord()
{
	return mImpl->getColorCoord();
}

bool SensorManager::getDepth(unsigned short *dest)
{
	return mImpl->getDepth(dest);
}

bool SensorManager::getColor(unsigned char *dest)
{
	return mImpl->getColor(dest);
}

bool SensorManager::MapColorToDepth(unsigned char* colorFrame, unsigned short* depthFrame)
{
	// TBD
	return true;
}

int SensorManager::getColorWidth()
{
	return mImpl->getColorWidth();
}

int SensorManager::getDepthWidth()
{
	return mImpl->getDepthWidth();
}

int SensorManager::getColorHeight()
{
	return mImpl->getColorHeight();
}

int SensorManager::getDepthHeight()
{
	return mImpl->getDepthHeight();
}
