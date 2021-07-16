
#include "hwKinectManager.h"

SensorManager::Impl::Impl()
{
	// initializing all the pointers and variables
	DWORD widthd = 0;
	DWORD heightd = 0;

	NuiImageResolutionToSize(depthResolution, widthd, heightd);
	dwidth = static_cast<LONG>(widthd);
	dheight = static_cast<LONG>(heightd);

	NuiImageResolutionToSize(colorResolution, widthd, heightd);
	cwidth = static_cast<LONG>(widthd);
	cheight = static_cast<LONG>(heightd);

	colordata = (unsigned char *)malloc(cwidth*cheight * 4 * sizeof(unsigned char));
	depthdata = (unsigned short *)malloc(dwidth*dheight * sizeof(unsigned short));
	colorCoordinates = (LONG*)malloc(dwidth*dheight * 2 * sizeof(LONG));

	CtoDdiv = cwidth / dwidth;
}

SensorManager::Impl::~Impl()
{
	if (NULL != sensor)
	{
		sensor->NuiShutdown();
		cout << "Release Kinect" << endl;
		sensor->Release();
	}

	free(colordata);
	free(depthdata);
	free(colorCoordinates);
}

bool SensorManager::Impl::init()
{
	INuiSensor * pNuiSensor;
	HRESULT hr;

	int iSensorCount = 0;
	hr = NuiGetSensorCount(&iSensorCount);

	// Look at each Kinect sensor
	for (int i = 0; i < iSensorCount; ++i)
	{
		// Create the sensor so we can check status, if we can't create it, move on to the next
		hr = NuiCreateSensorByIndex(i, &pNuiSensor);
		if (FAILED(hr))
		{
			continue;
		}

		// Get the status of the sensor, and if connected, then we can initialize it
		hr = pNuiSensor->NuiStatus();
		if (S_OK == hr)
		{
			sensor = pNuiSensor;
			break;
		}

		// This sensor wasn't OK, so release it since we're not using it
		pNuiSensor->Release();
	}

	if (NULL != sensor)
	{
		// Initialize the Kinect and specify that we'll be using depth
		hr = sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH | NUI_INITIALIZE_FLAG_USES_COLOR);
		if (SUCCEEDED(hr))
		{
			// Create an event that will be signaled when depth data is available
			NextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

			// Open a depth image stream to receive depth frames
			hr = sensor->NuiImageStreamOpen(
				NUI_IMAGE_TYPE_DEPTH,
				depthResolution,
				NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE,
				2,
				NextDepthFrameEvent,
				&depthStream);

			// Create an event that will be signaled when color data is available
			NextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

			// Initialize sensor to open up color stream
			hr = sensor->NuiImageStreamOpen(
				NUI_IMAGE_TYPE_COLOR,
				colorResolution,
				0,
				2,
				NextColorFrameEvent,
				&rgbStream);

			if (FAILED(hr)) return hr;

			/*INuiColorCameraSettings *camSettings;
			hr = sensor->NuiGetColorCameraSettings(&camSettings);

			if (FAILED(hr))      return hr;

			hr = camSettings->SetAutoExposure(FALSE);
			if (FAILED(hr))      return hr;

			hr = camSettings->SetAutoWhiteBalance(FALSE);
			if (FAILED(hr))      return hr;

			hr = camSettings->SetWhiteBalance(3500);
			if (FAILED(hr))      return hr;*/
		}
	}

	if (NULL == sensor || FAILED(hr))
	{
		return false;
	}

	return true;
}

bool SensorManager::Impl::getDepth(unsigned short *dest)
{
	NUI_IMAGE_FRAME imageFrame;
	NUI_LOCKED_RECT LockedRect;
	HRESULT hr;

	hr = sensor->NuiImageStreamGetNextFrame(depthStream, 0, &imageFrame);
	if (FAILED(hr)) 		return hr;

	INuiFrameTexture *texture = imageFrame.pFrameTexture;
	hr = texture->LockRect(0, &LockedRect, NULL, 0);
	if (FAILED(hr)) 		return hr;

	// Now copy the data to our own memory location
	if (LockedRect.Pitch != 0)
	{
		const GLushort* curr = (const unsigned short*)LockedRect.pBits;

		// copy the texture contents from current to destination
		memcpy(dest, curr, sizeof(unsigned short)*(dwidth*dheight));
	}

	hr = texture->UnlockRect(0);
	if (FAILED(hr)) 		return hr;

	hr = sensor->NuiImageStreamReleaseFrame(depthStream, &imageFrame);
	if (FAILED(hr)) 		return hr;

	return S_OK;
}

bool SensorManager::Impl::getColor(unsigned char *dest)
{
	NUI_IMAGE_FRAME imageFrame; // structure containing all the metadata about the frame
	NUI_LOCKED_RECT LockedRect; // contains the pointer to the actual data
	HRESULT hr;                 // Error handling

	hr = sensor->NuiImageStreamGetNextFrame(rgbStream, 0, &imageFrame);
	if (FAILED(hr))		return hr;

	INuiFrameTexture *texture = imageFrame.pFrameTexture;
	hr = texture->LockRect(0, &LockedRect, NULL, 0);
	if (FAILED(hr))      return hr;

	// Now copy the data to our own memory location
	if (LockedRect.Pitch != 0)
	{
		const BYTE* curr = (const BYTE*)LockedRect.pBits;

		// copy the texture contents from current to destination
		memcpy(dest, curr, sizeof(BYTE)*(cwidth*cheight * 4));
	}

	hr = texture->UnlockRect(0);
	if (FAILED(hr))      return hr;

	hr = sensor->NuiImageStreamReleaseFrame(rgbStream, &imageFrame);
	if (FAILED(hr))      return hr;

	return S_OK;
}

bool SensorManager::Impl::MapColorToDepth(unsigned char* colorFrame, unsigned short* depthFrame)
{
	HRESULT hr;

	// Find the location in the color image corresponding to the depth image
	hr = sensor->NuiImageGetColorPixelCoordinateFrameFromDepthPixelFrameAtResolution(
		colorResolution,
		depthResolution,
		dwidth*dheight,
		depthFrame,
		(dwidth*dheight) * 2,
		colorCoordinates);

	if (FAILED(hr))    return hr;

	return S_OK;
}

void SensorManager::Impl::update()
{
	bool needToMapColorToDepth = false;
	HRESULT hr;

	while (true) {
		if (WAIT_OBJECT_0 == WaitForSingleObject(NextDepthFrameEvent, 0))
		{
			// if we have received any valid new depth data we proceed to obtain new color data
			if ((hr = getDepth(depthdata)) == S_OK)
			{
				if (WAIT_OBJECT_0 == WaitForSingleObject(NextColorFrameEvent, 0))
				{
					// if we have received any valid new color data we proceed to extract skeletal information
					if ((hr = getColor(colordata)) == S_OK)
					{
						MapColorToDepth((BYTE*)colordata, (USHORT*)depthdata);
						return;
					}
				}
			}
		}
	}
}

int SensorManager::Impl::getColorWidth()
{
	return cwidth;
}
int SensorManager::Impl::getColorHeight()
{
	return cheight;
}
int SensorManager::Impl::getDepthWidth()
{
	return dwidth;
}
int SensorManager::Impl::getDepthHeight()
{
	return dheight;
}

unsigned char* SensorManager::Impl::getColorFrame()
{
	return colordata;
}

unsigned short* SensorManager::Impl::getDepthFrame()
{
	return depthdata;
}

long* SensorManager::Impl::getColorCoord()
{
	return colorCoordinates;
}
