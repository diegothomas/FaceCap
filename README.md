# FaceCap(KARI)
Facial Capture Project

## Environment

+ Windows 10 (also 7) 64bit  
+ Visual Studio 2017 64bit  
+ CMake 3.10.0
+ Android Studio 3.0   

## Dependency
+ Kinect for Windows SDK (v1.8 and v2.0)
  + [Kinect for Windows SDK 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278)
  + [Kinect for Windows Developer Toolkit v1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40276)

+ AMD APP SDK (OpenGL)
  + [3.0]( https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/)

+ GLFW
  + [3.2.1](http://www.glfw.org/)

+ OpenCV
  + [3.4.0 (sources)](https://opencv.org/releases.html)

+ Intraface
  + [From diego's code](http://limu.ait.kyushu-u.ac.jp/files/diego_data_CVPR2016/CodeCVPR.zip)

## Project Build Flow

+ Please run CMake
  + Specified the generator: Visual Studio 15 (2017) Win64
  + set the build directory such as "FaceCap/build"
  + Configure and Generate
+ Open Visual Studio project
   : Load and track the face from RGBD images saved + Set Release mPush key ode : Load and track the face from RGBD images saved 
  + Set facecap_pc as startup project
+ After Running
  + Push key "1" : ONLINE MODE WITHOUT SAVING (start capturing in real-time)  
  + Push key "2" : ONLINE MODE WITH SAVING (saving RGBD images in real-time)  
  + Push key "3" : OFFLINE MODE (Load and track the face from RGBD images saved)  