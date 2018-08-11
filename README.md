# Slate

## About
Slate is a handwriting recognition program that takes input with the help of a pen by writing in air, with more than 95% * accuracy which does so much more than just recognition.There's a calculator,brightness control, music player, sketch pad and an ASCII art all which can be used with minimum keyboard interaction by the user. The possibilities are limitless and we will release updates at regular intevals. 

### Video Link
https://youtu.be/K1Mnbz6ATeo

### Pen
![alt text](https://github.com/kingsisland/Slate/blob/master/penWork.jpg)

## CONTRIBUTORS
* Anubhaw Bhalotia
* Karthik Vedantam
* Muskan Jhunjhunwalla

## Installation Instruction

### Updating Instruction

$ sudo apt-get update

$ sudo apt-get upgrade

### Install Dependencies

$ sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

$ sudo apt-get install python3.5-dev python3-numpy libtbb2 libtbb-dev

$ sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libjasper-dev libdc1394-22-dev libeigen3-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev sphinx-common libtbb-dev yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libavresample-dev

### Get OpenCV

$ sudo -s

$ cd /opt

/opt$ git clone https://github.com/Itseez/opencv.git

/opt$ git clone https://github.com/Itseez/opencv_contrib.git

### Build and install opencv

/opt$ cd opencv

/opt/opencv$ mkdir release
 
/opt/opencv$ cd release
 
/opt/opencv/release$ cmake -D BUILD_TIFF=ON -D WITH_CUDA=OFF -D ENABLE_AVX=OFF -D WITH_OPENGL=OFF -D WITH_OPENCL=OFF -D WITH_IPP=OFF -D WITH_TBB=ON -D BUILD_TBB=ON -D WITH_EIGEN=OFF -D WITH_V4L=OFF -D WITH_VTK=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules /opt/opencv/
 
/opt/opencv/release$ make -j4
 
/opt/opencv/release$ make install
 
/opt/opencv/release$ ldconfig
 
/opt/opencv/release$ exit
 
/opt/opencv/release$ cd ~

### Check if openCV is correctly installed

$ pkg-config --modversion opencv

## Compile Instructions

$ g++ -std=gnu++11 slate.cpp -o output `pkg-config --cflags --libs opencv`

## Running the output file

$ ./output


