QT += core
QT -= gui

CONFIG += c++14

TARGET = test
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv
INCLUDEPATH += /usr/local/include/opencv2

LIBS += -L/usr/local/lib\
     -lopencv_calib3d\
     -lopencv_contrib\
     -lopencv_core\
     -lopencv_features2d \
     -lopencv_flann\
     -lopencv_gpu\
     -lopencv_highgui\
     -lopencv_imgproc\
     -lopencv_legacy\
     -lopencv_ml\
     -lopencv_objdetect\
     -lopencv_ocl\
     -lopencv_photo\
     -lopencv_stitching\
     -lopencv_superres\
     -lopencv_ts\
     -lopencv_video\
     -lopencv_videostab\
     -lopencv_videoio

SOURCES += \
    main.cpp


