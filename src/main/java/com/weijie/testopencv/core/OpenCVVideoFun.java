package com.weijie.testopencv.core;

import org.opencv.core.Core;
import org.opencv.videoio.VideoCapture;

/**
 * @Author: liangjiazhang
 * @Description:  OpenCV 视频处理相关类
 * @Date: Created in 10:02 PM 2018/8/29
 * @Modified By:
 */
public class OpenCVVideoFun {


    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public OpenCVVideoFun () {

    }


    public void videoRead(String videoPath) {

        // open video or camero
        VideoCapture capture = new VideoCapture();
        //capture.open(0);
        capture.open("/Users/liangjiazhang/Downloads/videoplayback.mp4");

        if (!capture.isOpened()) {
            System.out.println("could not load video data...");
            return;
        }
        int frame_width = (int) capture.get(3);
        int frame_heigth = (int) capture.get(4);

    }


}
