package com.weijie.testopencv.util;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 9:34 AM 2018/8/29
 * @Modified By:
 */
public class ImageOpenCVUtil {

//    private static final int BLACK = 0;
//    private static final int WHITE = 255;

    private Mat mat;

    /**
     * 空参构造函数
     */
    public ImageOpenCVUtil() {

    }

    /**
     * 通过图像路径创建一个mat矩阵
     *
     * @param imgFilePath
     *            图像路径
     */
    public ImageOpenCVUtil(String imgFilePath) {
        mat = Imgcodecs.imread(imgFilePath);
    }

    public ImageOpenCVUtil(Mat mat) {
        this.mat = mat;
    }

    /**
     * 加载图片
     *
     * @param imgFilePath
     */
    public void loadImg(String imgFilePath) {
        mat = Imgcodecs.imread(imgFilePath);
    }

    /**
     * 获取图片高度的函数
     *
     * @return
     */
    public int getHeight() {
        return mat.rows();
    }

    /**
     * 获取图片宽度的函数
     *
     * @return
     */
    public int getWidth() {
        return mat.cols();
    }

    /**
     * 获取图片像素点的函数
     *
     * @param y
     * @param x
     * @return
     */
    public int getPixel(int y, int x) {
        // 我们处理的是单通道灰度图
        return (int) mat.get(y, x)[0];
    }

    /**
     * 设置图片像素点的函数
     *
     * @param y
     * @param x
     * @param color
     */
    public void setPixel(int y, int x, int color) {
        // 我们处理的是单通道灰度图
        mat.put(y, x, color);
    }

    /**
     * 保存图片的函数
     *
     * @param filename
     * @return
     */
    public boolean saveImg(String filename) {
        return Imgcodecs.imwrite(filename, mat);
    }
}
