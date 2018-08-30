package com.weijie.testopencv.core;




import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.cvShowImage;
import static org.bytedeco.javacpp.opencv_highgui.cvWaitKey;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2HSV;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.equalizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.opencv_highgui;


/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 2:49 PM 2018/8/21
 * @Modified By:
 */
public class CoreFunc {



    // Load the image
//    Mat img = Imgcodecs.imread(file);
//
//    // Creating an empty matrix
//    Mat equ = new Mat();
//      img.copyTo(equ);
//
//    // Applying blur
//      Imgproc.blur(equ, equ, new Size(3, 3));
//
//    // Applying color
//      Imgproc.cvtColor(equ, equ, Imgproc.COLOR_BGR2YCrCb);
//    List<Mat> channels = new ArrayList<Mat>();
//
//    // Splitting the channels
//      Core.split(equ, channels);
//
//    // Equalizing the histogram of the image
//      Imgproc.equalizeHist(channels.get(0), channels.get(0));
//      Core.merge(channels, equ);
//      Imgproc.cvtColor(equ, equ, Imgproc.COLOR_YCrCb2BGR);
//
//    Mat gray = new Mat();
//      Imgproc.cvtColor(equ, gray, Imgproc.COLOR_BGR2GRAY);
//    Mat grayOrig = new Mat();
//      Imgproc.cvtColor(img, grayOrig, Imgproc.COLOR_BGR2GRAY);
//
//      Imgcodecs.imwrite("F:/worksp/opencv/images/histo_output.jpg", equ);
//      System.out.println("Image Processed");

//    public enum Color {
//        UNKNOWN, BLUE, YELLOW
//    };
//
//    public enum Direction {
//        UNKNOWN, VERTICAL, HORIZONTAL
//    }
//
//
//    public static Mat colorMatch(final Mat src, final Color color, final boolean adaptive_minsv) {
//
//        final float max_sv = 255;
//        final float minref_sv = 64;
//        final float minabs_sv = 95;
//
//
//        // blue 的 H 范围
//        final int min_blue = 100;
//        final int max_blue = 140;
//
//        // yellow 的 H 范围
//        final int min_yellow = 15;
//        final int max_yellow = 40;
//
//
//        /**
//         * 主要使用cvtColor方法来实现颜色空间的互转。
//         *  Mat img = Imgcodecs.imread("img/tooth1.png");
//
//         Mat imgHSV = new Mat(img.rows(), img.cols(), CvType.CV_8UC3);
//         Mat img2 = new Mat(img.rows(), img.cols(), CvType.CV_8UC3);
//
//         //转成HSV空间
//         Imgproc.cvtColor(img, imgHSV, Imgproc.COLOR_BGR2HSV);
//         //转回BGR空间
//         Imgproc.cvtColor(imgHSV, img2, Imgproc.COLOR_HSV2BGR);
//         */
//
//        // 转到HSV 空间进行处理, 颜色搜索主要使用的是H分量进行蓝色与黄色的匹配工作
//        Mat src_hsv = new Mat();
//        Imgproc.cvtColor(src, src_hsv, Imgproc.COLOR_BGR2HSV);
//        List<Mat> channelsy = new ArrayList<Mat>();
//
////        opencv_core.MatVector hsvSplit = new opencv_core.MatVector();
//        Core.split(src_hsv, channelsy);
//        Imgproc.equalizeHist(channelsy.get(2), channelsy.get(2));
//        Core.merge(channelsy, src_hsv);
//
//        // 匹配模板基色,切换以查找想要的基色
//        int min_h = 0;
//        int max_h = 0;
//        switch (color) {
//            case BLUE:
//                min_h = min_blue;
//                max_h = max_blue;
//                break;
//            case YELLOW:
//                min_h = min_yellow;
//                max_h = max_yellow;
//                break;
//            default:
//                break;
//        }
//
//        float diff_h = (float) ((max_h - min_h) / 2);
//        int avg_h = (int) (min_h + diff_h);
//
//        int channels = src_hsv.channels();
//        int nRows = src_hsv.rows();
//        // 图像数据列需要考虑通道数的影响；
//        int nCols = src_hsv.cols() * channels;
//
//        // 连续存储的数据，按一行处理
//        if (src_hsv.isContinuous()) {
//            nCols *= nRows;
//            nRows = 1;
//        }
//
//        for (int i = 0; i < nRows; ++i) {
//            BytePointer p = src_hsv.ptr(i);
//            for (int j = 0; j < nCols; j += 3) {
//                int H = p.get(j) & 0xFF;
//                int S = p.get(j + 1) & 0xFF;
//                int V = p.get(j + 2) & 0xFF;
//
//                boolean colorMatched = false;
//
//                if (H > min_h && H < max_h) {
//                    int Hdiff = 0;
//                    if (H > avg_h)
//                        Hdiff = H - avg_h;
//                    else
//                        Hdiff = avg_h - H;
//
//                    float Hdiff_p = Hdiff / diff_h;
//
//                    float min_sv = 0;
//                    if (true == adaptive_minsv)
//                        min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p);
//                    else
//                        min_sv = minabs_sv;
//
//                    if ((S > min_sv && S <= max_sv) && (V > min_sv && V <= max_sv))
//                        colorMatched = true;
//                }
//
//                if (colorMatched == true) {
//                    p.put(j, (byte) 0);
//                    p.put(j + 1, (byte) 0);
//                    p.put(j + 2, (byte) 255);
//                } else {
//                    p.put(j, (byte) 0);
//                    p.put(j + 1, (byte) 0);
//                    p.put(j + 2, (byte) 0);
//                }
//            }
//        }
//
//        // 获取颜色匹配后的二值灰度图
//        opencv_core.MatVector hsvSplit_done = new opencv_core.MatVector();
//        split(src_hsv, hsvSplit_done);
//        Mat src_grey = hsvSplit_done.get(2);
//
//        return src_grey;
//    }
//
//
//
//
//
//
//    /**
//     * 判断一个车牌的颜色
//     * @param src  车牌mat
//     * @param color  颜色模板
//     * @param adaptive_minsv  S和V的最小值由adaptive_minsv这个bool值判断
//     *                        如果为true，则最小值取决于H值，按比例衰减
//     *                        如果为false，则不再自适应，使用固定的最小值minabs_sv
//     * @return
//     */
//    public static boolean plateColorJudge(final Mat src, final Color color, final boolean adaptive_minsv) {
//        // 判断阈值
//        final float thresh = 0.49f;
//
//        Mat gray = colorMatch(src, color, adaptive_minsv);
//
//        float percent = (float) countNonZero(gray) / (gray.rows() * gray.cols());
//
//        return (percent > thresh) ? true : false;
//    }
//
//
//    /**
//     * 判断车牌的类型
//     * @param src
//     * @param adaptive_minsv  S和V的最小值由adaptive_minsv这个bool值判断
//     *                        如果为true，则最小值取决于H值，按比例衰减
//     *                        如果为false，则不再自适应，使用固定的最小值minabs_sv
//     *
//     * @return
//     */
//    public static Color getPlateType(final Mat src, final boolean adaptive_minsv) {
//        if (plateColorJudge(src, BLUE, adaptive_minsv) == true) {
//            return BLUE;
//        } else if (plateColorJudge(src, YELLOW, adaptive_minsv) == true) {
//            return YELLOW;
//        } else {
//            return Color.UNKNOWN;
//        }
//    }
//
//
//    /**
//     * 获取垂直或水平方向直方图
//     * @param img
//     * @param direction
//     * @return
//     */
//    public static float[] projectedHistogram(final Mat img, Direction direction) {
//        int sz = 0;
//        switch (direction) {
//            case HORIZONTAL:
//                sz = img.rows();
//                break;
//            case VERTICAL:
//                sz = img.cols();
//                break;
//            default:
//                break;
//        }
//        // 统计这一行或一列中，非零元素的个数，并保存到nonZeroMat中
//        float[] nonZeroMat = new float[sz];
//        extractChannel(img, img, 0);
//        for (int j = 0; j < sz; j++) {
//            Mat data = (direction == Direction.HORIZONTAL) ? img.row(j) : img.col(j);
//            int count = countNonZero(data);
//            nonZeroMat[j] = count;
//        }
//
//        // Normalize histogram
//        float max = 0;
//        for (int j = 0; j < nonZeroMat.length; ++j) {
//            max = Math.max(max, nonZeroMat[j]);
//        }
//
//        if (max > 0) {
//            for (int j = 0; j < nonZeroMat.length; ++j) {
//                nonZeroMat[j] /= max;
//            }
//        }
//
//        return nonZeroMat;
//    }
//
//    /**
//     * Assign values to feature
//     * <p>
//     * 样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
//     *
//     * @param in
//     * @param sizeData - 低分辨率图像size = sizeData*sizeData, 可以为0
//     * @return
//     */
//    public static Mat features(final Mat in, final int sizeData) {
//        float[] vhist = projectedHistogram(in, Direction.VERTICAL);
//        float[] hhist = projectedHistogram(in, Direction.HORIZONTAL);
//        Mat lowData = new Mat();
//        if (sizeData > 0) {
//            resize(in, lowData, new Size(sizeData, sizeData));
//        }
//
//        int numCols = vhist.length + hhist.length + lowData.cols() * lowData.rows();
//        Mat out = Mat.zeros(1, numCols, CV_32F).asMat();
//        FloatIndexer idx = out.createIndexer();
//
//        int j = 0;
//        for (int i = 0; i < vhist.length; ++i, ++j) {
//            idx.put(0, j, vhist[i]);
//        }
//        for (int i = 0; i < hhist.length; ++i, ++j) {
//            idx.put(0, j, hhist[i]);
//        }
//        for (int x = 0; x < lowData.cols(); x++) {
//            for (int y = 0; y < lowData.rows(); y++, ++j) {
//                float val = lowData.ptr(x, y).get() & 0xFF;
//                idx.put(0, j, val);
//            }
//        }
//
//        return out;
//    }
//
//
//
//
//    /**
//     * 显示图像
//     *
//     */
//    public static void showImage(final String title, final Mat src) {
//        if (src !=null) {
//            HighGui.imshow(title, src);
//            HighGui.waitKey();
//        }
//    }


    public enum Color {
        UNKNOWN, BLUE, YELLOW
    };

    public enum Direction {
        UNKNOWN, VERTICAL, HORIZONTAL
    }

    /**
     * 根据一幅图像与颜色模板获取对应的二值图
     *
     * @param src
     *            输入RGB图像
     * @param r
     *            颜色模板（蓝色、黄色）
     * @param adaptive_minsv
     *            S和V的最小值由adaptive_minsv这个bool值判断
     *            <ul>
     *            <li>如果为true，则最小值取决于H值，按比例衰减
     *            <li>如果为false，则不再自适应，使用固定的最小值minabs_sv
     *            </ul>
     * @return 输出灰度图（只有0和255两个值，255代表匹配，0代表不匹配）
     */
    public static Mat colorMatch(final Mat src, final Color r, final boolean adaptive_minsv) {
        final float max_sv = 255;
        final float minref_sv = 64;
        final float minabs_sv = 95;

        // blue的H范围
        final int min_blue = 100;
        final int max_blue = 140;

        // yellow的H范围
        final int min_yellow = 15;
        final int max_yellow = 40;

        // 转到HSV空间进行处理，颜色搜索主要使用的是H分量进行蓝色与黄色的匹配工作
        Mat src_hsv = new Mat();
        cvtColor(src, src_hsv, CV_BGR2HSV);
        MatVector hsvSplit = new MatVector();
        split(src_hsv, hsvSplit);
        equalizeHist(hsvSplit.get(2), hsvSplit.get(2));
        merge(hsvSplit, src_hsv);

        // 匹配模板基色,切换以查找想要的基色
        int min_h = 0;
        int max_h = 0;
        switch (r) {
            case BLUE:
                min_h = min_blue;
                max_h = max_blue;
                break;
            case YELLOW:
                min_h = min_yellow;
                max_h = max_yellow;
                break;
            default:
                break;
        }

        float diff_h = (float) ((max_h - min_h) / 2);
        int avg_h = (int) (min_h + diff_h);

        int channels = src_hsv.channels();
        int nRows = src_hsv.rows();
        // 图像数据列需要考虑通道数的影响；
        int nCols = src_hsv.cols() * channels;

        // 连续存储的数据，按一行处理
        if (src_hsv.isContinuous()) {
            nCols *= nRows;
            nRows = 1;
        }

        for (int i = 0; i < nRows; ++i) {
            BytePointer p = src_hsv.ptr(i);
            for (int j = 0; j < nCols; j += 3) {
                int H = p.get(j) & 0xFF;
                int S = p.get(j + 1) & 0xFF;
                int V = p.get(j + 2) & 0xFF;

                boolean colorMatched = false;

                if (H > min_h && H < max_h) {
                    int Hdiff = 0;
                    if (H > avg_h)
                        Hdiff = H - avg_h;
                    else
                        Hdiff = avg_h - H;

                    float Hdiff_p = Hdiff / diff_h;

                    float min_sv = 0;
                    if (true == adaptive_minsv)
                        min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p);
                    else
                        min_sv = minabs_sv;

                    if ((S > min_sv && S <= max_sv) && (V > min_sv && V <= max_sv))
                        colorMatched = true;
                }

                if (colorMatched == true) {
                    p.put(j, (byte) 0);
                    p.put(j + 1, (byte) 0);
                    p.put(j + 2, (byte) 255);
                } else {
                    p.put(j, (byte) 0);
                    p.put(j + 1, (byte) 0);
                    p.put(j + 2, (byte) 0);
                }
            }
        }

        // 获取颜色匹配后的二值灰度图
        MatVector hsvSplit_done = new MatVector();
        split(src_hsv, hsvSplit_done);
        Mat src_grey = hsvSplit_done.get(2);

        return src_grey;
    }

    /**
     * 判断一个车牌的颜色
     *
     * @param src
     *            车牌mat
     * @param color
     *            颜色模板
     * @param adaptive_minsv
     *            S和V的最小值由adaptive_minsv这个bool值判断
     *            <ul>
     *            <li>如果为true，则最小值取决于H值，按比例衰减
     *            <li>如果为false，则不再自适应，使用固定的最小值minabs_sv
     *            </ul>
     * @return
     */
    public static boolean plateColorJudge(final Mat src, final Color color, final boolean adaptive_minsv) {
        // 判断阈值
        final float thresh = 0.49f;

        Mat gray = colorMatch(src, color, adaptive_minsv);

        float percent = (float) countNonZero(gray) / (gray.rows() * gray.cols());

        return (percent > thresh) ? true : false;
    }

    /**
     * getPlateType 判断车牌的类型
     *
     * @param src
     * @param adaptive_minsv
     *            S和V的最小值由adaptive_minsv这个bool值判断
     *            <ul>
     *            <li>如果为true，则最小值取决于H值，按比例衰减
     *            <li>如果为false，则不再自适应，使用固定的最小值minabs_sv
     *            </ul>
     * @return
     */
    public static Color getPlateType(final Mat src, final boolean adaptive_minsv) {
        if (plateColorJudge(src, Color.BLUE, adaptive_minsv) == true) {
            return Color.BLUE;
        } else if (plateColorJudge(src, Color.YELLOW, adaptive_minsv) == true) {
            return Color.YELLOW;
        } else {
            return Color.UNKNOWN;
        }
    }

    /**
     * 获取垂直或水平方向直方图
     *
     * @param img
     * @param direction
     * @return
     */
    public static float[] projectedHistogram(final Mat img, Direction direction) {
        int sz = 0;
        switch (direction) {
            case HORIZONTAL:
                sz = img.rows();
                break;
            case VERTICAL:
                sz = img.cols();
                break;
            default:
                break;
        }
        // 统计这一行或一列中，非零元素的个数，并保存到nonZeroMat中
        float[] nonZeroMat = new float[sz];
        extractChannel(img, img, 0);
        for (int j = 0; j < sz; j++) {
            Mat data = (direction == Direction.HORIZONTAL) ? img.row(j) : img.col(j);
            int count = countNonZero(data);
            nonZeroMat[j] = count;
        }

        // Normalize histogram
        float max = 0;
        for (int j = 0; j < nonZeroMat.length; ++j) {
            max = Math.max(max, nonZeroMat[j]);
        }

        if (max > 0) {
            for (int j = 0; j < nonZeroMat.length; ++j) {
                nonZeroMat[j] /= max;
            }
        }

        return nonZeroMat;
    }

    /**
     * Assign values to feature
     * <p>
     * 样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
     *
     * @param in
     * @param sizeData - 低分辨率图像size = sizeData*sizeData, 可以为0
     * @return
     */
    public static Mat features(final Mat in, final int sizeData) {
        float[] vhist = projectedHistogram(in, Direction.VERTICAL);
        float[] hhist = projectedHistogram(in, Direction.HORIZONTAL);
        Mat lowData = new Mat();
        if (sizeData > 0) {
            resize(in, lowData, new Size(sizeData, sizeData));
        }

        int numCols = vhist.length + hhist.length + lowData.cols() * lowData.rows();
        Mat out = Mat.zeros(1, numCols, CV_32F).asMat();
        FloatIndexer idx = out.createIndexer();

        int j = 0;
        for (int i = 0; i < vhist.length; ++i, ++j) {
            idx.put(0, j, vhist[i]);
        }
        for (int i = 0; i < hhist.length; ++i, ++j) {
            idx.put(0, j, hhist[i]);
        }
        for (int x = 0; x < lowData.cols(); x++) {
            for (int y = 0; y < lowData.rows(); y++, ++j) {
                float val = lowData.ptr(x, y).get() & 0xFF;
                idx.put(0, j, val);
            }
        }

        return out;
    }

    /**
     * 显示图像
     *
     * @param title
     * @param src
     */
    public static void showImage(final String title, final Mat src) {
        if(src!=null){
            opencv_highgui.imshow(title, src);
            cvWaitKey(0);
        }
    }

}
