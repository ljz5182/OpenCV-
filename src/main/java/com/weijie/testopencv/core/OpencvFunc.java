package com.weijie.testopencv.core;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;


import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.line;
import static org.opencv.imgproc.Imgproc.rectangle;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 9:13 AM 2018/8/22
 * @Modified By:
 */
public class OpencvFunc {



    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }


    /**
     *  opencv 读取力片的方式有很多种
     *  1.  Mat sourceImage = Imgcodecs.imread(this.p_test_file_path + "/5cent.jpg"); 这种是读取原图片
     *  2.  Mat sourceImage = Imgcodecs.imread(p_test_file_path + "/5cent.jpg",Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE); 这种方式
     *  是读取后，图像是一个 8位通道图像，也就是经过灰度处理的， 同时也可以手动处理
     *  Imgproc.cvtColor(image,image,Imgproc.COLOR_RGB2GRAY);
     *
     *
     */


    /**
     * 把感兴趣的区域勾画出来
     * @param imagePath
     * @param srcImagePath
     */
    public void testROI_one(String imagePath, String srcImagePath) {
        // 读取彩色图
        Mat sourceImage = Imgcodecs.imread(imagePath,Imgcodecs.CV_LOAD_IMAGE_COLOR);
        // 划线，设置2个点，分别为开始点，结束点，设置线条颜色
        rectangle(sourceImage,new Point(30,30),new Point(500,500) , new Scalar(0,255,0));

        Imgcodecs.imwrite(srcImagePath + "/" + "ROI_draw_area.png",sourceImage);
    }


    /**
     * 把感兴趣的区域截取出来
     * @param imagePath
     * @param srcImagePath
     */
    public void testROI_two(String imagePath, String srcImagePath) {

        // 读取彩色图
        Mat sourceImage = Imgcodecs.imread(imagePath,Imgcodecs.CV_LOAD_IMAGE_COLOR);
        /*
         * Rect 矩形  4个参数 ，开始点的 x,y 坐标， width,height 截取的宽高
         */
        Mat mat_roi = sourceImage.submat(new Rect(30,30,500,500));

        Imgcodecs.imwrite(srcImagePath + "/" + "ROI_cut_area_two.png",mat_roi);
    }


    /**
     * 用图片在原始图片上划定 ROI 区域，并替换(如添加 logo)
     * @param imagePath
     * @param srcImagePath
     */
    public void testROI_three(String imagePath, String srcImagePath) {
        // 读取彩色图
        Mat sourceImage = Imgcodecs.imread(imagePath,Imgcodecs.CV_LOAD_IMAGE_COLOR);

        // 读取logo 尽量用读取原图
        Mat logoImage = Imgcodecs.imread("/Users/liangjiazhang/Desktop/1.png",Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR);

        //根据 logo 的宽高，在原图上划定感兴趣区域
        Mat mat_roi = sourceImage.submat(new Rect(30,30,logoImage.cols(),logoImage.rows()));

        System.out.println(logoImage.channels());
        System.out.println(mat_roi.channels());

        // 在合并图像前注意，保证2个图像的通道是一致的，如果不一致需要转化
        //Imgproc.cvtColor(logoImage,newLogoImage,Imgproc.COLOR_GRAY2BGR);

        //第一第四个参数就是各自权重
        Core.addWeighted(mat_roi,0.1, logoImage, 0.9, 0., mat_roi);

        Imgcodecs.imwrite(srcImagePath + "/" + "ROI_add_area_image_three.png",sourceImage);
    }


    /**
     * 用图片在原始图片上划定 ROI 区域，并替换(如添加 logo)
     * @param imagePath
     * @param srcImagePath
     */
    public void testROI_four(String imagePath, String srcImagePath) {
        // 读取彩色图
        Mat sourceImage = Imgcodecs.imread(imagePath,Imgcodecs.CV_LOAD_IMAGE_COLOR);

        // 读取logo 尽量用读取原图
        Mat logoImage = Imgcodecs.imread("/Users/liangjiazhang/Desktop/1.png",Imgcodecs.CV_LOAD_IMAGE_ANYCOLOR);

        //原始图片转 BGRA 4通道图像（带透明层），作为 mask（遮罩）
        Mat mask = Imgcodecs.imread("/Users/liangjiazhang/Desktop/1.png",Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);

        //根据 logo 的宽高，在原图上划定感兴趣区域
        Mat mat_roi = sourceImage.submat(new Rect(30,30,logoImage.cols(),logoImage.rows()));

        logoImage.copyTo(mat_roi,mask);

        Imgcodecs.imwrite(srcImagePath + "/" + "ROI_add_area_image_four.png",sourceImage);
    }



    /**
     * 图片灰度处理即是把彩色图片转为灰度图片，目的是为了加快图片处理速度。24位彩色图像每个像素用3个字节表示，
     * 每个字节对应着RGB分量的亮度。当RGB分量值不同时，为彩色图像；当RGB分量相同时，为灰度图像。
     * 图像彩色空间转换方法：Imgproc.cvtColor(Mat src, Mat dst, int code, int dstCn)
     *
     * 参数说明：src：输入源图像
     *         dst：输出的目标图像
     *         code：code是一个掩码，表示由src到dst之间是怎么转的，比如是彩色转为灰度，还是彩色转为HSI模式。
     *         code的模式如：Imgproc.COLOR_BGR2GRAY：<彩色图像转灰度图像>
     *         dstCn：dst图像的波段数，这个值默认是0
     * @param imagePath  图片路径
     * @param srcImagePath      图片保存路径
     */
    public void colortoGrayscale(String imagePath, String srcImagePath) {

        Mat srcImage = Imgcodecs.imread(imagePath);
        Mat dstImage = new Mat();
        Imgproc.cvtColor(srcImage, dstImage, Imgproc.COLOR_BGR2GRAY,0);
        Imgcodecs.imwrite(srcImagePath +"/" + "gray.jpg", dstImage);
    }


    /**
     * 均值滤波主要是利用某像素点周边的像素的平均值来达到平滑噪声的目的。它是一种典型的线性滤波算法。均值滤波本身存在着缺陷，
     * 它不能很好的保护图像的细节，在去噪的同时会破坏图像的细节部分，不能很好的去除噪点
     * 均值滤波的方法：Imgproc.blur(Mat src, Mat dst, Size ksize, Point anchor, int borderType)
     * 参数说明：src：输入源图像
     *         dst：输出目标图像
     *         ksize：内核的大小
     *         anchor：锚点，有默认值new Point(-1,-1)，代表核的中心
     *         borderType：推断图像外部像素的边界模式，有默认值Core.BORDER_DEFAULT
     *         borderType的取值还有：
     *         BORDER_REPLICATE：复制法，既是复制最边缘像素，例如aaa|abc|ccc
     *         BORDER_REFLECT：对称法，例如cba|abc|cba
     *         BORDER_REFLECT_101：对称法，最边缘像素不会被复制，例如cb|abc|ba
     *         BORDER_CONSTANT：常量法，默认为0
     *         BORDER_WRAP：镜像对称复制
     * @param imagePath 图片路径　
     */
    public void meanFiltering(String imagePath) {
        Mat srcImage = Imgcodecs.imread(imagePath);
        Mat dstImage = srcImage.clone();
        Imgproc.blur(srcImage, dstImage, new Size(9,9),
                new Point(-1, -1), Core.BORDER_DEFAULT);
        Imgcodecs.imwrite("F:\\blur.jpg", dstImage);
    }


    /**
     * 高斯滤波器是利用高斯核的一个二维的卷积算子，用于图像模糊去噪。它也是一种线性滤波器，其模板系数会随着距离模板中心越远而越小。
     * 高斯滤波的结果和高斯分布的标准差σ有关，σ越大，平滑效果越好。高斯滤波的具体操作是：用一个模板扫描图像中的每一个像素，
     * 用模板确定的邻域内像素的加权平均值去替代模板中心像素点的值。
     * 实现高斯滤波的方法：Imgproc.GaussianBlur(Mat src, Mat dst, Size ksize, double sigmaX, double sigmaY, int borderType)
     *
     * 参数说明： src：输入源图像
     *          dst：输出目标图像
     *          ksize：内核模板大小
     *          sigmaX：高斯内核在X方向的标准偏差
     *          sigmaY：高斯内核在Y方向的标准偏差。如果sigmaY为0，他将和sigmaX的值相同，如果他们都为0，
     *          那么他们由ksize.width和ksize.height计算得出
     *          borderType： 用于判断图像边界的模式
     * @param imagePath 图片路径
     * @param srcImagePath  图片保存路径
     */
    public void gaussianFiltering(String imagePath, String srcImagePath) {

        Mat srcImage = Imgcodecs.imread(imagePath);

        Mat dstImage = srcImage.clone();

        Imgproc.GaussianBlur(srcImage, dstImage, new Size(9,9), 0, 0,
                Core.BORDER_DEFAULT);

        Imgcodecs.imwrite(srcImagePath +"/" + "GaussianBlur.jpg", dstImage);
    }



    /**
     * 中值滤波器是一种非线性的滤波技术，它将每一像素点的值设置为该点邻域窗口内所有像素点灰度值得中值。
     * 它能有效的消除椒盐噪声（椒盐噪声是由图像传感器，传输信道，解码处理等产生的黑白相间的亮暗点噪声）
     * @param imagePath 图片路径
     */
    public void medianFiltering(String imagePath) {

        Mat srcImage = Imgcodecs.imread(imagePath);

        Mat dstImage = srcImage.clone();

        Imgproc.medianBlur(srcImage, dstImage, 7);

        Imgcodecs.imwrite("F:\\medianBlur.jpg", dstImage);
    }


    /**
     * 腐蚀与膨胀是最基本的形态学操作，它们能够实现多种多样的功能，主要如下：
     1） 消除噪声
     2） 分割出独立的图像元素，在图像中连接相邻的元素
     3） 寻找图像中的明显的极大值区域或极小值区域
     4） 求出图像的梯度

     膨胀是求局部最大值的操作。本质上就是将图像A与核B进行卷积。

     腐蚀和膨胀相反，是求局部最小值。它也是需要图像A与核B进行卷积。

     实现膨胀的函数：Imgproc.dilate(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType, Scalar borderValue)
     参数说明：
     src：源图像
     dst：目标图像
     kernel：膨胀操作的核，当为Null时，表示的是使用参考点位于中心的3x3的核。我们一般使用getStructuringElement配合这个参数使用。
     anchor：锚的位置，默认值为（-1，-1），表示锚位于中心
     iterations：迭代使用膨胀的次数，默认为1
     borderType：推断外部像素的某种边界模式，默认值为BORDER_DEFAULT
     borderValue：当边界为常数时的边界值，有默认值，一般不去管它。

     实现腐蚀的函数：Imgproc.erode(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType, Scalar borderValue)
     参数说明：
     src：源图像
     dst：目标图像
     kernel：膨胀操作的核，当为Null时，表示的是使用参考点位于中心的3x3的核。我们一般使用getStructuringElement配合这个参数使用。
     anchor：锚的位置，默认值为（-1，-1），表示锚位于中心
     iterations：迭代使用膨胀的次数，默认为1
     borderType：推断外部像素的某种边界模式，默认值为BORDER_DEFAULT
     borderValue：当边界为常数时的边界值，有默认值，一般不去管它。
     函数Imgproc.getStructuringElement(int shape, Size ksize, Point anchor)会返回指定形状或尺寸的内核矩阵。
     参数shape在opencv3.2.0中有多达11种取值，这里给出三种：Imgproc.MORPH_RECT（矩形）、Imgproc.MORPH_CROSS（交叉形）、
     Imgproc.MORPH_ELLIPSE（椭圆形）。ksize和anchor分别代表内核的尺寸和锚点位置。
     * @param imagePath
     */

    public void dilateImageAnderodeImage(String imagePath) {

        Mat srcImage = Imgcodecs.imread(imagePath);

        Mat dilateImage = srcImage.clone();
        Mat erodeImage = srcImage.clone();

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        //膨胀
        Imgproc.dilate(srcImage, dilateImage, element, new Point(-1, -1), 1);
        //腐蚀
        Imgproc.erode(srcImage, erodeImage, element, new Point(-1, -1), 1);

        Imgcodecs.imwrite("F:\\dilateImage.jpg", dilateImage);
        Imgcodecs.imwrite("F:\\erodeImage.jpg", erodeImage);
    }


    /**
     * 尺寸调整顾名思义就是用来调整源图像或者ROI区域的大小。
     * 函数模型：
     Imgproc.resize(Mat src, Mat dst, Size dsize, double fx, double fy, int interpolation)
     参数说明：
     src：源图像
     dst：输出图像
     dsize：输出图像的大小。如果它为0，则计算dsize=new Size(Math.round(fx*src.cols()), Math.round(fy*src.rows()))，其中dsize、fx、fy不能同时为0
     fx：水平方向的方向系数，有默认值0。当fx=0时，会计算fx=(double) dsize.width() / src.cols()
     fy：垂直方向的方向系数，有默认值0。当fy=0时，会计算fy=(double) dsize.height() / src.rows()
     interpolation：插值方式。默认为INTER_LINEAR。可选的插值方式有：INTER_NEAREST（最邻近插值）、INTER_LINEAR（线性插值）、
     INTER_AREA（区域插值）、INTER_CUBIC（三次样条插值）、INTER_LANCZOS4（Lanczos插值）等
     * @param imagePath
     * @param srcImagePath
     */
    public void narrowImage(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);

        Mat dst = new Mat();
        Imgproc.resize(src, dst, new Size(src.cols()/2,src.rows()/2), 0, 0, Imgproc.INTER_AREA);
        Imgcodecs.imwrite(srcImagePath +"/" + "narrow.jpg", dst);

        Mat endst = new Mat();
        Imgproc.resize(src, endst, new Size(src.cols()*2,src.rows()*2), 0, 0, Imgproc.INTER_LINEAR);
        Imgcodecs.imwrite(srcImagePath +"/" + "enlarge.jpg", endst);
    }


    /**
     * 直方图均衡化是通过拉伸像素强度分布范围来增强图像对比度的一种方法。
     直方图均衡化的步骤：
     1、计算输入图像的直方图H
     2、进行直方图归一化，使直方图组距的和为255
     3、计算直方图积分
     4、采用H’作为查询表：dst(x,y)=H’(src(x,y))进行图像变换

     函数：Imgproc.equalizeHist(Mat src, Mat dst)
     参数说明：
     src：源图像
     dst：运算结果图像

     * @param imagePath
     * @param srcImagePath
     */
    public void histogramEqualization(String imagePath, String srcImagePath) {
        Mat source = Imgcodecs.imread(imagePath);
        Mat dst = new Mat();
        List<Mat> mv = new ArrayList<Mat>();
        Core.split(source, mv);
        for (int i = 0; i < source.channels(); i++) {
            Imgproc.equalizeHist(mv.get(i), mv.get(i));
        }
        Core.merge(mv, dst);
        Imgcodecs.imwrite(srcImagePath + "/" + "histogram.jpg", dst);
    }


    /**
     * Canny边缘检测的步骤：
     （1）消除噪声，一般使用高斯平滑滤波器卷积降噪
     （2）计算梯度幅值和方向，此处按照sobel滤波器步骤来操作
     （3）非极大值抑制，排除非边缘像素
     （4）滞后阈值（高阈值和低阈值），若某一像素位置的幅值超过高阈值，该像素被保留为边缘像素；若小于低阈值，则被排除；若在两者之间，该像素仅在连接到高阈值像素时被保留。推荐高低阈值比在2:1和3:1之间

     函数：Imgproc.Canny(image, edges, threshold1, threshold2, apertureSize, L2gradient);
     参数说明：
     image：输入图像，即源图像，填Mat类的对象即可，且需为单通道8位图像
     threshold1：双阀值抑制中的低阀值
     threshold2：双阀值抑制中的高阀值
     apertureSize：sobel算子模板大小，默认为3
     L2gradient：计算图像梯度幅值的标识，有默认值false,梯度幅值指沿某方向的方向导数最大的值，即梯度的模


     * @param imagePath
     * @param srcImagePath
     */
    public void cannyImage(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat dst = src.clone();
        Imgproc.GaussianBlur(src, dst, new Size(3, 3), 0);
        Imgproc.Canny(dst, dst, 40, 100);
        Imgcodecs.imwrite(srcImagePath + "/" +"canny.jpg", dst);

        // Reading the image
//        Mat src = Imgcodecs.imread(file);
//
//        // Creating an empty matrix to store the result
//        Mat gray = new Mat();
//
//        // Converting the image from color to Gray
//        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
//        Mat edges = new Mat();
//
//        // Detecting the edges
//        Imgproc.Canny(gray, edges, 60, 60*3);
//
//        // Writing the image
//        Imgcodecs.imwrite("F:/worksp/opencv/images/canny_output.jpg", edges);
    }

    /**
     * 模板匹配是一项在一幅图像中寻找与另一幅模板图像最匹配(相似)部分的技术
     * 函数：Imgproc.matchTemplate(Mat image, Mat templ, Mat result, int method)
     * 参数说明：
     * image：源图像
     * templ：模板图像
     * result：比较结果
     * method：匹配算法
     * 匹配算法：
     * TM_SQDIFF 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。
     * TM_CCORR 相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好。
     * TM_CCOEFF 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。
     * TM_SQDIFF_NORMED 归一化平方差匹配法。
     * TM_CCORR_NORMED 归一化相关匹配法。
     * TM_CCOEFF_NORMED 归一化相关系数匹配法。
     * @param imagePath
     * @param matchImagePath
     */
    public void match(String imagePath, String matchImagePath) {

        Mat g_tem = Imgcodecs.imread("F:\\mould.jpg");
        Mat g_src = Imgcodecs.imread("F:\\source.jpg");

        int result_rows = g_src.rows() - g_tem.rows() + 1;
        int result_cols = g_src.cols() - g_tem.cols() + 1;
        Mat g_result = new Mat(result_rows, result_cols, CvType.CV_32FC1);
        Imgproc.matchTemplate(g_src, g_tem, g_result, Imgproc.TM_CCORR_NORMED); // 归一化平方差匹配法
        // Imgproc.matchTemplate(g_src, g_tem, g_result,
        // Imgproc.TM_CCOEFF_NORMED); // 归一化相关系数匹配法

        // Imgproc.matchTemplate(g_src, g_tem, g_result, Imgproc.TM_CCOEFF);
        // //
        // 相关系数匹配法：1表示完美的匹配；-1表示最差的匹配。

        // Imgproc.matchTemplate(g_src, g_tem, g_result, Imgproc.TM_CCORR); //
        // 相关匹配法

        // Imgproc.matchTemplate(g_src, g_tem, g_result,Imgproc.TM_SQDIFF); //
        // 平方差匹配法：该方法采用平方差来进行匹配；最好的匹配值为0；匹配越差，匹配值越大。

        // Imgproc.matchTemplate(g_src, g_tem,g_result,Imgproc.TM_CCORR_NORMED);
        // // 归一化相关匹配法
        Core.normalize(g_result, g_result, 0, 1, Core.NORM_MINMAX, -1, new Mat());
        Point matchLocation = new Point();
        Core.MinMaxLocResult mmlr = Core.minMaxLoc(g_result);

        matchLocation = mmlr.maxLoc; // 此处使用maxLoc还是minLoc取决于使用的匹配算法
        rectangle(g_src, matchLocation,
                new Point(matchLocation.x + g_tem.cols(), matchLocation.y + g_tem.rows()),
                new Scalar(0, 0, 0, 0));

        Imgcodecs.imwrite("F:\\match.jpg", g_src);
    }


    /**
     * sobel算子主要是应用于边缘检测的一个离散的一阶差分算子，用来计算图像亮度函数的一阶梯度的近似值。
     sobel算子的计算过程：
     假设作用图像为I，
     （1）分别求得在x和y方向的导数
     水平变化：将I与一个奇数大小的内核Gx卷积
     垂直变化：将I与一个奇数大小的内核Gy卷积
     （2）在图像的每一点，求出近似梯度

     函数： Imgproc.Sobel(Mat src, Mat dst, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType)
     参数说明：
     src：源图像
     dst：检测结果图像
     ddepth：输出图像的深度
     dx：x方向上的差分阶数
     dy：y方向上的差分阶数
     ksize：sobel核的大小，默认为3
     scale：缩放因子
     delta：结果存入输出图像前可选的delta值，默认为0
     borderType：边界模式，默认BORDER_DEFAULT

     * @param imagePath
     * @param srcImagePath
     */
    public void sobelImage(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat dst = src.clone();
        Mat dstx = src.clone();
        Mat dsty = src.clone();
        Imgproc.GaussianBlur(src, dst, new Size(3, 3), 0);
        Imgproc.Sobel(dst, dstx, -1, 1, 0, 3, 1, 0, Core.BORDER_DEFAULT);
        Imgcodecs.imwrite(srcImagePath + "/" + "sobel_x.jpg", dstx);

        Imgproc.Sobel(dst, dsty, -1, 0, 1, 3, 1, 0, Core.BORDER_DEFAULT);
        Imgcodecs.imwrite(srcImagePath + "/" + "sobel_y.jpg", dsty);

        Core.addWeighted(dstx, 0.5, dsty, 0.5, 0, dst);

        Imgcodecs.imwrite( srcImagePath + "/" + "sobel.jpg", dst);
    }


    /**
     * 拉普拉斯算子是n维欧几里德空间中的一个二阶微分算子，定义为梯度（▽f）的散度（▽·f）。
     * 因此如果f是二阶可微的实函数，则f的拉普拉斯算子定义为：


     f的拉普拉斯算子也是笛卡儿坐标系中的所有非混合二阶偏导数：


     作为一个二阶微分算子，拉普拉斯算子把C函数映射到C函数，对于k ≥ 2。表达式(1)（或(2)）定义了一个算子Δ : C(R) → C(R)，或更一般地，
     定义了一个算子Δ : C(Ω) → C(Ω)，对于任何开集Ω。

     函数的拉普拉斯算子也是该函数的黑塞矩阵的迹：


     注：让一幅图像减去它的Laplacian算子可以增强对比度

     函数： Imgproc.Laplacian(Mat src, Mat dst, int ddepth, int ksize, double scale, double delta, int borderType)
     参数说明：
     src：源图像
     dst：输出图像
     ddepth：目标图像的深度
     ksize：计算二阶导数的滤波器的孔径大小，必须为正奇数，默认为1
     scale：计算Laplacian的时候可选的比例因子，默认为1
     detla：结果存入目标图之前可选的detla值，默认为0
     boederType：边界模式，默认为BORDER_DEFAULT
     * @param imagePath
     * @param srcImagePath
     */
    public void laplacianImage(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        Mat dst = src.clone();
        Imgproc.GaussianBlur(src, dst, new Size(3, 3), 0);
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Laplacian(dst, dst, -1, 3, 1, 0, Core.BORDER_DEFAULT);

        Imgcodecs.imwrite(srcImagePath + "/" + "laplacian.jpg", dst);
    }


    /**
     * 在opencv中scharr滤波器是配合sobel算子的运算而存在的。当sobel内核为3时，结果可能会产生比较明显的误差，针对这一问题，
     * Opencv提供了scharr函数。该函数只针对大小为3的核，并且运算速率和sobel函数一样快，结果更加精确，但抗噪性不如sobel函数。
     使用scharr滤波器计算x或y方向的图像差分，它的参数变量和sobel一样。

     函数： Imgproc.Scharr(Mat src, Mat dst, int ddepth, int dx, int dy, double scale, double delta, int borderType)
     参数说明：
     src：源图像
     dst：检测结果图像
     ddepth：输出图像的深度
     dx：x方向上的差分阶数
     dy：y方向上的差分阶数
     scale：缩放因子
     delta：结果存入输出图像前可选的delta值，默认为0
     borderType：边界模式，默认BORDER_DEFAULT
     * @param imagePath
     * @param srcImagePath
     */
    public void scharrFilter(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        Mat dst = src.clone();
        Mat dstx = src.clone();
        Mat dsty = src.clone();
        Imgproc.GaussianBlur(src, dst, new Size(3, 3), 0);

        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_RGB2GRAY);

        Imgproc.Scharr(dst, dstx, -1, 1, 0, 1, 0, Core.BORDER_DEFAULT);
        Imgcodecs.imwrite(srcImagePath + "/" + "scharr_x.jpg", dstx);

        Imgproc.Scharr(dst, dsty, -1, 0, 1, 1, 0, Core.BORDER_DEFAULT);
        Imgcodecs.imwrite(srcImagePath + "/" + "scharr_y.jpg", dsty);

        Core.addWeighted(dstx, 0.5, dsty, 0.5, 0, dst);
        Imgcodecs.imwrite(srcImagePath + "/" + "scharr.jpg", dst);
    }


    /**
     * 重映射
     * 通过重映射来表达每个像素的位置(x,y) :g(x,y)=f(h(x,y))，h(x,y)是映射方法函数。当h(x,y) = (I.cols()-x,y)，表示按照x轴方向发生偏转。

     函数： Imgproc.remap(Mat src, Mat dst, Mat map1, Mat map2, int interpolation, int borderMode, Scalar borderValue)
     参数说明：
     src：源图像
     dst：目标图像
     map1：它有两种可能表示的对象，一种是表示点(x,y)的第一个映射，另一种是CV_16SC2、CV_32FC1、CV_32FC2类型的X值
     map2：它有两种可能表示的对象，一种是当map1表示点(x,y)的第一个映射时，不代表任何值，另一种是CV_16UC1、CV_32FC1类型的Y值
     interpolation：插值方式，不支持INTER_AREA
     borderMode：边界模式，默认BORDER_CONTANT
     borderValue：当有常数边界时使用的值，默认为0

     * @param imagePath
     * @param srcImagePath
     */
    public void remapping(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        Mat dst = src.clone();
        Mat map_x = new Mat(src.size(), CvType.CV_32FC1);
        Mat map_y = new Mat(src.size(), CvType.CV_32FC1);
        int key = 1; // key取值1、2、3、4
        for (int i = 0; i < src.rows(); i++)
        {
            for (int j = 0; j < src.cols(); j++)
            {
                switch (key)
                {
                    case 1: // 重映射1
                        if (j > src.cols() * 0.25 && j < src.cols() * 0.75 && i > src.rows() * 0.25
                                && i < src.rows() * 0.75)
                        {
                            map_x.put(i, j, 2 * (j - src.cols() * 0.25) + 0.5);
                            map_y.put(i, j, 2 * (i - src.rows() * 0.25) + 0.5);
                        }
                        else
                        {
                            map_x.put(i, j, 0.0);
                            map_y.put(i, j, 0.0);
                        }
                        break;
                    case 2: // 重映射2
                        map_x.put(i, j, j);
                        map_y.put(i, j, src.rows() - i);
                        break;
                    case 3: // 重映射3
                        map_x.put(i, j, src.cols() - j);
                        map_y.put(i, j, i);
                        break;
                    case 4: // 重映射4
                        map_x.put(i, j, src.cols() - j);
                        map_y.put(i, j, src.rows() - i);
                        break;
                    default:
                        break;
                }
            }
        }
        Imgproc.remap(src, dst, map_x, map_y, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT,
                new Scalar(0, 0, 0));

        Imgcodecs.imwrite(srcImagePath + "/" + "remapping.jpg", dst);
    }


    /**
     * hough圆检测和hough线检测的原理近似，对于圆来说，在参数坐标系中表示为C:(x,y,r)。

     函数： Imgproc.HoughCircles(Mat image, Mat circles, int method, double dp, double minDist, double param1,
     double param2, int minRadius, int maxRadius)
     参数说明：
     image：源图像
     circles：检测到的圆的输出矢量(x,y,r)
     method：使用的检测方法，目前只有一种Imgproc.HOUGH_GRADIENT
     dp：检测圆心的累加器图像与源图像之间的比值倒数
     minDist：检测到的圆的圆心之间的最小距离
     param1：method设置的检测方法对应参数，针对HOUGH_GRADIENT，表示边缘检测算子的高阈值（低阈值是高阈值的一半），默认值100
     param2：method设置的检测方法对应参数，针对HOUGH_GRADIENT，表示累加器的阈值。值越小，检测到的无关的圆
     minRadius：圆半径的最小半径，默认为0
     maxRadius：圆半径的最大半径，默认为0（若minRadius和maxRadius都默认为0，则HoughCircles函数会自动计算半径）

     * @param imagePath
     * @param srcImagePath
     */
    public void houghCircleDetection(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        Mat dst = src.clone();
        Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGR2GRAY);
        Mat circles = new Mat();
        Imgproc.HoughCircles(dst, circles, Imgproc.HOUGH_GRADIENT, 1, 100, 440,
                50, 0, 345);
        // Imgproc.HoughCircles(dst, circles, Imgproc.HOUGH_GRADIENT, 1, 100,
        // 440, 50, 0, 0);
        for (int i = 0; i < circles.cols(); i++)
        {
            double[] vCircle = circles.get(0, i);

            Point center = new Point(vCircle[0], vCircle[1]);
            int radius = (int) Math.round(vCircle[2]);

            // circle center
            Imgproc.circle(src, center, 3, new Scalar(0, 255, 0), -1, 8, 0);
            // circle outline
            Imgproc.circle(src, center, radius, new Scalar(0, 0, 255), 3, 8, 0);
        }

        Imgcodecs.imwrite(srcImagePath + "/" + "houghCircle.jpg", src);
    }


    /**
     * 人脸检测
     * 说到人脸检测，首先要了解Haar特征分类器。Haar特征分类器说白了就是一个个的xml文件，不同的xml里面描述人体各个部位的特征值，
     * 比如人脸、眼睛等等。OpenCV3.2.0中提供了如下特征文件：
     * haarcascade_eye.xml
     haarcascade_eye_tree_eyeglasses.xml
     haarcascade_frontalcatface.xml
     haarcascade_frontalcatface_extended.xml
     haarcascade_frontalface_alt.xml
     haarcascade_frontalface_alt_tree.xml
     haarcascade_frontalface_alt2.xml
     haarcascade_frontalface_default.xml
     haarcascade_fullbody.xml
     haarcascade_lefteye_2splits.xml
     haarcascade_licence_plate_rus_16stages.xml
     haarcascade_lowerbody.xml
     haarcascade_profileface.xml
     haarcascade_righteye_2splits.xml
     haarcascade_russian_plate_number.xml
     haarcascade_smile.xml
     haarcascade_upperbody.xml

     通过加载不同的特征文件，就能达到相应的检测效果

     detectMultiScale函数参数说明：
     detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flags, Size minSize, Size maxSize)
     image：待检测图片，一般为灰度图（提高效率）
     objects：被检测物体的矩形框向量组
     scaleFactor：前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%
     minNeighbors：构成检测目标的相邻矩形的最小个数(默认为3个)
     flags：要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为CV_HAAR_DO_CANNY_PRUNING，
     那么函数将会使用Canny边缘检测来排除边缘过多或过少的区域，因此这些区域通常不会是人脸所在区域
     minSize：得到的目标区域的最小范围
     maxSize：得到的目标区域的最大范围
     *
     *
     * @param imagePath
     * @param srcImagePath
     */
     public void faceDetection(String imagePath, String srcImagePath) {

         CascadeClassifier faceDetector = new CascadeClassifier();
         // 加载人脸检测的模型文件
         faceDetector.load(
                 "/Users/liangjiazhang/Documents/Opencv_two/opencv-3.4.2/build/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml");
         Mat image = Imgcodecs.imread(imagePath);

         MatOfRect faceDetections = new MatOfRect();
         faceDetector.detectMultiScale(image, faceDetections);
         System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));
         for (Rect rect : faceDetections.toArray())
         {
             rectangle(image, new Point(rect.x, rect.y),
                     new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
         }

         String filename = srcImagePath + "/" + "ouputFace.jpg";
         Imgcodecs.imwrite(filename, image);
     }

    /**
     * 轮廓是由一系列的点组成的集合，表现在图像中就是一条曲线。
     * 查找轮廓的方法： Imgproc.findContours(Mat image, List contours, Mat hierarchy, int mode, int method, Point offset)
     *
     * 参数说明： image：8位单通道图像。
     *           contours：存储检测到的轮廓的集合。
     *           hierarchy：可选的输出向量，包含了图像轮廓的拓扑信息。
     *           mode：轮廓检索模式。有如下几种模式：
     *          1、RETR_EXTERNAL只检测最外围的轮廓
     *          2、RETR_LIST提取所有的轮廓,不建立上下等级关系,只有兄弟等级关系
     *          3、RETR_CCOMP提取所有轮廓,建立为双层结构
     *          4、RETR_TREE提取所有轮廓,建立网状结构
     *          method：轮廓的近似方法。取值如下：
     *          1、CHAIN_APPROX_NONE获取轮廓的每一个像素,像素的最大间距不超过1
     *          2、CHAIN_APPROX_SIMPLE压缩水平垂直对角线的元素,只保留该方向的终点坐标(也就是说一条中垂线a-b,中间的点被忽略了)
     *          3、CHAIN_APPROX_TC89_LI使用TEH_CHAIN逼近算法中的LI算法
     *          4、CHAIN_APPROX_TC89_KCOS使用TEH_CHAIN逼近算法中的KCOS算法
     *          offset：每个轮廓点的可选偏移量。
     * @param imagePath
     */
    public void contourDetection(String imagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        Mat dst = src.clone();
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.adaptiveThreshold(dst, dst, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY_INV, 3, 3);

        java.util.List<MatOfPoint> contours = new java.util.ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dst, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE,
                new Point(0, 0));
        System.out.println(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(src, contours, i, new Scalar(0, 0, 0, 0), 1);
        }

        Imgcodecs.imwrite("F:\\test.jpg", src);
    }


    /**
     * 基于像素的模板匹配
     *
     *  OpenCV 里的模板匹配，其原理是通过一张模板图片去另一张图中找到与模板相似部分
     *  模板匹配算法是指通过滑窗的方式在待匹配的图像上滑动，通过比较模板与子图的相似度，找到相似度最大的子图
     *
     *
     * 所谓滑窗就是通过滑动图片，使得图像块一次移动一个像素（从左到右，从上往下）。在每一个位置，都进行一次度量计算来这个图像块和原图像的
     * 特定区域的像素值相似程度。当相似度足够高时，就认为找到了目标。显然，这里“相似程度”的定义依赖于具体的计算公式给出的结果，
     * 不同算法结果也不一样。
     *
     * 目前 OpenCV 里提供了六种算法：TM_SQDIFF（平方差匹配法）、TM_SQDIFF_NORMED（归一化平方差匹配法）、
     * TM_CCORR（相关匹配法）、TM_CCORR_NORMED（归一化相关匹配法）、TM_CCOEFF（相关系数匹配法）、
     * TM_CCOEFF_NORMED（归一化相关系数匹配法）。
     *
     * @param imagePath
     * @param templatePath
     * @param saveImagePath
     */
    public void templateMatching(String imagePath, String templatePath, String saveImagePath) {
        Mat source, template;
        //将文件读入为OpenCV的Mat格式
        source = Imgcodecs.imread(imagePath); // 源图片
        template = Imgcodecs.imread(templatePath); // 模板图片
        //创建于原图相同的大小，储存匹配度
        Mat result = Mat.zeros(source.rows() - template.rows() + 1, source.cols() - template.cols() + 1,
                CvType.CV_32FC1);
        //调用模板匹配方法
        Imgproc.matchTemplate(source, template, result, Imgproc.TM_SQDIFF_NORMED);
        //规格化
        Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1);
        //获得最可能点，MinMaxLocResult是其数据格式，包括了最大、最小点的位置x、y
        Core.MinMaxLocResult mlr = Core.minMaxLoc(result);
        Point matchLoc = mlr.minLoc;
        //在原图上的对应模板可能位置画一个绿色矩形
        rectangle(source, matchLoc, new Point(matchLoc.x + template.width(), matchLoc.y + template.height()),
                new Scalar(0, 255, 0));
        //将结果输出到对应位置
        Imgcodecs.imwrite(saveImagePath + "/" + "template_result.jpeg", source);

    }


    /**
     * 此部份，基于特征点的 SURF 匹配
     */
    private float nndrRatio = 0.7f;//这里设置既定值为0.7，该值可自行调整

    private int matchesPointCount = 0;

    public float getNndrRatio() {
        return nndrRatio;
    }

    public void setNndrRatio(float nndrRatio) {
        this.nndrRatio = nndrRatio;
    }

    public int getMatchesPointCount() {
        return matchesPointCount;
    }

    public void setMatchesPointCount(int matchesPointCount) {
        this.matchesPointCount = matchesPointCount;
    }


    public void matchImage(Mat templateImage, Mat originalImage) {

        MatOfKeyPoint templateKeyPoints = new MatOfKeyPoint();
        //指定特征点算法SURF
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
        //获取模板图的特征点
        featureDetector.detect(templateImage, templateKeyPoints);
        //提取模板图的特征点
        MatOfKeyPoint templateDescriptors = new MatOfKeyPoint();
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        System.out.println("提取模板图的特征点");
        descriptorExtractor.compute(templateImage, templateKeyPoints, templateDescriptors);

        //显示模板图的特征点图片
        Mat outputImage = new Mat(templateImage.rows(), templateImage.cols(), Imgcodecs.CV_LOAD_IMAGE_COLOR);
        System.out.println("在图片上显示提取的特征点");
        Features2d.drawKeypoints(templateImage, templateKeyPoints, outputImage, new Scalar(255, 0, 0), 0);

        //获取原图的特征点
        MatOfKeyPoint originalKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint originalDescriptors = new MatOfKeyPoint();
        featureDetector.detect(originalImage, originalKeyPoints);
        System.out.println("提取原图的特征点");
        descriptorExtractor.compute(originalImage, originalKeyPoints, originalDescriptors);

        List<MatOfDMatch> matches = new LinkedList();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        System.out.println("寻找最佳匹配");
        /**
         * knnMatch方法的作用就是在给定特征描述集合中寻找最佳匹配
         * 使用KNN-matching算法，令K=2，则每个match得到两个最接近的descriptor，然后计算最接近距离和次接近距离之间的比值，
         * 当比值大于既定值时，才作为最终match。
         */
        descriptorMatcher.knnMatch(templateDescriptors, originalDescriptors, matches, 2);

        System.out.println("计算匹配结果");
        LinkedList<DMatch> goodMatchesList = new LinkedList();

        //对匹配结果进行筛选，依据distance进行筛选
        matches.forEach(match -> {
            DMatch[] dmatcharray = match.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);
            }
        });

        matchesPointCount = goodMatchesList.size();
        //当匹配后的特征点大于等于 4 个，则认为模板图在原图中，该值可以自行调整
        if (matchesPointCount >= 4) {
            System.out.println("模板图在原图匹配成功！");

            List<KeyPoint> templateKeyPointList = templateKeyPoints.toList();
            List<KeyPoint> originalKeyPointList = originalKeyPoints.toList();
            LinkedList<Point> objectPoints = new LinkedList();
            LinkedList<Point> scenePoints = new LinkedList();
            goodMatchesList.forEach(goodMatch -> {
                objectPoints.addLast(templateKeyPointList.get(goodMatch.queryIdx).pt);
                scenePoints.addLast(originalKeyPointList.get(goodMatch.trainIdx).pt);
            });
            MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
            objMatOfPoint2f.fromList(objectPoints);
            MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
            scnMatOfPoint2f.fromList(scenePoints);
            //使用 findHomography 寻找匹配上的关键点的变换
            Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);

            /**
             * 透视变换(Perspective Transformation)是将图片投影到一个新的视平面(Viewing Plane)，也称作投影映射(Projective Mapping)。
             */
            Mat templateCorners = new Mat(4, 1, CvType.CV_32FC2);
            Mat templateTransformResult = new Mat(4, 1, CvType.CV_32FC2);
            templateCorners.put(0, 0, new double[]{0, 0});
            templateCorners.put(1, 0, new double[]{templateImage.cols(), 0});
            templateCorners.put(2, 0, new double[]{templateImage.cols(), templateImage.rows()});
            templateCorners.put(3, 0, new double[]{0, templateImage.rows()});
            //使用 perspectiveTransform 将模板图进行透视变以矫正图象得到标准图片
            Core.perspectiveTransform(templateCorners, templateTransformResult, homography);

            //矩形四个顶点
            double[] pointA = templateTransformResult.get(0, 0);
            double[] pointB = templateTransformResult.get(1, 0);
            double[] pointC = templateTransformResult.get(2, 0);
            double[] pointD = templateTransformResult.get(3, 0);

            //指定取得数组子集的范围
            int rowStart = (int) pointA[1];
            int rowEnd = (int) pointC[1];
            int colStart = (int) pointD[0];
            int colEnd = (int) pointB[0];
            Mat subMat = originalImage.submat(rowStart, rowEnd, colStart, colEnd);
            Imgcodecs.imwrite("/Users/niwei/Desktop/opencv/原图中的匹配图.jpg", subMat);

            //将匹配的图像用用四条线框出来
            line(originalImage, new Point(pointA), new Point(pointB), new Scalar(0, 255, 0), 4);//上 A->B
            line(originalImage, new Point(pointB), new Point(pointC), new Scalar(0, 255, 0), 4);//右 B->C
            line(originalImage, new Point(pointC), new Point(pointD), new Scalar(0, 255, 0), 4);//下 C->D
            line(originalImage, new Point(pointD), new Point(pointA), new Scalar(0, 255, 0), 4);//左 D->A

            MatOfDMatch goodMatches = new MatOfDMatch();
            goodMatches.fromList(goodMatchesList);
            Mat matchOutput = new Mat(originalImage.rows() * 2, originalImage.cols() * 2, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            Features2d.drawMatches(templateImage, templateKeyPoints, originalImage, originalKeyPoints, goodMatches,
                    matchOutput, new Scalar(0, 255, 0), new Scalar(255, 0, 0), new MatOfByte(), 2);

            Imgcodecs.imwrite("/Users/niwei/Desktop/opencv/特征点匹配过程.jpg", matchOutput);
            Imgcodecs.imwrite("/Users/niwei/Desktop/opencv/模板图在原图中的位置.jpg", originalImage);
        } else {
            System.out.println("模板图不在原图中！");
        }

        Imgcodecs.imwrite("/Users/niwei/Desktop/opencv/模板特征点.jpg", outputImage);
    }


    public void templateMatchingOne(String imagePath, String templatePath, String saveImagePath) {
//        String templateFilePath = "/Users/niwei/Desktop/opencv/模板.jpeg";
//        String originalFilePath = "/Users/niwei/Desktop/opencv/原图.jpeg";
        //读取图片文件
        Mat templateImage = Imgcodecs.imread(templatePath, Imgcodecs.CV_LOAD_IMAGE_COLOR);
        Mat originalImage = Imgcodecs.imread(imagePath, Imgcodecs.CV_LOAD_IMAGE_COLOR);

//        ImageRecognition imageRecognition = new ImageRecognition();
//        imageRecognition.matchImage(templateImage, originalImage);
//
//        System.out.println("匹配的像素点总数：" + imageRecognition.getMatchesPointCount());

    }


    /**
     * OpenCV仿射转换
     *
     * OpenCV 函数 Imgproc.warpAffine(src, dst, tranformMatrix, size);
     *
     * 该方法接受以下参数 -
     * src - 表示此操作的源(输入图像)的Mat对象。
     * dst - 表示此操作的目标(输出图像)的Mat对象。
     * tranformMatrix - 表示变换矩阵的Mat对象。
     * size - 表示输出图像大小的整数类型变量。
     *
     * @param imagePath
     * @param srcImagePath
     */
    public void addineTranslation(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        //Creating an empty matrix to store the result
        Mat dst = new Mat();
        Point p1 = new Point( 0,0 );
        Point p2 = new Point( src.cols() - 1, 0 );
        Point p3 = new Point( 0, src.rows() - 1 );
        Point p4 = new Point( src.cols()*0.0, src.rows()*0.33 );
        Point p5 = new Point( src.cols()*0.85, src.rows()*0.25 );
        Point p6 = new Point( src.cols()*0.15, src.rows()*0.7 );
        MatOfPoint2f ma1 = new MatOfPoint2f(p1,p2,p3);
        MatOfPoint2f ma2 = new MatOfPoint2f(p4,p5,p6);
        // Creating the transformation matrix
        Mat tranformMatrix = Imgproc.getAffineTransform(ma1,ma2);
        // Creating object of the class Size
        Size size = new Size(src.cols(), src.cols());
        // Applying Wrap Affine
        Imgproc.warpAffine(src, dst, tranformMatrix, size);
        // Writing the image
        Imgcodecs.imwrite(srcImagePath + "/" + "affinetranslate.jpg", dst);
    }


    /**
     * OpenCV旋转
     *
     * OpenCV 函数 Imgproc.warpAffine(src, dst, rotationMatrix, size);
     *
     * 该方法接受以下参数 -
     * src - 表示此操作的源(输入图像)的Mat对象。
     * dst - 表示此操作的目标(输出图像)的Mat对象。
     * rotationMatrix - 表示旋转矩阵的Mat对象。
     * size - 表示输出图像大小的整数类型变量。
     *
     *
     * @param imagePath
     * @param srcImagePath
     */
    public void rotation(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        // Creating an empty matrix to store the result
        Mat dst = new Mat();
        // Creating a Point object
        Point point = new Point(300, 200);
        // Creating the transformation matrix M
        Mat rotationMatrix = Imgproc.getRotationMatrix2D(point, 30, 1);
        // Creating the object of the class Size
        Size size = new Size(src.cols(), src.cols());
        // Rotating the given image
        Imgproc.warpAffine(src, dst, rotationMatrix, size);
        // Writing the image
        Imgcodecs.imwrite(srcImagePath + "/" + "rotate_output.jpg", dst);
    }


    /**
     * 伽马校正
     * @param imagePath
     * @param srcImagePath
     */
    public void openCVGamma(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        src.convertTo(src, CvType.CV_32FC3);
        Mat i = new Mat();

        // pow(src, p, dst ) 矩阵的p次幂
        Core.pow(src, 3, i);
        Mat dst = new Mat();

        // 归一化
        Core.normalize(i, dst, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC3);

        // 保存图片
        Imgcodecs.imwrite(srcImagePath + "/" + "gamma.jpg", dst);
    }


    /**
     * OpenCV 图像处理 ------ 白平衡算法
     * @param imagePath
     * @param srcImagePath
     */
    public void openCVBalanceWhite(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        if (src.empty()) {
            System.err.println("The picture doesn't exist");
            return;
        }
        Mat dstImage = new Mat();
        List<Mat> imageChannels = new ArrayList<>();
        // 分离通道
        Core.split(src, imageChannels);
        Mat imageBlueChannel = imageChannels.get(0);
        Mat imageGreenChannel = imageChannels.get(1);
        Mat imageRedChannel = imageChannels.get(2);

        // 求各通道的平均值
        double imageBlueChannelAvg = Core.mean(imageBlueChannel).val[0];
        double imageGreenChannelAvg = Core.mean(imageGreenChannel).val[0];
        double imageRedChannelAvg = Core.mean(imageRedChannel).val[0];

        // 求出各通道所占增益
        double K = (imageBlueChannelAvg + imageGreenChannelAvg + imageRedChannelAvg) / 3;
        double Kb = K / imageBlueChannelAvg;
        double Kg = K / imageGreenChannelAvg;
        double Kr = K / imageRedChannelAvg;


        // 更新白平衡后的各通道BGR值，原来是用addWeighted()方法，为了知道清楚的了解内部的运算，写了一个方法。
        addK(imageBlueChannel, Kb);
        addK(imageGreenChannel, Kg);
        addK(imageRedChannel, Kr);

        // 使用 addWeighted() 方法，效果和 addK() 方法一样
        // Core.addWeighted(imageBlueChannel, Kb, imageBlueChannel, 0, 0,
        // imageBlueChannel);
        // Core.addWeighted(imageGreenChannel, Kg, imageGreenChannel, 0, 0,
        // imageGreenChannel);
        // Core.addWeighted(imageRedChannel, Kr, imageRedChannel, 0, 0,
        // imageRedChannel);

        Core.merge(imageChannels, dstImage);

        // Writing the image
        Imgcodecs.imwrite(srcImagePath + "/" + "balanceWhite.jpg", dstImage);

    }

    // 增加每个元素的增益值
    public void addK(Mat mat, double k) {
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                double val = mat.get(i, j)[0] * k;
                mat.put(i, j, val);
            }
        }
    }


    /**
     * opencv 图像处理 ------ 高亮图片处理
     * @param imagePath
     * @param srcImagePath
     */
    public void openCVHightlightRemove(String imagePath, String srcImagePath) {

        Mat src = Imgcodecs.imread(imagePath);
        if (src.empty()) {
            System.err.println("The picture doesn't exist");
            return;
        }
        Mat dstImage = new Mat(src.size(), src.type());

        for ( int i = 0; i < src.rows(); i++) {
            for (int j = 0; j < src.cols(); j++) {
                double B = src.get(i,j)[0];
                double G = src.get(i,j)[1];
                double R = src.get(i,j)[2];

                double alpha_r = R / (R + G + B);
                double alpha_g = G / (R + G + B);
                double alpha_b = B / (R + G + B);

                double alpha = Math.max(Math.max(alpha_r, alpha_g), alpha_b);
                double MaxC = Math.max(Math.max(R, G), B);
                double minalpha = Math.min(Math.min(alpha_r, alpha_g), alpha_b);
                double beta_r = 1 - (alpha - alpha_r) / (3 * alpha - 1);
                double beta_g = 1 - (alpha - alpha_g) / (3 * alpha - 1);
                double beta_b = 1 - (alpha - alpha_b) / (3 * alpha - 1);
                double beta = Math.max(Math.max(beta_r, beta_g), beta_b);
                double gama_r = (alpha_r - minalpha) / (1 - 3 * minalpha);
                double gama_g = (alpha_g - minalpha) / (1 - 3 * minalpha);
                double gama_b = (alpha_b - minalpha) / (1 - 3 * minalpha);
                double gama = Math.max(Math.max(gama_r, gama_g), gama_b);

                double temp = (gama * (R + G + B) - MaxC) / (3 * gama - 1);

                double[] data = new double[3];
                data[0] = B - (temp + 0.5);
                data[1] = G - (temp + 0.5);
                data[2] = R - (temp + 0.5);
                dstImage.put(i, j, data);
            }
        }

        // Writing the image
        Imgcodecs.imwrite(srcImagePath + "/" + "hightlight.jpg", dstImage);
    }


}
