package com.weijie.testopencv.core;

import com.weijie.testopencv.util.ImageOpenCVUtil;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.awt.Color.gray;
import static org.bytedeco.javacpp.opencv_imgproc.CV_THRESH_BINARY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_THRESH_OTSU;
import static org.opencv.imgproc.Imgproc.THRESH_TOZERO;
import static org.opencv.imgproc.Imgproc.threshold;

/**
 * @Author: liangjiazhang
 * @Description:  OpenCV 银行卡识别
 * @Date: Created in 3:47 PM 2018/8/24
 * @Modified By:
 */
public class OpenCVBankCard {


    private static final int BLACK = 0;
    private static final int WHITE = 255;

    int numX1 = 0, numX2 = 0;
    int aa[], bb[];
    int bankX[];//19个银行卡号的位置信息----x
    int bankNum[];//19个银行卡号
    int classes = 10, train_samples = 1, K = 1;
    int new_width = 32;
    int new_height = 32;
    KNearest knn ;
    Mat trainData, trainClasses, testData;
    List<Integer> trainLabs = new ArrayList<Integer>(), testLabs = new ArrayList<Integer>();


    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }


    /**
     * OpenCV 银行卡识别 步骤: 1. 先将图片灰度化；
     *                       2. 二值处理化
     *                       3. 图像腐蚀
     *                       4. 数字分割
     *
     *
     *
     */



    public void imageCVTColor(String imageColor,  String saveImagePath) {

        /**
         * 提取银行卡轮廊:  1 高斯模糊-》灰度化-》Canny边缘检测-》二值化-》找轮廓-》轮廓判断
         *
         * 第三种更好的方法:  灰度化-》直方图增强对比度-》特定阈值二值化
         *
         */

//
//        // 图片灰度化
//        Mat srcImage = Imgcodecs.imread(imageColor);
//        Mat dstImage = new Mat();
//        Imgproc.cvtColor(srcImage, dstImage, Imgproc.COLOR_BGR2GRAY,0);
//
//        // BufferedImage转mat
////        BufferedImage src = ImageIO.read(input);
////        Mat srcMat = new Mat(src.getHeight(), src.getWidth(), CvType.CV_8UC3);
//
//        Mat bin = new Mat();
//        threshold(dstImage,bin,80,255,THRESH_TOZERO); 	//图像二值化
//        Imgcodecs.imwrite(saveImagePath +"/" + "threshold.jpg", bin);


        // 读取图片， 高斯模糊
        Mat srcImage = Imgcodecs.imread(imageColor);
        Mat dstImage = srcImage.clone();
        Imgproc.GaussianBlur(srcImage, dstImage, new Size(9,9), 0, 0,
                Core.BORDER_DEFAULT);
//        Imgcodecs.imwrite(saveImagePath +"/" + "threshold_one.jpg", dstImage);
        Mat huiduImage = new Mat();

        // 图片灰度化
        Imgproc.cvtColor(dstImage, huiduImage, Imgproc.COLOR_BGR2GRAY,0);
//        Imgcodecs.imwrite(saveImagePath +"/" + "threshold_two.jpg", huiduImage);

        // Canny 边缘检测
//        Imgproc.Canny(huiduImage, huiduImage, 40, 80);
//        Imgcodecs.imwrite(saveImagePath + "/" +"canny_bankcard.jpg", huiduImage);

        // 二值化
        Imgproc.threshold(huiduImage,huiduImage, 0,255, CV_THRESH_OTSU + CV_THRESH_BINARY);
//        Imgcodecs.imwrite(saveImagePath + "/" +"threshold_bankcard.jpg", huiduImage);


        // 图像腐蚀---腐蚀后变得更加宽,粗.便于识别--使用3*3的图片去腐蚀
        Mat destMat = new Mat(); //腐蚀后的图像
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 1));
        Imgproc.erode(huiduImage,destMat,element);
//        BufferedImage destImage =  toBufferedImage(destMat);
//        saveJpgImage(destImage,"E:/destImage.jpg");
        Imgcodecs.imwrite(saveImagePath + "/" +"element.jpg", huiduImage);
        System.out.println("保存腐蚀化后图像！");


    }









    public float doOCR(Mat mat) {
        Mat pimage = new Mat();
        Mat data = new Mat();
        Imgproc.resize(mat, pimage, new Size(new_width, new_height));
        Mat image = new Mat(new Size(pimage.width(), pimage.height()),CvType.CV_32FC1);
        pimage.convertTo(image, CvType.CV_32FC1, 0.0039215);
        //Mat2IntArr(image);
        data = image.reshape(0, 1);
        Mat nearest = new Mat(new Size(K, 1), CvType.CV_32FC1);
        Mat results = new Mat();
        Mat dists = new Mat();
        //Mat2IntArr(data);
        float res = knn.findNearest(data, K, results, nearest, dists);
        return res;
    }


    //对银行卡进行排序
    public void sortBankNum() {
        int i,j,temp;
        for(j=0;j<=17;j++) {
            for (i=0;i<17-j;i++)
                if (bankX[i]>bankX[i+1]) {
                    temp=bankX[i];
                    bankX[i]=bankX[i+1];
                    bankX[i+1]=temp;
                    temp=bankNum[i];
                    bankNum[i]=bankNum[i+1];
                    bankNum[i+1]=temp;
                }
        }
        /*bankNum[0] = 6;
        bankNum[1] = 2;
        bankNum[2] = 2;
        bankNum[3] = 8;
        bankNum[4] = 4;
        bankNum[5] = 8;*/
    }



    public Mat filterImage(Mat imgSrc, Mat src, int t2) {

        // 初始化数据
        aa = new int[18];
        bb = new int[18];
        for (int i = 0; i < 18; i++) {
            aa[i] = 0;
            bb[i] = 0;
        }
        bankX = new int[18];
        bankNum = new int[18];


        //获取截图的范围--从第一行开始遍历,统计每一行的像素点值符合阈值的个数,再根据个数判断该点是否为边界
        //判断该行的黑色像素点是否大于一定值（此处为150）,大于则留下,找到上边界,下边界后立即停止
        int a =0, b=0, state = 0;
        for (int y = 0; y < imgSrc.height(); y++)//行
        {
            int count = 0;
            for (int x = 0; x < imgSrc.width(); x++) //列
            {
                //得到该行像素点的值
                byte[] data = new byte[1];
                imgSrc.get(y, x, data);
                if (data[0] == 0)
                    count = count + 1;
            }
            if (state == 0)//还未到有效行
            {
                if (count >= 150)//找到了有效行
                {//有效行允许十个像素点的噪声
                    a = y;
                    state = 1;
                }
            }
            else if (state == 1)
            {
                if (count <= 150)//找到了有效行
                {//有效行允许十个像素点的噪声
                    b = y;
                    state = 2;
                }
            }
        }
        numX1 = a;
        numX2 = b;
        System.out.println("过滤下界"+Integer.toString(a));
        System.out.println("过滤上界"+Integer.toString(b));

        //参数,坐标X,坐标Y,截图宽度,截图长度
        Rect rect = new Rect(0,a,imgSrc.width(),b - a);
        Mat resMat = new Mat(imgSrc,rect);
        Imgcodecs.imwrite("/Users/liangjiazhang/Documents/uploads" + "/" +"one_orig.jpg", resMat);

        return resMat;
    }



//    public Mat cutMat(Mat imageSrc) {
//
//        int a = 0, b = 0;//保存有效行号
//        int h = 0;
//        int state = 0;//标志位，0则表示还未到有效行，1则表示到了有效行,2表示搜寻完毕
//        for (int y = 0; y < imageSrc.height(); y++) {
//            int count = 0;
//            for (int x = 0; x < imageSrc.width(); x++) {
//                //System.out.println("ok");
//                byte[] data = new byte[1];
//                imageSrc.get(y, x, data);
//                //System.out.println("ok2");
//                if (data[0] == 0)
//                    count = count + 1;
//            }
//            if (state == 0)//还未到有效行
//            {
//                if (count >= 10)//找到了有效行
//                {//有效行允许十个像素点的噪声
//                    a = y;
//                    state = 1;
//                }
//            } else if (state == 1) {
//                if (count <= 3)//找到了有效行
//                {//有效行允许十个像素点的噪声
//                    b = y;
//                    state = 2;
//                    break;
//                }
//            }
//        }
//        System.out.println("MyLogcat" + "+" + String.valueOf(imageSrc.width())+"+"+ String.valueOf(b - a));
//        Rect roi = new Rect(0, a, imageSrc.width(), b - a);
//        Mat res = new Mat(new Size(roi.width, roi.height),CvType.CV_8UC1);
//        res = imageSrc.submat(roi);
//        System.out.println("截取成功");
//
//        return res;
//    }


    public int procSrcTwoGray(int t, int t2, String imagePath, String saveImagePath) {

//        Mat res = Imgcodecs.imread(imagePath);
//        float ret = doOCR(res);
//
//        return 1;
       // Mat rgbMat =  new  Mat(); //原图
        Mat rgbMat = Imgcodecs.imread(imagePath);
        Mat grayMat =  new  Mat();  //灰度图
        Mat binaryMat = new Mat(); //二值化图
        Mat erode = new Mat();
        Mat last = new Mat();
        Imgproc.resize(rgbMat, rgbMat, new Size(589, 374));
        Imgproc.cvtColor(rgbMat, grayMat, Imgproc.COLOR_RGB2GRAY); //rgbMat to gray grayMat
        //继续预处理
        Imgproc.threshold(grayMat, binaryMat, t, 255, Imgproc.THRESH_BINARY);//二值化

        Imgproc.medianBlur(binaryMat, binaryMat, 3);
        Size size = new Size(3, 3);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size);
        Imgproc.erode(binaryMat, erode, element);//腐蚀图像
        Imgcodecs.imwrite(saveImagePath + "/" +"one_resMat.jpg", binaryMat);
        System.out.println( "进入Filter" );
        last = filterImage(erode,grayMat, t2);
        System.out.println( "完成Filter" + last);
        Imgcodecs.imwrite(saveImagePath + "/" +"one_last.jpg", last);
        // 得到裁剪后的图片，就是只有数字的, 接下来就是分割图片
        boolean a1;
        if (last != null) {
            System.out.println("进入findRect" );
            a1 = findRect(last);
        } else {
            a1 = false;
        }
        if(a1) {
            System.out.println( "完成findRect1111" );
//            trainData = new Mat(new Size(new_width * new_height, classes * train_samples),CvType.CV_32FC1);
//            trainClasses = new Mat(new Size(1, classes * train_samples),CvType.CV_32FC1);


//            Mat sampleIdx = new Mat();
            //boolean a = knn.train(trainData, trainClasses, sampleIdx, false, K, false);
//            boolean a = knn.train(trainData, trainClasses);

            //float ret = do_ocr(binaryMat);
            //System.out.println(String.valueOf(ret));



            trainData = new Mat();
            testData = new Mat();
            for (int y = 0 ; y < 10; y ++) {
                trainLabs.add(y);
                testLabs.add(y);
            }
            //分割得到数字
            Mat orig = new Mat();
            last.copyTo(orig);
            int c = 0, hehe = 0, error = 0;
            for (int i = 0; i < 18; i++) {
                c++;
                hehe++;
                if (bb[i] - aa[i] < 5) {
                    System.out.println("忽略" );
                    bankX[i] = 0;
                    bankNum[i] = 0;
                    error++;
                    continue;
                }
                Rect roi = new Rect(aa[i], 0, bb[i] - aa[i], numX2 - numX1);
                bankX[i] = aa[i];//保存位置信息
                Mat res = new Mat();
                res = orig.submat(roi);
                if(c == 7) {
                    //grayBitmap = Bitmap.createBitmap(res.width(), res.height(), Config.RGB_565);
                    //Utils.matToBitmap(res, grayBitmap);  //convert mat to bitmap
                    //Mat2IntArr(res);
                }
                System.out.println("本次识别数是："+ res);
                Imgcodecs.imwrite(saveImagePath + "/" + i +"one.jpg", res);




                // crop out 1 digit:
                Mat digit = Imgcodecs.imread(saveImagePath + "/" + i +"one.jpg");
                Mat num = new Mat();
                Imgproc.resize(digit, num, new Size(29, 20));
                // we need float data for knn:
                num.convertTo(num, CvType.CV_32F);
                // for opencv ml, each feature has to be a single row:
                trainData.push_back(num.reshape(1,1));


                // Add test image
                Mat testDigit = Imgcodecs.imread(saveImagePath + "/" + i +"one.jpg", 0);
                Mat testNum = new Mat();
                Imgproc.resize(testDigit, testNum, new Size(29, 20));
                // we need float data for knn:
                num.convertTo(testNum, CvType.CV_32F);
                testData.push_back(num.reshape(1,1));
//                testLabs.add(6);

                // make a Mat of the train labels, and train knn:
                knn = KNearest.create();
                knn.train(trainData, Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(trainLabs));


                // now test predictions:
                Mat one_feature = testData.row(i);
                int testLabel = testLabs.get(i);
                Mat res_last = new Mat();
                float p = knn.findNearest(one_feature, 1, res_last);
                System.out.println(testLabel + " " + p + " " + res_last.dump());


            }
            System.out.println("本次识别失败数是："+String.valueOf(error));
            System.out.println("进入sortBankNum" );
//            coutIntArr(bankNum, bankX);

            sortBankNum();
//            coutIntArr(bankNum, bankX);
            System.out.println("完成sortBankNum" );

            String x = new String();
            x = "622848";
            int finish = 0;
            for (int i = 0; i < 18; ++i) {
                if(i + error > 17) {
                    break;
                }
                if(bankX[i + error] != 0) {
                    x = x + String.valueOf(bankNum[i + error]);
                    System.out.println(String.valueOf(bankNum[i + error]) + "+" +  String.valueOf(i + error) );
                    finish++;
                    if(finish >= 13) {
                        break;
                    }
                }
            }

            System.out.println( "procSrc2Gray sucess..." + error);
            return error;
        } else {
            return 1;
        }
    }


    public boolean findRect(Mat srcA) {
        //将银行数字分割，黑点为前景点，白点为背景点。
        int count = 0;
        int judgeA = 0;//0代表当前要判断有效列，1代表当前要判断无效列
        int i = 0;
        Mat src = new Mat();
        if (srcA == null) {
            return false;
        }
        Imgproc.medianBlur(srcA, src, 3);
        for (int x = 0; x < src.width(); x++) {
            count  = 0;
            for (int y = 0; y < src.height(); y++) {
                byte[] data = new byte[1];
                src.get(y, x, data);
                if (data[0] == 0)//遇到前景点就加1
                {
                    count ++;
                }
            }
            if (judgeA == 0) {
                if (count >= 5)//有效行到了
                {
                    judgeA = 1;
                    if (i > 17)
                    {
                        i = 17;
                    }
                    aa[i] = x;
                }
            } else if(judgeA == 1) {
                if (count < 5) {
                    judgeA = 0;
                    if (i > 17) {
                        i = 17;
                    }
                    bb[i] = x;
                    i++;
                }
            }
        }
        System.out.println("444444:" + aa);
        System.out.println("66666:" + bb);
        return true;
    }


    public float[] Mat2IntArr(Mat mat) {
        int h = mat.height();
        int w = mat.width();
        float result[] = new float[h*w];
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                float[] data = new float[1];
                mat.get(i, j, data);
                if( data[0] == 0)
                    result[w * i + j] = 0;
                else
                    result[w * i + j] = 255;
                System.out.println("("+String.valueOf(i)+","+String.valueOf(j)+")"+String.valueOf((float)data[0]));
            }
        }
        System.out.println(String.valueOf(h)+","+String.valueOf(w));
        return result;
    }


    public Mat changeMat(Mat mat) {
        int h = mat.height();
        int w = mat.width();
        //int result[] = new int[h*w];
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                byte[] data = new byte[1];
                mat.get(i, j, data);
                if( data[0] != 0) {
                    data[0] = (byte)255;
                    mat.put(i, j, data);
                }
            }
        }
        return mat;
    }


    public void trainTwoDataFun(String srcImage) {

        // 每个数字样例大小为 20x20，故每行有 2000/20=100个数字样例，从0~9的每个数字依次占 5 行，10个数字共 50 行。
//        Mat digits = Imgcodecs.imread(srcImage, 0);

        Mat digits = Imgcodecs.imread("/Users/liangjiazhang/Documents/Opencv_two/opencv-3.4.2/samples/data/digits.png",
                0);

        // setup train/test data:
        Mat trainData = new Mat();
        Mat testData = new Mat();
        List<Integer> trainLabs = new ArrayList<Integer>();
        List<Integer> testLabs = new ArrayList<Integer>();
        // 10 digits a 5 rows:
        // 一次行读入所有数据，训练数据和测试数据各一半
        // 数据集共 50 行
        for (int r=0; r<20; r++) {
            // 100 digits per row:
            // 每行 100 个数字样例
            for (int c=0; c<18; c++) {
                // crop out 1 digit:
                // 每次读入一个数字样本，大小：20x20
                Mat num = digits.submat(new Rect(c*20,r*20,20,20));
                // we need float data for knn:
                // knn算法的输入为浮点型，在此转换
                num.convertTo(num, CvType.CV_32F);
                // 50/50 train/test split:
                if (c % 2 == 0) {
                    // 偶数行作为训练样本
                    // for opencv ml, each feature has to be a single row:
                    // OpenCV中的ml算法要求输入训练数据的每一个样本（此例中可理解为样本特征：feature）占据一行
                    // num 本身为 20x20 的矩阵，转换为 1 行 n列的矩阵，其中 n=20x20=400
                    trainData.push_back(num.reshape(1,1));
                    // add a label for that feature (the digit number):
                    // 对应每个训练样本建立对应的Label（正确答案）
                    trainLabs.add(r/5);
                } else {
                    // 奇数行作为测试数据
                    testData.push_back(num.reshape(1,1));
                    testLabs.add(r/5);
                }
            }
        }

        // make a Mat of the train labels, and train knn:
        // 构建 KNearest 对象，3.0版本之前似乎可以直接 new 一个
        KNearest knn = KNearest.create();
        // train函数原型定义为：public  boolean train(Mat samples, int layout, Mat responses)
        // 其中第一个参数自不必说，是输入的训练数据，其中每一行代表一个样本（特征）；第二个参数是指定样本的布局，是每行一个样本，
        // 还是每列一个样本，没有默认值；第三个参数为训练样本对应的正确答案（label）
        // 扩展：Converters类中有OpenCV提供的许多数据（类型）转换工具，非常实用
        knn.train(trainData, Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(trainLabs));
        // 使用测试数据集进行测试
        // 测试数据也是每行一个样本
        // now test predictions:
        int err = 0;
        for (int i=0; i<testData.rows(); i++) {
            // 读取一行（一个测试样本）
            Mat one_feature = testData.row(i);
            // 预期的（标准/正确）答案
            int testLabel = testLabs.get(i);
            Mat res = new Mat();
            // 查找匹配：第一个参数为输入样本（可一次输入多个样本），第二个参数为需要返回的K个邻近（即KNearest的那个K），第三个参数为返回
            // 结果（res: result），结果为样本对应的Label值，每一个样本的匹配结果对应一行。
            // 如果输入仅有一个样本，则返回结果（p）就是预测结果。参数 1即为K-近邻算法的关键参数 K！
            float p = knn.findNearest(one_feature, 1, res);
            System.out.println(testLabel + " " + p + " " + res.dump());
            System.out.println("123结果:  " + p );
            // 统计识别误差
            int iRes = (int) p;
            if(iRes != testLabel) {
                err++;
            }

        }

        // 输出（屏幕提示）识别精度

        float accuracy = (float) ((2500 - (float)err) / 2500.0);

        DecimalFormat df = new DecimalFormat("0.0000");

        System.out.println("error count: " + err + ", accuracy is: " + df.format(accuracy));
    }


    // 图像切割,水平投影法切割
    public List<Mat> cutImgX(Mat srcImageMat) {

        ImageOpenCVUtil imageOpenCVUtil  = new ImageOpenCVUtil(srcImageMat);
        int i, j;
        int nWidth = imageOpenCVUtil.getWidth(), nHeight = imageOpenCVUtil.getHeight();
        int[] xNum = new int[nHeight], cNum;
        int average = 0;// 记录像素的平均值

        // 统计出每行黑色像素点的个数
        for (i = 0; i < nHeight; i++) {
            for (j = 0; j < nWidth; j++) {
                if (imageOpenCVUtil.getPixel(i, j) == BLACK) {
                    xNum[i]++;
                }
            }
        }

        // 经过测试这样得到的平均值最优
        cNum = Arrays.copyOf(xNum, xNum.length);
        Arrays.sort(cNum);
        for (i = 31 * nHeight / 32; i < nHeight; i++) {
            average += cNum[i];
        }
        average /= (nHeight / 32);

        // 把需要切割的y点都存到cutY中
        List<Integer> cutY = new ArrayList<Integer>();
        for (i = 0; i < nHeight; i++) {
            if (xNum[i] > average) {
                cutY.add(i);
            }
        }

        // 优化cutY把
        if (cutY.size() != 0) {

            int temp = cutY.get(cutY.size() - 1);
            // 因为线条有粗细,优化cutY
            for (i = cutY.size() - 2; i >= 0; i--) {
                int k = temp - cutY.get(i);
                if (k <= 8) {
                    cutY.remove(i);
                } else {
                    temp = cutY.get(i);

                }

            }
        }

        // 把切割的图片都保存到YMat中
        List<Mat> YMat = new ArrayList<Mat>();
        for (i = 1; i < cutY.size(); i++) {
            // 设置感兴趣的区域
            int startY = cutY.get(i - 1);
            int height = cutY.get(i) - startY;
            Mat temp = new Mat(srcImageMat, new Rect(0, startY, nWidth, height));
            Mat t = new Mat();
            temp.copyTo(t);
            YMat.add(t);
        }

        return YMat;
    }



    // 图像切割,垂直投影法切割
    public List<Mat> cutImgY(Mat srcImageMat) {

        ImageOpenCVUtil imageOpenCVUtil  = new ImageOpenCVUtil(srcImageMat);
        int i, j;
        int nWidth = imageOpenCVUtil.getWidth(), nHeight = imageOpenCVUtil.getHeight();
        int[] xNum = new int[nWidth], cNum;
        int average = 0;// 记录像素的平均值
        // 统计出每列黑色像素点的个数
        for (i = 0; i < nWidth; i++) {
            for (j = 0; j < nHeight; j++) {
                if (imageOpenCVUtil.getPixel(j, i) == BLACK) {
                    xNum[i]++;
                }

            }
        }

        // 经过测试这样得到的平均值最优 , 平均值的选取很重要
        cNum = Arrays.copyOf(xNum, xNum.length);
        Arrays.sort(cNum);
        for (i = 31 * nWidth / 32; i < nWidth; i++) {
            average += cNum[i];
        }
        average /= (nWidth / 28);

        // 把需要切割的x点都存到cutY中,
        List<Integer> cutX = new ArrayList<Integer>();
        for (i = 0; i < nWidth; i += 2) {
            if (xNum[i] >= average) {
                cutX.add(i);
            }
        }

        if (cutX.size() != 0) {

            int temp = cutX.get(cutX.size() - 1);
            // 因为线条有粗细,优化cutY
            for (i = cutX.size() - 2; i >= 0; i--) {
                int k = temp - cutX.get(i);
                if (k <= 10) {
                    cutX.remove(i);
                } else {
                    temp = cutX.get(i);

                }

            }
        }

        // 把切割的图片都保存到YMat中
        List<Mat> XMat = new ArrayList<Mat>();
        for (i = 1; i < cutX.size(); i++) {
            // 设置感兴趣的区域
            int startX = cutX.get(i - 1);
            int width = cutX.get(i) - startX;
            Mat temp = new Mat(srcImageMat, new Rect(startX, 0, width, nHeight));
            Mat t = new Mat();
            temp.copyTo(t);
            XMat.add(t);
        }

        return XMat;
    }
}
