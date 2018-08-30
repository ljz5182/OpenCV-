package com.weijie.testopencv.controller;

import com.weijie.testopencv.constant.StateCode;
import com.weijie.testopencv.model.ResponseEntity;
import org.apache.ibatis.executor.ExecutorException;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 3:26 PM 2018/8/20
 * @Modified By:
 */

@RestController
@RequestMapping("/testInfo")
public class TestController extends BaseController {

    private static final Logger logger = LoggerFactory.getLogger(TestController.class);

    @Value("${web.upload-path}")
    private String uploadDir;



    static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

    /**
     * 编辑发票信息
     * @param body
     * @return
     */
    @RequestMapping(value = "/oneInfo", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity oneInfo(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {

//            // 特征文件
//            String feature = "/Users/liangjiazhang/Documents/Opencv_two/opencv-3.4.2/build/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml";
//
//            // image 目录
////            String imagePath = uploadDir;
//
//            // 加载特征文件
//            CascadeClassifier faceDetector = new CascadeClassifier(feature);
//
//            // 读取待识别的图片
//            Mat image = Imgcodecs.imread(uploadDir + "12.jpg",1);
//
//            MatOfRect faceDetections = new MatOfRect();
//
//            faceDetector.detectMultiScale(image, faceDetections);
//
//            for (Rect rect: faceDetections.toArray()) {
//                Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
//                        new Scalar(0,205,0));
//            }
//
//            // 输入已检测的文件
//            Imgcodecs.imwrite(uploadDir + "/out.jpg", image);





        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }
}
