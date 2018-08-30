package com.weijie.testopencv.controller;

import com.alibaba.fastjson.JSON;
import com.weijie.testopencv.constant.StateCode;
import com.weijie.testopencv.core.OpenCVBankCard;
import com.weijie.testopencv.core.OpencvFunc;
import com.weijie.testopencv.model.RequestMap;
import com.weijie.testopencv.model.ResponseEntity;
import org.apache.ibatis.executor.ExecutorException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

/**
 * @Author: liangjiazhang
 * @Description:  opencv 的一些方法
 * @Date: Created in 10:51 PM 2018/8/21
 * @Modified By:
 */

@RestController
@RequestMapping(value = "/opencv")
public class OpencvController extends BaseController{


    @Value("${web.upload-path}")
    private String uploadDir;



    /**
     * 图片转灰阶
     * @param body
     * @return
     */
    @RequestMapping(value = "/gray", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity grayImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.colortoGrayscale(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片转灰阶成功");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     *  ROI -- Region Of Interest [感兴趣区域]
     *
     *  把感兴趣的区域勾画出来
     *
     *
     * @param body
     * @return
     */
    @RequestMapping(value = "/roiOne", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity roiOneImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.testROI_one(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片区域勾画成功");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }



    /**
     *  ROI -- Region Of Interest [感兴趣区域]
     *
     *  把感兴趣的区域勾画出来
     *
     *
     * @param body
     * @return
     */
    @RequestMapping(value = "/roiThree", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity roiThreeImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.testROI_three(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片区域勾画成功");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     * 图片直方图均衡化
     * @param body
     * @return
     */
    @RequestMapping(value = "/histogram", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity histogramEqualization(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.histogramEqualization(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片转直方图均衡化");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     * Canny边缘检测
     * @param body
     * @return
     */
    @RequestMapping(value = "/canny", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity cannyImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.cannyImage(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片转Canny边缘检测");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }

    /**
     * sobel算子
     * @param body
     * @return
     */
    @RequestMapping(value = "/sobel", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity sobelImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.sobelImage(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片转sobel算子");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }

    /**
     * Laplacian算子
     * @param body
     * @return
     */
    @RequestMapping(value = "/laplacian", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity laplacianImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.laplacianImage(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片转Laplacian算子");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     * scharr滤波器
     * @param body
     * @return
     */
    @RequestMapping(value = "/scharr", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity scharrFilterImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.scharrFilter(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片转scharr滤波器");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     * 重映射
     * @param body
     * @return
     */
    @RequestMapping(value = "/remapping", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity remappingImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.remapping(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片之重映射");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }

    /**
     * hough圆检测
     * @param body
     * @return
     */
    @RequestMapping(value = "/houghCircle", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity houghCircleDetectionImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.houghCircleDetection(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "图片之hough圆检测");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }

    /**
     * 人脸检测
     * @param body
     * @return
     */
    @RequestMapping(value = "/faceDetection", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity faceDetectionImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.faceDetection(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "人脸检测成功");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }



    /**
     * 基于像素的模板匹配
     * @param body
     * @return
     */
    @RequestMapping(value = "/templateMatching", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity templateMatchingImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null && requestMap.getTempImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.templateMatching(requestMap.getImagePath(),requestMap.getTempImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "基于像素的模板匹配成功");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     * 仿射转换
     *
     *
     * @param body
     * @return
     */
    @RequestMapping(value = "/affineTranslation", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity affineTranslationImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.addineTranslation(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "OpenCV图片仿射转换成功");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     * OpenCV旋转
     *
     *
     * @param body
     * @return
     */
    @RequestMapping(value = "/rotation", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity rotationImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpencvFunc opencvFunc = new OpencvFunc();
                opencvFunc.rotation(requestMap.getImagePath(), uploadDir);
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "OpenCV图片旋转成功");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     * OpenCV 识别银行卡
     *
     *
     * @param body
     * @return
     */
    @RequestMapping(value = "/ocrBankCard", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity ocrBankCardImage(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {
                OpenCVBankCard openCVBankCard = new OpenCVBankCard();
//                openCVBankCard.imageCVTColor(requestMap.getImagePath(), uploadDir);
                openCVBankCard.procSrcTwoGray(40, 400, requestMap.getImagePath(),uploadDir);
//                openCVBankCard.trainTwoDataFun("/Users/liangjiazhang/Documents/uploads/0one.jpg");
                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "OpenCV 识别银行卡");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }


    /**
     * OpenCV 图片白平衡
     *
     *
     * @param body
     * @return
     */
    @RequestMapping(value = "/balanceWhite", method = RequestMethod.POST, consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity balanceWhite(@RequestBody String body) {
        super.setRequestBody(body);
        ResponseEntity responseEntity = null;
        try {
            RequestMap requestMap = JSON.parseObject(body,RequestMap.class);
            if (requestMap.getImagePath() != null) {

                OpencvFunc opencvFunc = new OpencvFunc();

                opencvFunc.openCVBalanceWhite(requestMap.getImagePath(), uploadDir);

                Map<String, Object> data = new HashMap<String, Object>();
                data.put("value", "null");
                responseEntity = okay(data, "OpenCV 图片白平衡处理成功");
            }

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }



}
