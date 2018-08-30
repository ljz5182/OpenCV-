package com.weijie.testopencv.controller;

import com.weijie.testopencv.constant.Constant;
import com.weijie.testopencv.constant.StateCode;
import com.weijie.testopencv.model.LoggerEntity;
import com.weijie.testopencv.model.ResponseEntity;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.ModelAttribute;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 3:38 PM 2018/8/20
 * @Modified By:
 */

@Component
public class BaseController {

    protected final Logger logger = LoggerFactory.getLogger(getClass());

    protected HttpServletRequest request;
    protected HttpServletResponse response;


    @ModelAttribute
    public void ModelAttribute(HttpServletRequest request, HttpServletResponse response) {
        this.request = request;
        this.response = response;
    }

    /**
     * 设置请求值
     * @param requestBody
     */
    protected void setRequestBody(String requestBody) {

    }


    /**
     * 设置返回值
     * @param responseEntity
     */
    protected void setResponseEntity(ResponseEntity responseEntity) {
        LoggerEntity loggerEntity = (LoggerEntity) request.getAttribute(Constant.LOGGER_REQUEST);
        if (loggerEntity != null) {
            loggerEntity.setResponseEntity(responseEntity);
            request.setAttribute(Constant.LOGGER_REQUEST, loggerEntity);
        }
    }

    public ResponseEntity okay(Object object) {
        ResponseEntity responseEntity = new  ResponseEntity(StateCode.OK.getValue(), "Okay");
        responseEntity.setData(object);
        return responseEntity;
    }

    public ResponseEntity okay(Object object, String msg) {
        ResponseEntity responseEntity = new  ResponseEntity(StateCode.OK.getValue(), msg);
        responseEntity.setData(object);
        return responseEntity;
    }

    public ResponseEntity okay() {
        return okay(null);
    }

    public ResponseEntity error(Integer code, String msg) {
        return new ResponseEntity(code, msg);
    }
}
