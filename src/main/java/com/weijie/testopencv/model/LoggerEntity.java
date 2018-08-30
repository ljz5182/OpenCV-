package com.weijie.testopencv.model;

import java.io.Serializable;
import java.util.Map;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 3:41 PM 2018/8/20
 * @Modified By:
 */
public class LoggerEntity implements Serializable {

    /**
     * 请求地址
     */
    private String url;

    /**
     * 请求方法
     */
    private String method;


    /**
     * 请求头
     */
    private Map<String, Object> headers;


    /**
     * 请求参数
     */
    private Map<String, Object> requestParam;


    /**
     * 请求参数
     */
    private String requestBody;


    /**
     * 返回数据
     */
    private ResponseEntity responseEntity;


    /**
     * 请求消耗时间
     */
    private Long timeConsuming;


    public LoggerEntity() {

    }


    public LoggerEntity(String url, String method) {
        this.url = url;
        this.method = method;
    }

    public String getUrl() {
        return url;
    }

    public void setUrl(String url) {
        this.url = url;
    }

    public String getMethod() {
        return method;
    }

    public void setMethod(String method) {
        this.method = method;
    }

    public Map<String, Object> getHeaders() {
        return headers;
    }

    public void setHeaders(Map<String, Object> headers) {
        this.headers = headers;
    }

    public Map<String, Object> getRequestParam() {
        return requestParam;
    }

    public void setRequestParam(Map<String, Object> requestParam) {
        this.requestParam = requestParam;
    }

    public String getRequestBody() {
        return requestBody;
    }

    public void setRequestBody(String requestBody) {
        this.requestBody = requestBody;
    }

    public ResponseEntity getResponseEntity() {
        return responseEntity;
    }

    public void setResponseEntity(ResponseEntity responseEntity) {
        this.responseEntity = responseEntity;
    }

    public Long getTimeConsuming() {
        return timeConsuming;
    }

    public void setTimeConsuming(Long timeConsuming) {
        this.timeConsuming = timeConsuming;
    }
}
