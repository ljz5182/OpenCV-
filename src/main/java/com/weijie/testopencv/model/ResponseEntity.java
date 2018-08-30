package com.weijie.testopencv.model;

import java.io.Serializable;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 3:40 PM 2018/8/20
 * @Modified By:
 */
public class ResponseEntity implements Serializable {

    private Integer state;

    private String msg;

    private Object data;

    public Integer getState() {
        return state;
    }

    public void setState(Integer state) {
        this.state = state;
    }

    public String getMsg() {
        return msg;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }

    public Object getData() {
        return data;
    }

    public void setData(Object data) {
        this.data = data;
    }



    public ResponseEntity() {

    }

    public ResponseEntity(Integer state, String msg) {


        this.msg = msg;
        this.state = state;
    }

    public ResponseEntity(Integer state, String msg, Object data) {
        this.state = state;
        this.msg = msg;
        this.data = data;
    }
}
