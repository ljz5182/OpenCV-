package com.weijie.testopencv.constant;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 3:43 PM 2018/8/20
 * @Modified By:
 */
public enum StateCode {


    OK(0, "请求成功"),

    TOKENTIMEOUT(4, "会话过期!重新登录"),


    ERROR(3,"请求失败");

    private Integer value;
    private String reason;

    private StateCode(Integer value, String reason) {
        this.value = value;
        this.reason = reason;
    }


    public Integer getValue() {
        return value;
    }

    public String getReason() {
        return reason;
    }
}
