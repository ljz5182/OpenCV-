package com.weijie.testopencv.model;

import java.io.Serializable;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 9:32 AM 2018/8/22
 * @Modified By:
 */
public class RequestMap implements Serializable {

    private String imagePath;

    private String tempImagePath;

    public String getTempImagePath() {
        return tempImagePath;
    }

    public void setTempImagePath(String tempImagePath) {
        this.tempImagePath = tempImagePath;
    }

    public String getImagePath() {
        return imagePath;
    }

    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }
}
