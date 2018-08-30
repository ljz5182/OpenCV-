package com.weijie.testopencv.util;

import java.lang.reflect.Array;
import java.util.Collection;
import java.util.Map;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 9:23 AM 2018/8/22
 * @Modified By:
 */
public class CommonUtil {

    /**
     * 可以用于判断Object,String,Map,Collection,String,Array是否为空
     */

    public static boolean isNull(Object value) {
        if (value == null) {
            return true;
        } else if (value instanceof String) {
            if (((String) value).trim().replaceAll("\\s", "").equals("")) {
                return true;
            }
        } else if (value instanceof Collection) {
            if (((Collection) value).isEmpty()) {
                return true;
            }
        } else if (value.getClass().isArray()) {
            if (Array.getLength(value) == 0) {
                return true;
            }
        } else if (value instanceof Map) {
            if (((Map) value).isEmpty()) {
                return true;
            }
        } else {
            return false;
        }
        return false;

    }


    public static boolean isNotNull(Object value) {
        return !isNull(value);
    }
}
