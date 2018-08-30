package com.weijie.testopencv.util;

import java.io.Serializable;
import java.util.UUID;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 9:22 AM 2018/8/22
 * @Modified By:
 */
public class UUIDUtil implements Serializable {


    /**
     * 获取随机 UUID 值
     * @return
     */
    public static UUID getUUID() {
        return UUID.randomUUID();
    }


    /**
     * 根据指定 name 生成 UUID
     * @param name
     * @return
     */
    public static UUID getUUID(String name) {
        return UUID.fromString(name);
    }


    /**
     * 获取 32 个字符长度的 UUID
     * @return
     */
    public static String getUUID32() {
        return getUUID36().replaceAll("-", "");
    }


    /**
     * 获取指定name 的 32 个字符长度的 UUID
     * @param name
     * @return
     */
    public static String gtUUID32(String name) {
        return getUUID36(name).replaceAll("-", "");
    }

    /**
     * 获取 36 个字符长度的 UUID
     * @return
     */
    public static String getUUID36() {
        return UUID.randomUUID().toString();
    }


    /**
     * 获取指定name 的 36 个字符长度的 UUID
     * @param name
     * @return
     */
    public static String getUUID36(String name) {
        return getUUID(name).toString();
    }



}
