package com.weijie.testopencv.constant;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 3:43 PM 2018/8/20
 * @Modified By:
 */
public class Constant {


    public final static String JWTSECRET = "shenzhenfourpay";



    public final static String JWTISSUER = "payAdmin";

    /**
     * 接口请求参数
     */
    public final static String LOGGER_REQUEST = "LOGGER_REQUEST";
    /**
     * 接口请求开始时间
     */
    public final static String LOGGER_REQUEST_START = "LOGGER_REQUEST_START";


    /**
     *   用户部份
     *
     */

    public static final Integer USER_UN_REGISTER = 1001;   // 用户没有注册

    public static final Integer USER_UN_REGISTER_ERROR = 1002;   // 用户注册失败

    public static final Integer USER_LOGIN_ERROR = 1003;   // 用户登录失败

    public static final Integer USER_TOKEN_TIMEOUT = 1004;  // token 过期


    public static final Integer RANDOM_MIN   = 0; // 随机数的最小值 //  random

    public static final Integer RANDOM_MAX   = 100; // 随机数的最大值

    public static final Integer RANDOM_RANGE   = 6; // 随机数的位数


    /**
     * 操作部份
     */
    public static final Integer ACTIVEERROR  =  4005;  // 数据库操作失败



    /**
     * token错误
     */
    public static final Integer TOKENEERROR  =  2000;

    /**
     * 参数错误
     */
    public static final Integer  PARAMETERERROR  =  2001;




    // 用户信息
    public static String USER_INFO = "USER_INFO";//缓存中用到,（key+商户号）缓存商户信息
    // 用户角色权限信息
    public static String ROLE_PERMISSION_INFO = "ROLE_PERMISSION_INFO";//缓存中用到,（key+商户号）缓存用户角色权限信息
    // 角色所拥有的权限信息
    public static String PERMISSION_INFO = "PERMISSION_INFO";//缓存中用到,（key+角色ID）缓存角色所拥有的权限信息
    // 所有权限信息
    public static String ALL_PERMISSION_INFO = "All_PERMISSION_INFO";

    // 角色所拥有的权限的url
    public static String PERMISSION_URL_INFO = "PERMISSION_URL_INFO";

    // 用户的JWT签名
    public static String JWTSIGN_INFO = "JWTSIGN_INFO";




    public static final String ACA_ORIGIN = "*";
    public static final String ACA_METHOD = "*";
    public static final String ACA_HEADERS = "Content-Type,Authorization,Accept,Accept-Language,Content-Language";
}
