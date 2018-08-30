package com.weijie.testopencv.controller;

import com.weijie.testopencv.constant.StateCode;
import com.weijie.testopencv.model.ResponseEntity;
import com.weijie.testopencv.util.CommonUtil;
import com.weijie.testopencv.util.SaveFileUtil;
import org.apache.ibatis.executor.ExecutorException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.ArrayUtils;


import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * @Author: liangjiazhang
 * @Description:
 * @Date: Created in 9:17 AM 2018/8/22
 * @Modified By:
 */


@RestController
@RequestMapping(value = "/uploadAndDownload")

public class UploadDownloadController extends BaseController{


    private static final Logger logger = LoggerFactory.getLogger(UploadDownloadController.class);

    @Value("${web.upload-path}")
    private String uploadDir;

    private final String[] IMAGE_TYPE = {"jpg", "jpeg", "gif", "png"};

    private final Integer STATE_CODE_301 = 301; // 图片上传失败 - 上传文件为空
    private final Integer STATE_CODE_302 = 302; // 图片上传失败 - 上传文件格式不对


    /**
     * 上传图片接口
     * @param multipartFile  , consumes = "multipart/form-data"
     * @return
     */
    @RequestMapping(value = "/uploadImage", method = RequestMethod.POST)
    public ResponseEntity uploadImage(@RequestParam(value = "file") MultipartFile multipartFile) {


        ResponseEntity responseEntity = new ResponseEntity(StateCode.OK.getValue(), "请求成功");
        if (multipartFile == null || StringUtils.isBlank(multipartFile.getOriginalFilename())) {
            return error(STATE_CODE_301, "文件不能为空");
        }
        try {
            // 取得上传文件名
            String originalFilename = multipartFile.getOriginalFilename();
            String formatName = this.getLowerCaseFileType(originalFilename);
            if (StringUtils.isEmpty(formatName) || !ArrayUtils.contains(IMAGE_TYPE, formatName)) {
                formatName = this.getImageFormatName(multipartFile.getInputStream());
            }
            // 检查上传文件类型
            if (!ArrayUtils.contains(IMAGE_TYPE, formatName)) {
                return error(STATE_CODE_302, "上传文件格式错误！只能上传jpg,jpeg,gif, png格式图片");
            }
            // 生成上传图片名称
//            String fileName = UUIDUtil.getUUID32() + "." + formatName.toLowerCase();
//            String filePath = DateTimeUtil.format(new Date(), "yyyyMM") + "/" + fileName;
//            Images images = new Images();
//            images.setImageName(fileName);
//            images.setImageUrl(filePath);


            // 上传图片动作
            String okFileName = null;
            okFileName = SaveFileUtil.saveImg(multipartFile, uploadDir);
            Map<String, Object> data = new HashMap<String, Object>();
            data.put("value", okFileName);
            responseEntity = okay(data, "上传成功");

        } catch (ExecutorException e) {
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        } catch (IOException e) {
//            e.printStackTrace();
            responseEntity = super.error(StateCode.ERROR.getValue(), "请求异常");
        }
        super.setResponseEntity(responseEntity);
        return responseEntity;
    }



    /**
     * 得到小写的文件类型,不带.号
     */
    public static String getLowerCaseFileType(String filename) {
        if (CommonUtil.isNotNull(filename)) {
            int position = filename.lastIndexOf(".");
            if (position <= 0) {
                return null;
            } else {
                return filename.substring(position + 1).toLowerCase();
            }
        } else {
            return null;
        }
    }

    /**
     * 获取图片后缀格式
     *
     * @param inputStream
     * @return
     * @throws java.io.IOException
     */
    private String getImageFormatName(InputStream inputStream) throws IOException {
        ImageInputStream imageInputStream = null;
        try {
            imageInputStream = ImageIO.createImageInputStream(inputStream);
            Iterator<ImageReader> imageReaderIterator = ImageIO.getImageReaders(imageInputStream);
            if (!imageReaderIterator.hasNext()) {
                throw new IOException("没有获取到ImageReader 对象");
            }
            return imageReaderIterator.next().getFormatName();
        } finally {
            if (imageInputStream != null) {
                try {
                    imageInputStream.close();
                } catch (IOException e) {
                    logger.error("关闭图片输入流异常：{}", e.getMessage());
                }
            }
        }
    }
}
