package com.weijie.testopencv;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.opencv.core.*;
@SpringBootApplication
public class TestopencvApplication {


//	static {
//		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//	}

	public static void main(String[] args) {
		SpringApplication.run(TestopencvApplication.class, args);

//		System.out.println("Welcome to OpenCV " + Core.VERSION);
	}
}
