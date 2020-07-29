在Windows下build ncnn
参考：
[https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android](https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-android)
[https://www.jb51.net/article/183140.htm](https://www.jb51.net/article/183140.htm)

第一个链接为官网教程（可以参考到如何给vs进行build、如何给Android进行build）。详细步骤可以参考第二个链接。步骤如下：
- 下载protobuf，进入VS的本地命令行，按照命令可以完成protobuf的build。
- 但是会发现，cmake命令不在本地，先下载一个cmake，安装后，把bin目录添加到系统PATH。
-  下载ncnn项目，同protobuf类似进行build，就可以获得相应的工具。
- 安装opencv，添加到系统Path。

-------------------------

转换流程，参考：
[https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx](https://github.com/Tencent/ncnn/wiki/use-ncnn-with-pytorch-or-onnx)

- python部分，可以到onnx：
```
import torch
torch.onnx._export(model, x, "model_name.onnx", export_params=True)
```
其中，model是pytorch模型，x是输入的样本，可以为torch.rand(xxx)，维度自己设定为模型的输入维度即可。第三个参数是保存的文件名。这样就可以保存onnx文件了。

但是，官方说可能onnx会有冗余，那么使用onnx-simplifier库就可以将onnx进行简单化。
```
python3 -m onnxsim model_name.onnx model_name-sim.onnx
```
-m是指载入某个模块

也可以通过程序进行转换，但是尝试的时候，会在中间异常退出，命令行则不会，待解决。需要安装onnx库，才能进行加载。
```
import onnx
from onnxsim import simplify
model_onnx = onnx.load("model_name.onnx")

# simplify failed??
model_onnx_simp, check = simplify(model_onnx)
onnx.save_model(model_onnx_simp,"model_onnx-sim.onnx")
```

最后，由我们build好的程序进行onnx转ncnn的最后一步：
```
onnx2ncnn model_onnx-sim.onnx model_onnx.param model_onnx.bin
```
.param和.bin分别是参数和模型文件。

- c++部分，vs2015调用（依赖很多，根据参考配置的vs2015运行起来还是可以的。）
参考：[https://blog.csdn.net/zhaotun123/article/details/99671286](https://blog.csdn.net/zhaotun123/article/details/99671286)
1、 配置好include、lib路径等，（注意下载的程序为x64，默认的release为32位，会无法运行程序，）
2、 使用ncnn官方的代码，参考：[https://github.com/Tencent/ncnn/wiki](https://github.com/Tencent/ncnn/wiki)
使用opencv读Mat，然后加载成ncnn的Mat，然后处理后，经过模型的extract就可以获得最后的输出。
```
int main()
{
    cv::Mat img = cv::imread("image.ppm", CV_LOAD_IMAGE_GRAYSCALE);
    int w = img.cols;
    int h = img.rows;

    // subtract 128, norm to -1 ~ 1
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_GRAY, w, h, 60, 60);
    float mean[1] = { 128.f };
    float norm[1] = { 1/128.f };
    in.substract_mean_normalize(mean, norm);

    ncnn::Net net;
    net.load_param("model.param");
    net.load_model("model.bin");

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat feat;
    ex.extract("output", feat);

    return 0;
}
```
代码需要更改为自己的模型和图片，主要部分是通过load_param和load_model可以分别加载参数和模型。而通过input和extract则分别完成输入和输出。

- Android部分，配置新建jni目录。
将从官网已经build好的ncnn-android-lib.zip，将文件夹解压到jni目录下。新建Android.mk、Application.mk和对应的.h和.cpp文件。

配置Android.mk
```
LOCAL_PATH := $(call my-dir)

#把这个路径改成你自己刚才编译的install路径，用全路径！
NCNN_INSTALL_PATH := D:/KyoDante/androidApp/ncnnDemo/app/src/main/jni/ncnn-android-lib

include $(CLEAR_VARS)
LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := $(NCNN_INSTALL_PATH)/armeabi-v7a/libncnn.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := demo
#这个是你的cpp文件
LOCAL_SRC_FILES := demo.cpp

LOCAL_C_INCLUDES := $(NCNN_INSTALL_PATH)/include/ncnn

LOCAL_STATIC_LIBRARIES := ncnn

LOCAL_CFLAGS := -O2 -fvisibility=hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_CPPFLAGS := -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fomit-frame-pointer -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math
LOCAL_LDFLAGS += -Wl,--gc-sections

LOCAL_CFLAGS += -fopenmp
LOCAL_CPPFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp

LOCAL_LDLIBS := -lz -llog -ljnigraphics

include $(BUILD_SHARED_LIBRARY)
```

配置Application.mk
```
# APP_STL := stlport_static
# APP_STL := gnustl_static
APP_STL := c++_static

# APP_ABI := armeabi armeabi-v7a

APP_ABI := armeabi-v7a
APP_PLATFORM := android-9
#NDK_TOOLCHAIN_VERSION := 4.9
```

配置build.gradle,moduleName是和前面Android.mk里面的一项保持一致的。

![build.gradle1.png](https://upload-images.jianshu.io/upload_images/10512503-ab5af45f28bbf412.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

而这个srcDirs则是设定为ndk-build默认产生的文件夹路径。

![build.gradle2.png](https://upload-images.jianshu.io/upload_images/10512503-993f7763d6959d84.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


需要一个调用JNI的Java类，大致如下：
```
public class JNICaller {
    public JNICaller(){
        System.loadLibrary("demo");
    }

    public native boolean InitNcnn(String modelPath); //通过路径名加载网络
}
```
System.loadLibrary会找到libdemo.so的文件的。而native方法是需要对应到demo.cpp里面去的。

配置com_example_ncnndemo_JNICaller.h，进行方法的声明。

```
#include <jni.h>

#ifndef NCNNDEMO_COM_EXAMPLE_NCNNDEMO_JNICALLER_H
#define NCNNDEMO_COM_EXAMPLE_NCNNDEMO_JNICALLER_H

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     Java_com_example_ncnndemo_JNICaller
 * Method:    InitNcnn
 * Signature: None
 */
 JNIEXPORT jboolean JNICALL Java_com_example_ncnndemo_JNICaller_InitNcnn
             (JNIEnv *env, jobject instance,jstring DetectionModelPath_);

#ifdef __cplusplus
}
#endif
#endif //NCNNDEMO_COM_EXAMPLE_NCNNDEMO_JNICALLER_H
```

配置demo.cpp，进行方法的实现。

```
#include <jni.h>
#include <string>
#include <net.h>
#include "com_example_ncnndemo_JNICaller.h"

// 注意把这里的函数名改成你自己对应的，一定不能错！
JNIEXPORT jboolean
JNICALL Java_com_example_ncnndemo_JNICaller_InitNcnn
            (JNIEnv *env, jobject instance,jstring DetectionModelPath_) {

    const char *DetectionModelPath = env->GetStringUTFChars(DetectionModelPath_, 0);
    if (NULL == DetectionModelPath) {
        return false;
    }

    std::string tModelDir = DetectionModelPath;
    std::string tLastChar = tModelDir.substr(tModelDir.length() - 1, 1);


    if ("\\" == tLastChar) {
        tModelDir = tModelDir.substr(0, tModelDir.length() - 1) + "/";
    } else if (tLastChar != "/") {
        tModelDir += "/";
    }


    std::vector<std::string> param_files;
    param_files.resize(1);
    param_files[0] = tModelDir + "/demo.param";

    std::vector<std::string> bin_files;
    bin_files.resize(1);
    bin_files[0] = tModelDir + "/demo.bin";

    ncnn::Net net;
    net.load_param(param_files[0].data());
    net.load_model(bin_files[0].data());


    env->ReleaseStringUTFChars(DetectionModelPath_, DetectionModelPath);

    return true;
}
```

至此，准备工作接近尾声，jni就差最后的ndk-build了。

在命令行进入jni目录后，运行“ndk-build”命令（记得把ndk-build所在的目录添加到系统Path），可以自动产生libs、obj目录，可以在libs下找到.so库，也就是我们配置好的库了。
![ndk-build.png](https://upload-images.jianshu.io/upload_images/10512503-c3d125ebb88b9d79.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最终的目录结构大概为：

![目录：app/src/.png](https://upload-images.jianshu.io/upload_images/10512503-978adbe41539ee29.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

就可以使用JNICaller这个类调用其对应的native方法了。