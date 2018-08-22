package com.example.jh.activity;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;

public class WriterIdentify {

    // classifier
    private Interpreter tflite;

    // input
    private float[][] labelProbArray = null;
    private String [] activityItems = null;

    public static WriterIdentify newInstance(Context context) {
        WriterIdentify writerIdentify = new WriterIdentify(context);
        return writerIdentify;
    }

    private WriterIdentify(Context context) {
        activityItems = context.getResources().getStringArray(R.array.activity);
        try {
            tflite = new Interpreter(loadModelFile(context));
        } catch (Exception e) {

        }
        labelProbArray = new float[1][6];

    }

    public void run(Map<String, Float> SensorData) {
        tflite.run(convertBitmapToByteBuffer(SensorData), labelProbArray);
        //convertBitmapToByteBuffer(bitmap,width,height);
    }

    // return prediction
    public String getResult() {
        //int[] resultDict = new int[]{0, 1, 2, 3, 4, 5};
        for (int i = 0; i < labelProbArray[0].length; i++) {
            if (labelProbArray[0][i] == 1.0f) {
                return activityItems[i];
            }
        }
        return "Unknown";
    }

    private ByteBuffer convertBitmapToByteBuffer(Map<String, Float> SensorData) {
        int length = SensorData.size();
        ByteBuffer tempData = ByteBuffer.allocateDirect(length * 4);

        List<Float> SensorDatas = null;
        for (String key : SensorData.keySet()) {
            SensorDatas.add(SensorData.get(key));
        }

        tempData.order(ByteOrder.nativeOrder());
        //int[] pixels = getPicturePixel(bitmap);
        for (int i = 0; i < SensorDatas.size(); i++) {
            byte[] bytes = float2byte(SensorDatas.get(i));
            for (int k = 0; k < bytes.length; k++) {
                tempData.put(bytes[k]);
            }
        }
        return tempData;
    }
    // 读取图片像素
    private int[] getPicturePixel(Bitmap bitmap) {

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        // 保存所有的像素的数组，图片宽×高
        int[] pixels = new int[width * height];

        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        String str = "";
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] = pixels[i] & 0x000000ff;
        }
        return pixels;
    }
    // convert float to byte
    private byte[] float2byte(float f) {

        // float to byte
        int fbit = Float.floatToIntBits(f);

        byte[] b = new byte[4];
        for (int i = 0; i < 4; i++) {
            b[i] = (byte) (fbit >> (24 - i * 8));
        }

        // 翻转数组
        int len = b.length;
        // 建立一个与源数组元素类型相同的数组
        byte[] dest = new byte[len];
        // 为了防止修改源数组，将源数组拷贝一份副本
        System.arraycopy(b, 0, dest, 0, len);
        byte temp;
        // 将顺位第i个与倒数第i个交换
        for (int i = 0; i < len / 2; ++i) {
            temp = dest[i];
            dest[i] = dest[len - i - 1];
            dest[len - i - 1] = temp;
        }
        return dest;
    }

    // load model from assets
    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // get model name
    private String getModelPath() {
        return "ActivityRNN0.tflite";
    }
}

