package com.example.jh.activity;

import android.content.Context;
import android.content.res.AssetManager;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class ActivityInference {
    static {
        System.loadLibrary("tensorflow_inference");
    }

    public static int input_height = 1;
    public static int input_width = 400;
    public static int channels = 6;
    public static int num_label = 6;

    private static ActivityInference activityInferenceInstance;
    private TensorFlowInferenceInterface inferenceInterface;
    //ActivityCNNopt1
    private static final String MODEL_FILE = "file:///android_asset/ActivityCNNopt.pb";
    private static final String INPUT_NODE = "input_x";
    private static final String[] OUTPUT_NODES = {"prediction"};
    private static final String OUTPUT_NODE = "prediction";
    private static final long[] INPUT_SIZE = {1,input_height,input_width,channels};
    private static final int OUTPUT_SIZE = num_label;
    private static AssetManager assetManager;

    public static ActivityInference getInstance(final Context context)
    {
        if (activityInferenceInstance == null)
        {
            activityInferenceInstance = new ActivityInference(context);
        }
        return activityInferenceInstance;
    }

    public ActivityInference(final Context context) {
        this.assetManager = context.getAssets();
        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
    }

    public float[] getActivityProb(float[] input_signal)
    {
        float[] result = new float[OUTPUT_SIZE];
        inferenceInterface.feed(INPUT_NODE,input_signal,INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE,result);
        //"Biking", "In Vehicle", "Running", "Still", "Tilting", "Walking"
        return result;
    }
}
