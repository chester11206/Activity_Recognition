package com.example.jh.activity;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.support.annotation.NonNull;
import android.text.method.ScrollingMovementMethod;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.IgnoreExtraProperties;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseModelOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.google.firebase.ml.custom.model.FirebaseCloudModelSource;
import com.google.firebase.ml.custom.model.FirebaseModelDownloadConditions;

import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static java.text.DateFormat.getDateTimeInstance;

public class MultiSensors {
    public Activity context;
    public static final String TAG = "MultiSensorsApi";
    private DatabaseReference mDatabase;
    LinearLayout ll;

    TextView txvResult;
    TextView predicttxv;

    private ActivityInference activityInference;
    private SensorManager mSensorManager;

    private String [] activityItems = null;
    private String real_activity = null;
    private String last_activity = null;

    private boolean startPredict = false;
    private boolean startUpload = false;
    private boolean startListen_acce = false;
    private boolean startListen_gyro = false;
    private boolean startUpload_acce = false;
    private boolean startUpload_gyro = false;
    private boolean startPredict_acce = false;
    private boolean startPredict_gyro = false;

    private int uploadWait = 2000;
    private int input_width = ActivityInference.input_width;
    private int channels = ActivityInference.channels;

    private List<acceData> acceDataSet = new ArrayList<acceData>();
    private List<gyroData> gyroDataSet = new ArrayList<gyroData>();
    private int acceNum = 0;
    private int gyroNum = 0;
    private int dataNum = 0;
    private int startNum = 0;
    private int stopNum = 0;

    public static final Map<Integer, TextView> textview_map = new LinkedHashMap<Integer, TextView>();
    public static final Map<String, Integer> sensorstype_map = createSensorsTypeMap();
    private static Map<String, Integer> createSensorsTypeMap()
    {
        Map<String, Integer> myMap = new LinkedHashMap<String, Integer>();
        myMap.put("Accelerometer", Sensor.TYPE_ACCELEROMETER);
        myMap.put("Gyroscope", Sensor.TYPE_GYROSCOPE);
        return myMap;
    }

    @IgnoreExtraProperties
    public class acceData {
        private float accelerometerX = 0;
        private float accelerometerY = 0;
        private float accelerometerZ = 0;
        private long TimeNow;
        private String real_activity;

        public acceData(float aX, float aY, float aZ, String ra) {
            accelerometerX = aX;
            accelerometerY = aY;
            accelerometerZ = aZ;
            this.real_activity = ra;
            Calendar cal = Calendar.getInstance();
            Date now = new Date();
            cal.setTime(now);
            TimeNow = cal.getTimeInMillis();
        }
        public float getAccelerometerX() {return accelerometerX;}
        public float getAccelerometerY() {return accelerometerY;}
        public float getAccelerometerZ() {return accelerometerZ;}
        public long getTimeNow() {return TimeNow;}
        public String getReal_activity() {return this.real_activity;}
    }

    @IgnoreExtraProperties
    public class gyroData {
        private float gyroscopeX = 0;
        private float gyroscopeY = 0;
        private float gyroscopeZ = 0;

        public gyroData (float gX, float gY, float gZ) {
            gyroscopeX = gX;
            gyroscopeY = gY;
            gyroscopeZ = gZ;
        }
        public float getGyroscopeX() {return gyroscopeX;}
        public float getGyroscopeY() {return gyroscopeY;}
        public float getGyroscopeZ() {return gyroscopeZ;}
    }

    public void start(Activity activity, SensorManager SensorManager, List<String> sensors_list) {
        context = activity;
        ll = (LinearLayout) context.findViewById(R.id.sensors_display);
        activityItems = context.getResources().getStringArray(R.array.activity);
        mDatabase = FirebaseDatabase.getInstance().getReference();

        RadioGroup rg = (RadioGroup)context.findViewById(R.id.radioGroup);
        rg.setOnCheckedChangeListener(rglistener);

        txvResult = (TextView) this.context.findViewById(R.id.multisensorstxView);
        txvResult.setMovementMethod(new ScrollingMovementMethod());
        predicttxv = (TextView) this.context.findViewById(R.id.predict_txView);
        predicttxv.setMovementMethod(new ScrollingMovementMethod());

        this.mSensorManager = SensorManager;
        for (String sensor : sensors_list) {

            TextView txv = new TextView(context);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);

            txv.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 16);
            txv.setMovementMethod(new ScrollingMovementMethod());
            txv.setLayoutParams(params);
            ll.addView(txv);
            textview_map.put(sensorstype_map.get(sensor), txv);

            mSensorManager.registerListener(mSensorEventListener,
                    mSensorManager.getDefaultSensor(sensorstype_map.get(sensor)),
                    SensorManager.SENSOR_DELAY_FASTEST);
        }

        activityInference = new ActivityInference(context);

        Button uploadbtn = (Button) context.findViewById(R.id.uploadbtn);
        uploadbtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                if (real_activity == null) {
                    txvResult.setText("You haven't set the activity!");
                }
                else {
                    txvResult.setText("Uploading...");
                    startUpload = true;
                }
            }
        });

        Button predictbtn = (Button) context.findViewById(R.id.predictbtn);
        predictbtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                predicttxv.setText("Predicting...");
                startPredict = true;
            }
        });

        Button stopbtn = (Button) context.findViewById(R.id.stopbtn);
        stopbtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                if (mSensorManager != null) {
                    mSensorManager.unregisterListener(mSensorEventListener);
                }
                txvResult.append("\nStop!");
            }
        });

        sensorStart();
    }

    private void sensorStart() {
        startListen_acce = true;
        startListen_gyro = true;
    }

    private RadioGroup.OnCheckedChangeListener rglistener = new RadioGroup.OnCheckedChangeListener(){

        @Override
        public void onCheckedChanged(RadioGroup rg,
                                     int checkedId) {
            RadioButton rb = (RadioButton) context.findViewById(checkedId);
            real_activity = rb.getText().toString();
        }

    };


    private SensorEventListener mSensorEventListener = new SensorEventListener() {

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {}

        @Override
        public void onSensorChanged(SensorEvent event) {
            TextView txv = textview_map.get(event.sensor.getType());
            switch (event.sensor.getType()) {
                case Sensor.TYPE_ACCELEROMETER:
                    if (startListen_acce) {
                        txv.setText("\nAccelerometer"
                                + "\nX: " + event.values[0]
                                + "\nY: " + event.values[1]
                                + "\nZ: " + event.values[2]);

                        String ra = real_activity;
                        acceData acceData = new acceData(event.values[0], event.values[1], event.values[2], ra);
                        acceDataSet.add(acceData);
                        acceNum++;

                        if (startUpload && acceNum % uploadWait == 0) {
                            startListen_acce = false;
                            startUpload_acce = true;
                        }
                        if (startPredict && acceNum % input_width == 1 && acceNum > input_width) {
                            startListen_acce = false;
                            startPredict_acce = true;
                        }
                    }
                    if (startUpload_acce && startUpload_gyro) {
                        dataNum++;
                        String ra = real_activity;
                        startNum = acceNum - uploadWait;
                        stopNum = acceNum;
                        txvResult.setText("Num: " + startNum + " to " + stopNum + " " + "Upload Num: " + dataNum);

//                        Map<String, Float> SensorData = new LinkedHashMap<String, Float>();
//                        float [] predictData = new float[input_width * channels];
//                        int predictNum = 0;
                        for (int i = startNum; i < stopNum; i++) {
                            Map<String, Float> SensorData = new LinkedHashMap<String, Float>();
                            SensorData.put("accelerometerX", acceDataSet.get(i).getAccelerometerX());
                            SensorData.put("accelerometerY", acceDataSet.get(i).getAccelerometerY());
                            SensorData.put("accelerometerZ", acceDataSet.get(i).getAccelerometerZ());
                            SensorData.put("gyroscopeX", gyroDataSet.get(i).getGyroscopeX());
                            SensorData.put("gyroscopeY", gyroDataSet.get(i).getGyroscopeY());
                            SensorData.put("gyroscopeZ", gyroDataSet.get(i).getGyroscopeZ());
//                            for (String key : SensorData.keySet()) {
//                                predictData[predictNum] = SensorData.get(key);
//                                predictNum += 1;
//                            }
                            SensorData.put("timeNow", (float)acceDataSet.get(i).getTimeNow());

                            for (String activity : activityItems) {
                                if (activity.equals(acceDataSet.get(i).getReal_activity())) {
                                    SensorData.put(activity, (float) 1);
                                } else {
                                    SensorData.put(activity, (float) 0);
                                }
                            }
                            mDatabase.child("SensorDataSet").push().setValue(SensorData);
                        }
//                        if (startPredict) {
//                            activityPrediction(predictData);
//                        }
                        startUpload_acce = false;
                        startUpload_gyro = false;
                        sensorStart();
                    }
                    if (startPredict_acce && startPredict_gyro) {
                        float[] predictData = new float[input_width * channels];
                        int predictTime = acceNum / input_width - 1;
                        for (int i = 0; i < input_width; i++) {
                            predictData[i * channels + 0] = acceDataSet.get(predictTime * input_width + i).getAccelerometerX();
                            predictData[i * channels + 1] = acceDataSet.get(predictTime * input_width + i).getAccelerometerY();
                            predictData[i * channels + 2] = acceDataSet.get(predictTime * input_width + i).getAccelerometerZ();
                            predictData[i * channels + 3] = gyroDataSet.get(predictTime * input_width + i).getGyroscopeX();
                            predictData[i * channels + 3] = gyroDataSet.get(predictTime * input_width + i).getGyroscopeY();
                            predictData[i * channels + 3] = gyroDataSet.get(predictTime * input_width + i).getGyroscopeZ();
                        }
                        activityPrediction(predictData);
                    }
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    if (startListen_gyro) {
                        txv.setText("Gyroscope"
                                + "\nX: " + event.values[0]
                                + "\nY: " + event.values[1]
                                + "\nZ: " + event.values[2]);

                        gyroData gyroData = new gyroData(event.values[0], event.values[1], event.values[2]);
                        gyroDataSet.add(gyroData);
                        gyroNum++;

                        if (startUpload && gyroNum % uploadWait == 0) {
                            startListen_gyro = false;
                            startUpload_gyro = true;
                        }
                        if (startPredict && gyroNum % input_width == 1 && gyroNum > input_width) {
                            startListen_gyro = false;
                            startPredict_gyro = true;
                        }
                    }
                    break;
                default:
                    break;

            }
        }
    };

    private void activityPrediction(float[] predictData)
    {
        float[] results = activityInference.getActivityProb(predictData);
        int maxIndex = 0;
        float max = -1;
        for (int i = 0; i < results.length; i++) {
            if (results[i] > max) {
                max = results[i];
                maxIndex = i;
            }
        }
//        predicttxv.setText("\n" + activityItems[0] + ": " + String.format("%.8f", results[0])
//                + "\n" + activityItems[1] + ": " + String.format("%.8f", results[1])
//                + "\n" + activityItems[2] + ": " + String.format("%.8f", results[2])
//                + "\n" + activityItems[3] + ": " + String.format("%.8f", results[3])
//                + "\n" + activityItems[4] + ": " + String.format("%.8f", results[4])
//                + "\n" + activityItems[5] + ": " + String.format("%.8f", results[5])
//                + "\n\nResult: " + activityItems[maxIndex] + " " + String.format("%.8f", results[maxIndex]));
        if (!activityItems[maxIndex].equals(last_activity) || last_activity.equals(null)) {
            DateFormat dateFormat = getDateTimeInstance();
            predicttxv.append("\nTime: " + dateFormat.format(MainActivity.timeNow) + " Activity: " + activityItems[maxIndex]);
        }
        last_activity = activityItems[maxIndex];

        startPredict_acce = false;
        startPredict_gyro = false;
        sensorStart();
    }
}
