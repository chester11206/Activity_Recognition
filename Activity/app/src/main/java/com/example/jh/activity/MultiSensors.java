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

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class MultiSensors {
    public Activity context;
    public static final String TAG = "MultiSensorsApi";
    private DatabaseReference mDatabase;
    LinearLayout ll;

    TextView txvResult;
    TextView predicttxv;

    private SensorManager mSensorManager;

    private String [] activityItems = null;
    private String real_activity = "Still";

    private boolean startPredict = false;
    private boolean startListen_acce = false;
    private boolean startListen_gyro = false;

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
        private String real_activit;

        public acceData(float aX, float aY, float aZ, String ra) {
            accelerometerX = aX;
            accelerometerY = aY;
            accelerometerZ = aZ;
            real_activity = ra;
            Calendar cal = Calendar.getInstance();
            Date now = new Date();
            cal.setTime(now);
            TimeNow = cal.getTimeInMillis();
        }
        public float getAccelerometerX() {return accelerometerX;}
        public float getAccelerometerY() {return accelerometerY;}
        public float getAccelerometerZ() {return accelerometerZ;}
        public long getTimeNow() {return TimeNow;}
        public String getReal_activity() {return real_activity;}
    }

    @IgnoreExtraProperties
    public class gyroData {
        private float gyroscopeX = 0;
        private float gyroscopeY = 0;
        private float gyroscopeZ = 0;

        public gyroData() {}
        public void setGyroscope(float gX, float gY, float gZ) {
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

        txvResult = (TextView) this.context.findViewById(R.id.multisensorstxView);
        txvResult.setMovementMethod(new ScrollingMovementMethod());
        predicttxv = (TextView) this.context.findViewById(R.id.predictView);
        predicttxv.setMovementMethod(new ScrollingMovementMethod());

        RadioGroup rg = (RadioGroup)context.findViewById(R.id.radioGroup);
        rg.setOnCheckedChangeListener(rglistener);

        this.mSensorManager = SensorManager;
        for (String sensor : sensors_list) {

            TextView txv = new TextView(context);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);

            txv.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 17);
            txv.setMovementMethod(new ScrollingMovementMethod());
            txv.setLayoutParams(params);
            ll.addView(txv);
            textview_map.put(sensorstype_map.get(sensor), txv);

            mSensorManager.registerListener(mSensorEventListener,
                    mSensorManager.getDefaultSensor(sensorstype_map.get(sensor)),
                    SensorManager.SENSOR_DELAY_FASTEST);
        }

        Button stopbtn = (Button) context.findViewById(R.id.stopbtn);
        stopbtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                if (mSensorManager != null) {
                    mSensorManager.unregisterListener(mSensorEventListener);
                }
            }
        });

        Button predictbtn = (Button) context.findViewById(R.id.predictbtn);
        predictbtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                startPredict = true;
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
            txvResult.append("\n" + real_activity);
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

                        if (acceNum % 450 == 0) {
                            startListen_acce = false;
                        }
                    }
                    if (!startListen_acce && !startListen_gyro) {
                        dataNum++;
                        String ra = real_activity;
                        startNum = stopNum;
                        stopNum = acceNum;
                        txvResult.setText("\nNum: " + startNum + " to " + stopNum + " " + "Activity: " + ra + "\nDataNum: " + dataNum);

//                        Map<String, Float> SensorData = new LinkedHashMap<String, Float>();
                        for (int i = startNum; i < stopNum; i++) {
                            Map<String, Float> SensorData = new LinkedHashMap<String, Float>();
                            SensorData.put("accelerometerX", acceDataSet.get(i).getAccelerometerX());
                            SensorData.put("accelerometerY", acceDataSet.get(i).getAccelerometerY());
                            SensorData.put("accelerometerZ", acceDataSet.get(i).getAccelerometerZ());
                            SensorData.put("gyroscopeX", gyroDataSet.get(i).getGyroscopeX());
                            SensorData.put("gyroscopeY", gyroDataSet.get(i).getGyroscopeY());
                            SensorData.put("gyroscopeZ", gyroDataSet.get(i).getGyroscopeZ());
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
//                            WriterIdentify writerIdentify = WriterIdentify.newInstance(context);
//                            writerIdentify.run(SensorData);
//                            predicttxv.append("\nResult: " + writerIdentify.getResult());
//                        }
                        sensorStart();
                    }
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    if (startListen_gyro) {
                        txv.setText("\nGyroscope"
                                + "\nX: " + event.values[0]
                                + "\nY: " + event.values[1]
                                + "\nZ: " + event.values[2]);
                        gyroData gyroData = new gyroData();
                        gyroData.setGyroscope(event.values[0], event.values[1], event.values[2]);
                        gyroDataSet.add(gyroData);
                        gyroNum++;

                        if (gyroNum % 450 == 0) {
                            startListen_gyro = false;
                        }
                    }
                    break;
                default:
                    break;

            }
        }
    };

    private void tflitePredict(float[][][] inputData) throws FirebaseMLException {
        FirebaseModelDownloadConditions.Builder conditionsBuilder = new FirebaseModelDownloadConditions.Builder().requireWifi();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            // Enable advanced conditions on Android Nougat and newer.
            conditionsBuilder = conditionsBuilder
                    .requireCharging()
                    .requireDeviceIdle();
        }
        FirebaseModelDownloadConditions conditions = conditionsBuilder.build();

// Build a FirebaseCloudModelSource object by specifying the name you assigned the model
// when you uploaded it in the Firebase console.
        FirebaseCloudModelSource cloudSource = new FirebaseCloudModelSource.Builder("activity-rnn")
                .enableModelUpdates(true)
                .setInitialDownloadConditions(conditions)
                .setUpdatesDownloadConditions(conditions)
                .build();
        FirebaseModelManager.getInstance().registerCloudModelSource(cloudSource);

        FirebaseModelOptions options = new FirebaseModelOptions.Builder()
                .setCloudModelName("activity-rnn")
                .build();
        FirebaseModelInterpreter firebaseInterpreter =
                FirebaseModelInterpreter.getInstance(options);

        FirebaseModelInputOutputOptions inputOutputOptions =
                new FirebaseModelInputOutputOptions.Builder()
                        .setInputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 450, 7})
                        .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 6})
                        .build();

        float[][][] input = new float[1][450][7];
        input = inputData;
        FirebaseModelInputs inputs = new FirebaseModelInputs.Builder()
                .add(input)  // add() as many input arrays as your model requires
                .build();
        Task<FirebaseModelOutputs> result =
                firebaseInterpreter.run(inputs, inputOutputOptions)
                        .addOnSuccessListener(
                                new OnSuccessListener<FirebaseModelOutputs>() {
                                    @Override
                                    public void onSuccess(FirebaseModelOutputs result) {
                                        // ...
                                        float[][] output = result.<float[][]>getOutput(0);
                                        float[] probabilities = output[0];
                                    }
                                })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        // Task failed with an exception
                                        // ...
                                    }
                                });
    }
}
