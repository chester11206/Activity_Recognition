package com.example.jh.activity;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Environment;
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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
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

    String csvTitle = "Activity,AcceX,AcceY,AcceZ,GyroX,GyroY,GyroZ,Time";

    private ActivityInference activityInference;
    private SensorManager mSensorManager;

    private String [] activityItems = null;
    private String real_activity = null;
    private String last_activity = null;

    private boolean startPredict = false;
    private boolean startUpload = false;

    private boolean startListen_acce = false;
    private boolean startListen_gyro = false;
    private boolean startListen_grav = false;
    private boolean startListen_magn = false;

    private boolean startUpload_acce = false;
    private boolean startUpload_gyro = false;
    private boolean startPredict_acce = false;
    private boolean startPredict_gyro = false;

    private int uploadWait = 2000;
    private int input_width = ActivityInference.input_width;
    private int channels = ActivityInference.channels;

    private List<acceData> acceDataSet = new ArrayList<acceData>();
    private List<gyroData> gyroDataSet = new ArrayList<gyroData>();
    private float [] gravData = new float[3];
    private float [] magnData = new float[3];
    private int acceNum = 0;
    private int gyroNum = 0;
    private int dataNum = 0;
    private int startNum = 0;
    private int stopNum = 0;
    private int startUploadNum = 0;
    private int startPredictNum = 0;

    private int allPredictNum = 0;
    private int rightPredictNum = 0;
    private float test_accuracy = 0;

    public static final Map<Integer, TextView> textview_map = new LinkedHashMap<Integer, TextView>();
    public static final Map<String, Integer> sensorstype_map = createSensorsTypeMap();
    private static Map<String, Integer> createSensorsTypeMap()
    {
        Map<String, Integer> myMap = new LinkedHashMap<String, Integer>();
        myMap.put("Accelerometer", Sensor.TYPE_ACCELEROMETER);
        myMap.put("Gyroscope", Sensor.TYPE_GYROSCOPE);
        myMap.put("Gravity", Sensor.TYPE_GRAVITY);
        myMap.put("Magnetic", Sensor.TYPE_MAGNETIC_FIELD);
        return myMap;
    }
    public static final Map<String, Integer> activity_map = createActivityMap();
    private static Map<String, Integer> createActivityMap()
    {
        Map<String, Integer> myMap = new LinkedHashMap<String, Integer>();
        myMap.put("Biking", 0);
        myMap.put("In Vehicle", 1);
        myMap.put("Running", 2);
        myMap.put("Still", 3);
        myMap.put("Tilting", 4);
        myMap.put("Walking", 5);
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

            txv.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 15);
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
                    txvResult.setText("Saving...");
                    startUpload = true;
                    startUploadNum = acceNum;
                }
            }
        });

        Button predictbtn = (Button) context.findViewById(R.id.predictbtn);
        predictbtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                if (real_activity == null) {
                    predicttxv.setText("You haven't set the activity!");
                }
                else {
                    predicttxv.setText("Predicting...");
                    startPredict = true;
                    startPredictNum = acceNum;
                }
            }
        });
        Button stopUploadbtn = (Button) context.findViewById(R.id.stopUploadbtn);
        stopUploadbtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                startUpload = false;
                txvResult.append("\nStop Upload!");
            }
        });
        Button stopPredictbtn = (Button) context.findViewById(R.id.stopPredictbtn);
        stopPredictbtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                startPredict = false;
                last_activity = null;
                predicttxv.append("\nStop Predict");
            }
        });

//        Button stopbtn = (Button) context.findViewById(R.id.stopbtn);
//        stopbtn.setOnClickListener(new View.OnClickListener() {
//            public void onClick(View view) {
//                if (mSensorManager != null) {
//                    mSensorManager.unregisterListener(mSensorEventListener);
//                }
//                txvResult.append("\nStop!");
//            }
//        });

        sensorStart();
    }

    private void sensorStart() {
        startListen_acce = true;
        startListen_gyro = true;
        startListen_grav = true;
        startListen_magn = true;
    }

    private void uploadStop() {
        startUpload_acce = false;
        startUpload_gyro = false;
    }

    private void predictStop() {
        startPredict_acce = false;
        startPredict_gyro = false;
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
                        float [] earthAcce = new float[4];
                        earthAcce = phone2earth(event);

//                        txv.setText("\nAccelerometer"
//                                + "\nX: " + event.values[0]
//                                + "\nY: " + event.values[1]
//                                + "\nZ: " + event.values[2]);
                        txv.setText("\nAccelerometer(Earth)"
                                + "\nX: " + earthAcce[0]
                                + "\nY: " + earthAcce[1]
                                + "\nZ: " + earthAcce[2]);

                        String ra = real_activity;
//                        acceData acceData = new acceData(event.values[0] - gravData[0]
//                                , event.values[1] - gravData[1]
//                                , event.values[2] - gravData[2]
//                                , ra);
                        acceData acceData = new acceData(earthAcce[0], earthAcce[1], earthAcce[2], ra);
                        acceDataSet.add(acceData);
                        acceNum++;

                        if (startUpload && acceNum % uploadWait == 0 && acceNum - startUploadNum > uploadWait) {
                            startListen_acce = false;
                            startUpload_acce = true;
                        }
                        if (startPredict && acceNum % input_width == 1 && acceNum - startPredictNum > input_width) {
                            startListen_acce = false;
                            startPredict_acce = true;
                        }
                    }
                    if (startUpload_acce && startUpload_gyro) {
                        dataNum++;
                        String ra = real_activity;
                        startNum = acceNum - uploadWait;
                        stopNum = acceNum;

                        boolean writecsv;
                        writecsv = writeCSV();

//                        for (int i = startNum; i < stopNum; i++) {
//                            Map<String, Float> SensorData = new LinkedHashMap<String, Float>();
//                            SensorData.put("Activity", (float) activity_map.get(acceDataSet.get(i).getReal_activity()));
//                            SensorData.put("accelerometerX", acceDataSet.get(i).getAccelerometerX());
//                            SensorData.put("accelerometerY", acceDataSet.get(i).getAccelerometerY());
//                            SensorData.put("accelerometerZ", acceDataSet.get(i).getAccelerometerZ());
//                            SensorData.put("gyroscopeX", gyroDataSet.get(i).getGyroscopeX());
//                            SensorData.put("gyroscopeY", gyroDataSet.get(i).getGyroscopeY());
//                            SensorData.put("gyroscopeZ", gyroDataSet.get(i).getGyroscopeZ());
//                            SensorData.put("timeNow", (float)acceDataSet.get(i).getTimeNow());
//
//                            mDatabase.child("SensorDataSet").push().setValue(SensorData);
//                        }
                        if (writecsv) {
                            txvResult.setText("Num: " + startNum + " to " + stopNum + " " + "Upload Num: " + dataNum + "\nSave Success");
                        }
                        else {
                            txvResult.setText("Num: " + startNum + " to " + stopNum + " " + "Upload Num: " + dataNum + "\nSave Fail");
                        }

                        uploadStop();
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
                            predictData[i * channels + 4] = gyroDataSet.get(predictTime * input_width + i).getGyroscopeY();
                            predictData[i * channels + 5] = gyroDataSet.get(predictTime * input_width + i).getGyroscopeZ();
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

                        if (startUpload && gyroNum % uploadWait == 0 && gyroNum - startUploadNum > uploadWait) {
                            startListen_gyro = false;
                            startUpload_gyro = true;
                        }
                        if (startPredict && gyroNum % input_width == 1 && gyroNum - startPredictNum > input_width) {
                            startListen_gyro = false;
                            startPredict_gyro = true;
                        }
                    }
                    break;
                case Sensor.TYPE_GRAVITY:
                    if (startListen_grav) {
//                        txv.setText("Gravity"
//                                + "\nX: " + event.values[0]
//                                + "\nY: " + event.values[1]
//                                + "\nZ: " + event.values[2]);
                        gravData = event.values;
                    }
                    break;
                case Sensor.TYPE_MAGNETIC_FIELD:
                    if (startListen_magn) {
//                        txv.setText("Magnetic"
//                                + "\nX: " + event.values[0]
//                                + "\nY: " + event.values[1]
//                                + "\nZ: " + event.values[2]);
                        magnData = event.values;
                    }
                    break;
                default:
                    break;

            }
        }
    };

    private float[] phone2earth(SensorEvent event)
    {
        float[] Rotate = new float[16];
        float[] earthAcce = new float[4];

        mSensorManager.getRotationMatrix(Rotate, null, gravData, magnData);
        float[] relativacc = new float[4];
        float[] inv = new float[16];
        relativacc[0] = event.values[0];
        relativacc[1] = event.values[1];
        relativacc[2] = event.values[2];
        relativacc[3] = 0;
        android.opengl.Matrix.invertM(inv, 0, Rotate, 0);
        android.opengl.Matrix.multiplyMV(earthAcce, 0, inv, 0, relativacc, 0);

        return earthAcce;
    }

    private boolean writeCSV()
    {
        String folderName = null;
        String mFileName;
        boolean initFile = false;

        if(Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)){
            String path = Environment.getExternalStorageDirectory().getAbsolutePath();
            if (path != null) {
                folderName = path +"/ActivityCSV/";
            }
        }

        File folder = new File(folderName);
        if (!folder.exists()) {
            initFile = folder.mkdirs();
        }

        mFileName = folderName + "data.csv";
        File file = new File(mFileName);

        try {

            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true), "GBK"), 1024);
            StringBuffer sbtitle = new StringBuffer();
            if (initFile) {
                sbtitle.append(csvTitle + "\r\n");
            }
            bw.write(sbtitle.toString());

            for (int i = startNum; i < stopNum; i++) {
                StringBuffer sbdata = new StringBuffer();
                sbdata.append((float) activity_map.get(acceDataSet.get(i).getReal_activity()) + ",");
                sbdata.append(acceDataSet.get(i).getAccelerometerX() + ",");
                sbdata.append(acceDataSet.get(i).getAccelerometerY() + ",");
                sbdata.append(acceDataSet.get(i).getAccelerometerZ() + ",");
                sbdata.append(gyroDataSet.get(i).getGyroscopeX() + ",");
                sbdata.append(gyroDataSet.get(i).getGyroscopeY() + ",");
                sbdata.append(gyroDataSet.get(i).getGyroscopeZ() + ",");
                sbdata.append((float)acceDataSet.get(i).getTimeNow() + "\r\n");
                bw.write(sbdata.toString());

                if(i % 1000==0)
                    bw.flush();
            }

            bw.flush();
            bw.close();
            return true;

        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

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
        allPredictNum += 1;
        if (activityItems[maxIndex].equals(real_activity)) {
            rightPredictNum += 1;
        }
        test_accuracy = rightPredictNum / allPredictNum;

        Map<String, String> activityAccuracy = new LinkedHashMap<String, String>();
        activityAccuracy.put("Predict", activityItems[maxIndex]);
        activityAccuracy.put("Ground Truth", real_activity);
        mDatabase.child("TestAccuracy").push().setValue(activityAccuracy);

        predicttxv.setText("\n" + activityItems[0] + ": " + String.format("%.8f", results[0])
                + "\n" + activityItems[1] + ": " + String.format("%.8f", results[1])
                + "\n" + activityItems[2] + ": " + String.format("%.8f", results[2])
                + "\n" + activityItems[3] + ": " + String.format("%.8f", results[3])
                + "\n" + activityItems[4] + ": " + String.format("%.8f", results[4])
                + "\n" + activityItems[5] + ": " + String.format("%.8f", results[5])
                + "\n\nResult: " + activityItems[maxIndex] + " " + String.format("%.8f", results[maxIndex])
                + "\nTest Accuracy: " + String.format("%.8f", test_accuracy));


//        if (!activityItems[maxIndex].equals(last_activity) || last_activity.equals(null)) {
//            DateFormat dateFormat = getDateTimeInstance();
//            predicttxv.append("\nTime: " + dateFormat.format(MainActivity.timeNow) + " Activity: " + activityItems[maxIndex]);
//        }
//        last_activity = activityItems[maxIndex];

        predictStop();
        sensorStart();
    }
}
