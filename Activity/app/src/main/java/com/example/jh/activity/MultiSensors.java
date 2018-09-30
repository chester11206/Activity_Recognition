package com.example.jh.activity;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Criteria;
import android.location.GpsSatellite;
import android.location.GpsStatus;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.location.LocationProvider;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.provider.Settings;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.text.method.ScrollingMovementMethod;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

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
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static java.text.DateFormat.getDateTimeInstance;

public class MultiSensors {
    public Activity context;
    public static final String TAG = "MultiSensorsApi";
    LinearLayout ll;

    TextView txvResult;
    TextView predicttxv;

    String datacsvTitle = "Activity,AcceX,AcceY,AcceZ,GyroX,GyroY,GyroZ,Time";
    String testcsvTitle = "Right,All,Time";

    private ActivityInference activityInference;
    private SensorManager mSensorManager;
    private LocationManager mLocationManager;

    private float mSpeed = 0;
    private double mLatitude = 0;
    private double mLongitude = 0;

    private String [] activityItems = null;
    private String real_activity = null;
    private String last_activity = null;

    private boolean startPredict = false;
    private boolean startUpload = false;
    private boolean Uploading = false;
    private boolean Predicting = false;

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
    private List<acceData> noGacceDataSet = new ArrayList<acceData>();
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

    private float allPredictNum = 0;
    private float rightPredictNum = 0;
    private float test_accuracy = 0;

    private static final int uploadmsgKey = 0;
    private static final int predictmsgKey = 1;

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

    public class DataThread implements Runnable {
        private float [] Data;
        private int msgKey;

        public void setData(float [] Data)
        {
            this.Data = Data;
        }
        public void setMsgKey(int msgKey)
        {
            this.msgKey = msgKey;
        }

        @Override
        public void run () {
            try {
                Message message = new Message();
                message.what = this.msgKey;
                Bundle bundle = new Bundle();
                bundle.putFloatArray("Data", this.Data);
                message.setData(bundle);
                mHandler.sendMessage(message);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
//            while (!exit)
//            {
//
//            }
        }
    }
    private Handler mHandler = new Handler() {
        @Override
        public void handleMessage (Message msg) {
            super.handleMessage(msg);
            switch (msg.what) {
                case uploadmsgKey:
                    boolean writecsv;
                    writecsv = writeCSV();
                    if (writecsv) {
                        txvResult.setText("Num: " + startNum + " to " + stopNum + " " + "Upload Num: " + dataNum + "\nSave Success");
                    }
                    else {
                        txvResult.setText("Num: " + startNum + " to " + stopNum + " " + "Upload Num: " + dataNum + "\nSave Fail");
                    }

                    uploadStop();
                    sensorStart();
                    break;
                case predictmsgKey:
                    Bundle predictbData = msg.getData();
                    float [] predictData = predictbData.getFloatArray("Data");
                    activityPrediction(predictData);
                    break;
                default:
                    break;
            }
        }
    };

    public void start(Activity activity, SensorManager SensorManager, List<String> sensors_list, LocationManager LocationManager) {
        context = activity;
        ll = (LinearLayout) context.findViewById(R.id.sensors_display);
        activityItems = context.getResources().getStringArray(R.array.activity);

        RadioGroup rg = (RadioGroup)context.findViewById(R.id.radioGroup);
        rg.setOnCheckedChangeListener(rglistener);

        txvResult = (TextView) this.context.findViewById(R.id.multisensorstxView);
        txvResult.setMovementMethod(new ScrollingMovementMethod());
        predicttxv = (TextView) this.context.findViewById(R.id.predict_txView);
        predicttxv.setMovementMethod(new ScrollingMovementMethod());

        mSensorManager = SensorManager;
        for (String sensor : sensors_list) {

            TextView txv = new TextView(context);
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);

            txv.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 12);
            txv.setMovementMethod(new ScrollingMovementMethod());
            txv.setLayoutParams(params);
            ll.addView(txv);
            textview_map.put(sensorstype_map.get(sensor), txv);

            mSensorManager.registerListener(mSensorEventListener,
                    mSensorManager.getDefaultSensor(sensorstype_map.get(sensor)),
                    SensorManager.SENSOR_DELAY_FASTEST);
        }

        mLocationManager = LocationManager;
        initLocation();

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
                txvResult.append("\nStop Save!");
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
        Uploading = false;
    }

    private void predictStop() {
        startPredict_acce = false;
        startPredict_gyro = false;
        Predicting = false;
    }

    private RadioGroup.OnCheckedChangeListener rglistener = new RadioGroup.OnCheckedChangeListener(){

        @Override
        public void onCheckedChanged(RadioGroup rg,
                                     int checkedId) {
            RadioButton rb = (RadioButton) context.findViewById(checkedId);
            real_activity = rb.getText().toString();
        }

    };

    private void initLocation() {
        checkPermission(new String []{
                Manifest.permission.ACCESS_COARSE_LOCATION,
                Manifest.permission.ACCESS_FINE_LOCATION
        });

        if (mLocationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
            txvResult.setText("GPS Provider");
            String bestProvider = mLocationManager.getBestProvider(
                    getLocationCriteria(), true);
            // 获取位置信息
            // 如果不设置查询要求，getLastKnownLocation方法传人的参数为LocationManager.GPS_PROVIDER
            Location location = mLocationManager
                    .getLastKnownLocation(LocationManager.GPS_PROVIDER);
            updateLocation(location);
            // 监听状态
            mLocationManager.addGpsStatusListener(gpsStatusListener);
            // 绑定监听，有4个参数
            // 参数1，设备：有GPS_PROVIDER和NETWORK_PROVIDER两种
            // 参数2，位置信息更新周期，单位毫秒
            // 参数3，位置变化最小距离：当位置距离变化超过此值时，将更新位置信息
            // 参数4，监听
            // 备注：参数2和3，如果参数3不为0，则以参数3为准；参数3为0，则通过时间来定时更新；两者为0，则随时刷新

            // 1秒更新一次，或最小位移变化超过1米更新一次；
            // 注意：此处更新准确度非常低，推荐在service里面启动一个Thread，在run中sleep(10000);然后执行handler.sendMessage(),更新位置
            mLocationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER, 0, 0, locationListener);
        }
        else if (mLocationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
            txvResult.setText("NET Provider");
            Location location = mLocationManager
                    .getLastKnownLocation(LocationManager.NETWORK_PROVIDER);
            updateLocation(location);
            mLocationManager.addGpsStatusListener(gpsStatusListener);
            mLocationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER, 0, 0, locationListener);
        }
    }

    private void checkPermission(String[] permissions) {
        int permission_granted = PackageManager.PERMISSION_GRANTED;
        boolean flag = false;
        for (int i=0;i<permissions.length;i++){
            int checkPermission = ActivityCompat.checkSelfPermission(context,permissions[i]);
            if(permission_granted != checkPermission){
                flag = true;
                break;
            }
        }
        if(flag){
            ActivityCompat.requestPermissions(context,permissions,111);
            return;
        }
    }

    private Criteria getLocationCriteria() {
        Criteria criteria = new Criteria();
        // 设置定位精确度 Criteria.ACCURACY_COARSE比较粗略，Criteria.ACCURACY_FINE则比较精细
        criteria.setAccuracy(Criteria.ACCURACY_FINE);
        criteria.setSpeedRequired(true); // 设置是否要求速度
        criteria.setCostAllowed(false); // 设置是否允许运营商收费
        criteria.setBearingRequired(false); // 设置是否需要方位信息
        criteria.setAltitudeRequired(false); // 设置是否需要海拔信息
        criteria.setPowerRequirement(Criteria.POWER_LOW); // 设置对电源的需求
        return criteria;
    }

    // 状态监听
    GpsStatus.Listener gpsStatusListener = new GpsStatus.Listener() {
        public void onGpsStatusChanged(int event) {
            switch (event) {
                case GpsStatus.GPS_EVENT_FIRST_FIX: // 第一次定位
                    break;

                case GpsStatus.GPS_EVENT_SATELLITE_STATUS: // 卫星状态改变
                    checkPermission(new String []{
                            Manifest.permission.ACCESS_COARSE_LOCATION,
                            Manifest.permission.ACCESS_FINE_LOCATION
                    });
                    GpsStatus gpsStatus = mLocationManager.getGpsStatus(null); // 获取当前状态
                    int maxSatellites = gpsStatus.getMaxSatellites(); // 获取卫星颗数的默认最大值
                    Iterator<GpsSatellite> iters = gpsStatus.getSatellites()
                            .iterator(); // 创建一个迭代器保存所有卫星
                    int count = 0;
                    while (iters.hasNext() && count <= maxSatellites) {
                        GpsSatellite s = iters.next();
                        count++;
                    }
                    break;

                case GpsStatus.GPS_EVENT_STARTED: // 定位启动
                    break;

                case GpsStatus.GPS_EVENT_STOPPED: // 定位结束
                    break;
            }
        };
    };

    // 位置监听
    private LocationListener locationListener = new LocationListener() {

        /**
         * 位置信息变化时触发
         */
        public void onLocationChanged(Location location) {
            // location.getAltitude(); -- 海拔
           updateLocation(location);
        }

        /**
         * GPS状态变化时触发
         */
        public void onStatusChanged(String provider, int status, Bundle extras) {
            switch (status) {
                case LocationProvider.AVAILABLE: // GPS状态为可见时
                    break;

                case LocationProvider.OUT_OF_SERVICE: // GPS状态为服务区外时
                    break;

                case LocationProvider.TEMPORARILY_UNAVAILABLE: // GPS状态为暂停服务时
                    break;
            }
        }

        /**
         * GPS开启时触发
         */
        public void onProviderEnabled(String provider) {
            checkPermission(new String []{
                    Manifest.permission.ACCESS_COARSE_LOCATION,
                    Manifest.permission.ACCESS_FINE_LOCATION
            });
            Location location = mLocationManager.getLastKnownLocation(provider);
            updateLocation(location);
        }

        /**
         * GPS禁用时触发
         */
        public void onProviderDisabled(String provider) {
            // updateView(null);
        }

    };

    private void updateLocation(Location location) {
        if (location != null) {
            mLatitude = location.getLatitude();
            mLongitude = location.getLongitude();
            mSpeed = location.getSpeed();
            txvResult.setText("\nLatitude: " + mLatitude
                    + "\nLongitude: " + mLongitude
                    + "\nSpeed: " + mSpeed);
        }
    }

    private SensorEventListener mSensorEventListener = new SensorEventListener() {

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {}

        @Override
        public void onSensorChanged(SensorEvent event) {
            TextView txv = textview_map.get(event.sensor.getType());
            switch (event.sensor.getType()) {
                case Sensor.TYPE_ACCELEROMETER:
                    if (startListen_acce) {
                        float [] noGAcce = new float[3];
                        float [] earthAcce = new float[4];
                        float [] noGEarthAcce = new float[4];
                        earthAcce = phone2earth(event.values);
                        for (int i = 0; i < gravData.length; i++) {
                            noGAcce[i] = event.values[i] - gravData[i];
                        }
                        noGEarthAcce = phone2earth(noGAcce);

//                        txv.setText("\nAccelerometer"
//                                + "\nX: " + event.values[0]
//                                + "\nY: " + event.values[1]
//                                + "\nZ: " + event.values[2]);
                        txv.setText("\nAccelerometer(Earth)"
                                + "\nX: " + earthAcce[1]
                                + "\nY: " + earthAcce[0]
                                + "\nZ: " + earthAcce[2]
                                + "\nAccelerometer(Earth noG)"
                                + "\nX: " + noGEarthAcce[1]
                                + "\nY: " + noGEarthAcce[0]
                                + "\nZ: " + noGEarthAcce[2]);

                        String ra = real_activity;
//                        acceData acceData = new acceData(event.values[0] - gravData[0]
//                                , event.values[1] - gravData[1]
//                                , event.values[2] - gravData[2]
//                                , ra);
                        acceData acceData = new acceData(earthAcce[1], earthAcce[0], earthAcce[2], ra);
                        acceData noGacceData = new acceData(noGEarthAcce[1], noGEarthAcce[0], noGEarthAcce[2], ra);
                        acceDataSet.add(acceData);
                        noGacceDataSet.add(noGacceData);
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
                    if (startUpload_acce && startUpload_gyro && !Uploading) {
                        Uploading = true;
                        startListen_grav = false;
                        startListen_magn = false;
                        dataNum++;
                        String ra = real_activity;
                        startNum = acceNum - uploadWait;
                        stopNum = acceNum;

                        DataThread dt = new DataThread();
                        dt.setMsgKey(uploadmsgKey);
                        Thread t = new Thread(dt);
                        t.start();

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
//                        boolean writecsv;
//                        writecsv = writeCSV();
//                        if (writecsv) {
//                            txvResult.setText("Num: " + startNum + " to " + stopNum + " " + "Upload Num: " + dataNum + "\nSave Success");
//                        }
//                        else {
//                            txvResult.setText("Num: " + startNum + " to " + stopNum + " " + "Upload Num: " + dataNum + "\nSave Fail");
//                        }
//
//                        uploadStop();
//                        sensorStart();
                    }
                    if (startPredict_acce && startPredict_gyro && !Predicting) {
                        Predicting = true;
                        startListen_grav = false;
                        startListen_magn = false;
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

                        DataThread dt = new DataThread();
                        dt.setData(predictData);
                        dt.setMsgKey(predictmsgKey);
                        Thread t = new Thread(dt);
                        t.start();
                        //activityPrediction(predictData);
                    }
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    if (startListen_gyro) {
                        txv.setText("Gyroscope"
                                + "\nX: " + event.values[0]
                                + "\nY: " + event.values[1]
                                + "\nZ: " + event.values[2]
                                + "\nLatitude: " + mLatitude
                                + "\nLongitude: " + mLongitude);

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

    private float[] phone2earth(float[] value)
    {
        float[] Rotate = new float[16];
        float[] result = new float[4];
        float[] acce = new float[3];

        mSensorManager.getRotationMatrix(Rotate, null, gravData, magnData);
        acce[0] = Rotate[0] * value[0] + Rotate[3] * value[1] + Rotate[6] * value[2];
        acce[1] = Rotate[1] * value[0] + Rotate[4] * value[1] + Rotate[7] * value[2];
        acce[2] = Rotate[2] * value[0] + Rotate[5] * value[1] + Rotate[8] * value[2];

        float[] relativacc = new float[4];
        float[] inv = new float[16];
        relativacc[0] = value[0];
        relativacc[1] = value[1];
        relativacc[2] = value[2];
        relativacc[3] = 0;
        android.opengl.Matrix.invertM(inv, 0, Rotate, 0);
        android.opengl.Matrix.multiplyMV(result, 0, inv, 0, relativacc, 0);

        return result;
    }

    private boolean writeCSV()
    {
        String folderName = null;
        String mFileName;
        String noGFileName;
        boolean initFile = false;
        boolean initFilenoG = false;

        if(Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)){
            String path = Environment.getExternalStorageDirectory().getAbsolutePath();
            if (path != null) {
                folderName = path +"/ActivityCSV/";
            }
        }

        File folder = new File(folderName);
        if (!folder.exists()) {
            initFile = folder.mkdirs();
            initFilenoG = initFile;
        }

        mFileName = folderName + "data.csv";
        noGFileName = folderName + "data_noG.csv";
        File file = new File(mFileName);
        File noGfile = new File(noGFileName);
        if (!file.exists()) {
            initFile = true;
        }
        if (!noGfile.exists()) {
            initFilenoG = true;
        }

        try {

            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true), "GBK"), 1024);
            BufferedWriter noGbw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(noGfile, true), "GBK"), 1024);

            if (initFile) {
                StringBuffer sbtitle = new StringBuffer();
                sbtitle.append(datacsvTitle + "\r\n");
                bw.write(sbtitle.toString());
            }

            if (initFilenoG) {
                StringBuffer sbtitle = new StringBuffer();
                sbtitle.append(datacsvTitle + "\r\n");
                noGbw.write(sbtitle.toString());
            }

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

                StringBuffer noGsbdata = new StringBuffer();
                noGsbdata.append((float) activity_map.get(noGacceDataSet.get(i).getReal_activity()) + ",");
                noGsbdata.append(noGacceDataSet.get(i).getAccelerometerX() + ",");
                noGsbdata.append(noGacceDataSet.get(i).getAccelerometerY() + ",");
                noGsbdata.append(noGacceDataSet.get(i).getAccelerometerZ() + ",");
                noGsbdata.append(gyroDataSet.get(i).getGyroscopeX() + ",");
                noGsbdata.append(gyroDataSet.get(i).getGyroscopeY() + ",");
                noGsbdata.append(gyroDataSet.get(i).getGyroscopeZ() + ",");
                noGsbdata.append((float)noGacceDataSet.get(i).getTimeNow() + "\r\n");
                noGbw.write(noGsbdata.toString());

                if(i % 1000==0) {
                    bw.flush();
                    noGbw.flush();
                }
            }

            bw.flush();
            bw.close();
            noGbw.flush();
            noGbw.close();

            return true;

        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }
    private boolean writeTest(String predict, String real, long timeNow)
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

        mFileName = folderName + "test.csv";
        File file = new File(mFileName);
        if (!file.exists()) {
            initFile = true;
        }

        try {

            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file, true), "GBK"), 1024);
            if (initFile) {
                StringBuffer sbtitle = new StringBuffer();
                sbtitle.append(testcsvTitle + "\r\n");
                bw.write(sbtitle.toString());
            }
            SimpleDateFormat sdf= new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
            java.util.Date dt = new Date(timeNow);
            String sDateTime = sdf.format(dt);

            StringBuffer sbdata = new StringBuffer();
            sbdata.append(predict + ",");
            sbdata.append(real + ",");
            sbdata.append(sDateTime + "\r\n");
            bw.write(sbdata.toString());

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

        boolean writeTest;
        writeTest = writeTest(activityItems[maxIndex], real_activity, MainActivity.timeNow);

        if (writeTest) {
            predicttxv.setText("\nSave Predict Success"
                    + "\n" + activityItems[0] + ": " + String.format("%.8f", results[0])
                    + "\n" + activityItems[1] + ": " + String.format("%.8f", results[1])
                    + "\n" + activityItems[2] + ": " + String.format("%.8f", results[2])
                    + "\n" + activityItems[3] + ": " + String.format("%.8f", results[3])
                    + "\n" + activityItems[4] + ": " + String.format("%.8f", results[4])
                    + "\n" + activityItems[5] + ": " + String.format("%.8f", results[5])
                    + "\n\nResult: " + activityItems[maxIndex] + " " + String.format("%.8f", results[maxIndex])
                    + "\n" + rightPredictNum + " " + allPredictNum
                    + "\nTest Accuracy: " + String.format("%.8f", test_accuracy));
        }
        else {
            predicttxv.setText("\nSave Predict Fail");
        }


//        if (!activityItems[maxIndex].equals(last_activity) || last_activity.equals(null)) {
//            DateFormat dateFormat = getDateTimeInstance();
//            predicttxv.append("\nTime: " + dateFormat.format(MainActivity.timeNow) + " Activity: " + activityItems[maxIndex]);
//        }
//        last_activity = activityItems[maxIndex];

        predictStop();
        sensorStart();
    }
}
