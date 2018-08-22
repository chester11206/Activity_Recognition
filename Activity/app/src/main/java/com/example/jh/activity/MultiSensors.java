package com.example.jh.activity;

import android.app.Activity;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.text.method.ScrollingMovementMethod;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.IgnoreExtraProperties;

import java.util.ArrayList;
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

        public acceData() {}
        public void setAccelerometer(float aX, float aY, float aZ) {
            accelerometerX = aX;
            accelerometerY = aY;
            accelerometerZ = aZ;
        }
        public float getAccelerometerX() {return accelerometerX;}
        public float getAccelerometerY() {return accelerometerY;}
        public float getAccelerometerZ() {return accelerometerZ;}

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

            txv.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 15);
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
                txvResult.append("\nStop");
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
                        txv.setText("\nAccelerometer X: " + event.values[0]
                                + "\nAccelerometer Y: " + event.values[1]
                                + "\nAccelerometer Z: " + event.values[2]);
                        acceData acceData = new acceData();
                        acceData.setAccelerometer(event.values[0], event.values[1], event.values[2]);
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

                        Map<String, Float> SensorData = new LinkedHashMap<String, Float>();
                        int idx = 0;
                        for (int i = startNum; i < stopNum; i++) {
                            SensorData.put("accelerometerX " + idx, acceDataSet.get(i).getAccelerometerX());
                            SensorData.put("accelerometerY " + idx, acceDataSet.get(i).getAccelerometerY());
                            SensorData.put("accelerometerZ " + idx, acceDataSet.get(i).getAccelerometerZ());
                            SensorData.put("gyroscopeX " + idx, gyroDataSet.get(i).getGyroscopeX());
                            SensorData.put("gyroscopeY " + idx, gyroDataSet.get(i).getGyroscopeY());
                            SensorData.put("gyroscopeZ " + idx, gyroDataSet.get(i).getGyroscopeZ());
                            idx++;
                        }
                        for (String activity : activityItems) {
                            if (activity.equals(ra)) {
                                SensorData.put(activity, (float) 1);
                            } else {
                                SensorData.put(activity, (float) 0);
                            }
                        }
                        mDatabase.child("SensorDataSet").push().setValue(SensorData);

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
                        txv.setText("\nGyroscope X: " + event.values[0]
                                + "\nGyroscope Y: " + event.values[1]
                                + "\nGyroscope Z: " + event.values[2]);
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
}
