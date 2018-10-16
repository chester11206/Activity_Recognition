package com.example.jh.activity;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.SensorManager;
import android.Manifest;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.provider.Settings;
import android.support.annotation.NonNull;
import android.support.design.widget.TabLayout;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.view.PagerAdapter;
import android.support.v4.view.ViewPager;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import java.text.DateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

import static java.text.DateFormat.getDateTimeInstance;

public class MainActivity extends AppCompatActivity {

    public Activity context = this;
    public static final String TAG = "Activity";
    private static final int REQUEST_CODE_WRITE_EXTERNAL_STORAGE_PERMISSION = 1;
    private static final int REQUEST_CODE_LOCATION_PERMISSION = 1001;
    TextView txvResult;

    private MultiSensors multiSensorsapi;
    private SensorManager mSensorManager;
    private LocationManager mLocationManager;

    private ViewPager mViewPager;
    private TabLayout mTabLayout;
    private View multiSensorsView;
    public static int lastPosition = 0;

    boolean[] multiflags = new boolean[]{};//init multichoice = false
    String[] multiSensorItems = null;
    private List<String> multiSensors_list = new ArrayList<String>();


    private static final int msgKey1 = 1;
    private TextView mTime;
    public static long timeNow = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mTime = (TextView) findViewById(R.id.mytime);
        new TimeThread().start();

        /** init three api */
        multiSensorsapi = new com.example.jh.activity.MultiSensors();

        /** set viewpage and layout */
        mViewPager = (ViewPager) findViewById(R.id.container);
        setupViewPager(mViewPager);
        mViewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() {
            @Override
            public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {}
            @Override
            public void onPageScrollStateChanged(int state) {}
            @Override
            public void onPageSelected(int position) {
                Log.i(TAG, "selected page = " + position);
                lastPosition = position;
            }
        });
        mTabLayout = (TabLayout) findViewById(R.id.tabs);
        mTabLayout.setupWithViewPager(mViewPager);

        /** set multichoice dialog */
        multiSensorItems = getResources().getStringArray(R.array.multiSensors);
        multiflags = new boolean[multiSensorItems.length];
        Arrays.fill(multiflags, false);

        // Check whether this app has write external storage permission or not.
        int writeExternalStoragePermission = ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        // If do not grant write external storage permission.
        if(writeExternalStoragePermission!= PackageManager.PERMISSION_GRANTED)
        {
            // Request user to grant write external storage permission.
            ActivityCompat.requestPermissions(context, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CODE_WRITE_EXTERNAL_STORAGE_PERMISSION);
        }
    }

    /** Show Now Time */
    public class TimeThread extends Thread {
        @Override
        public void run () {
            do {
                try {
                    Thread.sleep(1000);
                    Message msg = new Message();
                    msg.what = msgKey1;
                    mHandler.sendMessage(msg);
                }
                catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } while(true);
        }
    }
    private Handler mHandler = new Handler() {
        @Override
        public void handleMessage (Message msg) {
            super.handleMessage(msg);
            switch (msg.what) {
                case msgKey1:
                    getTime();
                    break;
                default:
                    break;
            }
        }
    };
    public void getTime(){
        Calendar cal = Calendar.getInstance();
        Date now = new Date();
        cal.setTime(now);
        long TimeNow = cal.getTimeInMillis();
        timeNow = TimeNow;
        DateFormat dateFormat = getDateTimeInstance();
        mTime.setText("\t" + dateFormat.format(TimeNow));
    }

    private boolean initGPS() {
        mLocationManager = (LocationManager) getApplicationContext().getSystemService(Context.LOCATION_SERVICE); // 位置
        boolean haveGPS = mLocationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);

        if (!haveGPS) {
            // TODO: Open GPS
            Toast.makeText(context.getApplicationContext(), "Please open your GPS", Toast.LENGTH_SHORT).show();
            final AlertDialog.Builder dialog = new AlertDialog.Builder(context);
            dialog.setTitle("Open GPS");
            dialog.setMessage("To detect your speed, please open your GPS");
            dialog.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface arg0, int arg1) {
                    // 转到手机设置界面，用户设置GPS
                    Intent intent = new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS);
                    Toast.makeText(context.getApplicationContext(), "Open GPS and then click the return button", Toast.LENGTH_SHORT).show();
                    context.startActivityForResult(intent, 0); // 设置完成后返回到原来的界面

                }
            });
            dialog.setNeutralButton("Cancel", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface arg0, int arg1) {
                    arg0.dismiss();
                }
            });
            dialog.show();
        }

        return haveGPS;
    }

    /** Activity Result */
    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
//        if(requestCode == 0){
//            if (mLocationManager.isProviderEnabled(android.location.LocationManager.GPS_PROVIDER)){
//                initGPS();
//            }
//        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {
            case REQUEST_CODE_WRITE_EXTERNAL_STORAGE_PERMISSION:
                int grantResultsLength_write = grantResults.length;
                if (grantResultsLength_write > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(getApplicationContext(), "You grant write external storage permission. Please click original button again to continue.", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(getApplicationContext(), "You denied write external storage permission.", Toast.LENGTH_LONG).show();
                }
                break;
            case REQUEST_CODE_LOCATION_PERMISSION:
                int grantResultsLength_location = grantResults.length;
                if (grantResultsLength_location > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(getApplicationContext(), "You grant location permission. Please click original button again to continue.", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(getApplicationContext(), "You denied location permission.", Toast.LENGTH_LONG).show();
                }

            default:
                break;
        }
    }


        /** Viewpage Setting */
    private void setupViewPager(ViewPager viewPager) {

        MyViewPagerAdapter adapter = new MyViewPagerAdapter();
        LayoutInflater inflater=getLayoutInflater();
        multiSensorsView = inflater.inflate(R.layout.multisensors_view, null);

        adapter.add(multiSensorsView, "MultiSensors");
        viewPager.setAdapter(adapter);
    }

    public class MyViewPagerAdapter extends PagerAdapter {
        private final List<View> mListViews = new ArrayList<>();
        private final List<String> mListTitles = new ArrayList<>();

        @Override
        public void destroyItem(ViewGroup container, int position, Object object) 	{
            container.removeView(mListViews.get(position));
        }

        @Override
        public Object instantiateItem(ViewGroup container, int position) {
            container.addView(mListViews.get(position), 0);
            switch (position) {
                case 0:
                    Button multisensorsbtn = (Button) findViewById(R.id.multisensorsbtn);
                    multisensorsbtn.setOnClickListener(new View.OnClickListener() {
                        public void onClick(View view) {
                            txvResult = (TextView) findViewById(R.id.multisensorstxView);
                            txvResult.setMovementMethod(new ScrollingMovementMethod());
//                            if (multiSensors_list.size() <= 0){
//                                txvResult.setText("You haven't choose the sensors!");
//
//                            }
//                            else {
//                                boolean haveGPS = initGPS();
//                                if (haveGPS) {
//                                    mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
//                                    multiSensorsapi.start(context, mSensorManager, multiSensors_list);
//                                }
//                            }
                            boolean haveGPS = initGPS();
                            if (haveGPS) {
                                mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
                                multiSensorsapi.start(context, mSensorManager, multiSensorItems);
                            }
                        }
                    });
                    break;
                case 2:
                    break;
                default:
                    break;

            }

            return mListViews.get(position);
        }

        @Override
        public int getCount() {
            return  mListViews.size();
        }

        @Override
        public boolean isViewFromObject(View arg0, Object arg1) {
            return arg0==arg1;
        }

        public void add(View view, String title) {
            mListViews.add(view);
            mListTitles.add(title);
        }

        @Override
        public CharSequence getPageTitle(int position) {
            return mListTitles.get(position);
        }
    }



    /** menu setting */
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        switch (mViewPager.getCurrentItem()) {
            case 0:
                //getMenuInflater().inflate(R.menu.multisensors_menu, menu);
                break;
            default:
                break;

        }
        return true;
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        invalidateOptionsMenu();
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.set_multiSensors) {
            showDialog(1);
        }

        return super.onOptionsItemSelected(item);
    }

    /** create dialog */
    @Override
    protected Dialog onCreateDialog(int id) {
        Dialog dialog = null;
        switch (id) {
            case 1:
                boolean[] tempMultiFlags = multiflags.clone();
                AlertDialog.Builder builderMultiSensor = new AlertDialog.Builder(this);
                builderMultiSensor.setTitle("Choose Sensors");
                builderMultiSensor.setMultiChoiceItems(multiSensorItems, multiflags, new DialogInterface.OnMultiChoiceClickListener() {

                    @Override
                    public void onClick(DialogInterface dialog, int which, boolean isChecked) {
                        tempMultiFlags[which] = isChecked;
                    }
                });
                builderMultiSensor.setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        boolean hasChoose = false;
                        for (boolean flag : tempMultiFlags) {
                            if (flag) {
                                hasChoose = true;
                                break;
                            }
                        }
                        if (hasChoose){
                            List<String> result = new ArrayList<String>();
                            multiflags = tempMultiFlags.clone();
                            for (int i = 0; i < multiflags.length; i++) {
                                if(multiflags[i])
                                {
                                    result.add(multiSensorItems[i]);
                                }
                            }
                            multiSensors_list = new ArrayList<String>(result);
                        }
                        else {
                            txvResult = (TextView) findViewById(R.id.multisensorstxView);
                            txvResult.setMovementMethod(new ScrollingMovementMethod());
                            txvResult.setText("");
                            txvResult.setText("You haven't choose the sensors!");
                        }
                    }
                });
                builderMultiSensor.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {

                    }
                });
                dialog = builderMultiSensor.create();
                break;

            default:
                break;
        }
        return dialog;
    }
}

