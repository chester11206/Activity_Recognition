package com.example.jh.activity;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.design.widget.TabLayout;
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
import android.widget.TextView;

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
    TextView txvResult;

    private MultiSensors multiSensorsapi;
    private SensorManager mSensorManager;

    private ViewPager mViewPager;
    private TabLayout mTabLayout;
    private View multiSensorsView;
    private View predictView;
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
        mTime.setText(dateFormat.format(TimeNow));
    }

    /** Activity Result */
    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {}

    /** Viewpage Setting */
    private void setupViewPager(ViewPager viewPager) {

        MyViewPagerAdapter adapter = new MyViewPagerAdapter();
        LayoutInflater inflater=getLayoutInflater();
        multiSensorsView = inflater.inflate(R.layout.multisensors_view, null);
        predictView = inflater.inflate(R.layout.predict_view, null);

        adapter.add(multiSensorsView, "MultiSensors");
        adapter.add(predictView, "Predict");
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
                            if (multiSensors_list.size() > 0){
                                mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
                                multiSensorsapi.start(context, mSensorManager, multiSensors_list);
                            }
                            else {
                                txvResult.setText("You haven't choose the sensors!");
                            }
                        }
                    });
                    break;
                case 2:
                    Button predictbtn = (Button) findViewById(R.id.predictbtn);
                    predictbtn.setOnClickListener(new View.OnClickListener() {
                        public void onClick(View view) {
                            MultiSensors.startPredict = true;
                        }
                    });
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
                getMenuInflater().inflate(R.menu.multisensors_menu, menu);
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

