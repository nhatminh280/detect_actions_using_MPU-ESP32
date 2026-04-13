#include <Adafruit_GFX.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <math.h>
#include <string.h>
#include "gesture_model-1.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define LED_PIN 2

namespace
{

    constexpr int NUM_AXES = 6;
    constexpr int WINDOW_SIZE = 100;
    constexpr int WINDOW_OVERLAP = 50;
    constexpr int RAW_SAMPLE_RATE_HZ = 200;
    constexpr int RAW_WINDOW_SIZE = WINDOW_SIZE * 2;
    constexpr int RAW_WINDOW_STRIDE = (WINDOW_SIZE - WINDOW_OVERLAP) * 2;

    //  Features
    constexpr int FEATURES_PER_AXIS = 15;
    constexpr int NUM_STREAMS = 3;
    // 15 * 6 * 3 = 270
    // indices   0– 89: luồng A (HPF 5Hz)
    // indices  90–179: luồng B (HPF 0.5Hz)
    // indices 180–269: luồng C (RAW normalize gravity)
    constexpr int NUM_FEATURES = FEATURES_PER_AXIS * NUM_AXES * NUM_STREAMS;

    // ─── Luồng A – HPF 2Hz ───────────────────────────────────────────────────
    constexpr float HPF_A_CUTOFF_HZ = 2.0f;
    constexpr float ACCEL_DEADBAND_A = 0.08f;
    constexpr float GYRO_DEADBAND_A = 0.04f;

    // ─── Luồng B – HPF 0.5Hz (hạ từ 1.5 → 0.5 để bắt Thumbs Up ~0.6Hz) ──────
    constexpr float HPF_B_CUTOFF_HZ = 0.3f;
    // Không deadband cho B và C

    // ─── Shared ──────────────────────────────────────────────────────────────
    constexpr float PI_F = 3.14159265f;
    constexpr float SAMPLE_DT = 1.0f / RAW_SAMPLE_RATE_HZ;
    constexpr float EPSILON = 1e-6f;
    constexpr int MIN_ACTIVE_VALUES = 10;

    // HPF alpha (tự tính lại từ cutoff mới)
    constexpr float HPF_A_RC = 1.0f / (2.0f * PI_F * HPF_A_CUTOFF_HZ);
    constexpr float HPF_A_ALPHA = HPF_A_RC / (HPF_A_RC + SAMPLE_DT);
    constexpr float HPF_B_RC = 1.0f / (2.0f * PI_F * HPF_B_CUTOFF_HZ);
    constexpr float HPF_B_ALPHA = HPF_B_RC / (HPF_B_RC + SAMPLE_DT);

    // ─── Labels ──────────────────────────────────────────────────────────────
    const char *GESTURE_LABELS[3] = {
        "Clapping",
        "Fist Making",
        "Thumbs Up"};

    // ─── Scaler (270 features) ────────────────────────────────────────────────
    const float FEATURE_MEAN[NUM_FEATURES] = {
        -0.00656023f,
        0.78533324f,
        -3.07247746f,
        3.06074370f,
        6.13322115f,
        0.37568434f,
        62.93340395f,
        -0.11953220f,
        0.01565845f,
        0.78545096f,
        0.91400345f,
        1.01523976f,
        0.95804873f,
        1.02085620f,
        1.01378767f,
        0.00467265f,
        1.30264843f,
        -5.28122556f,
        4.22902820f,
        9.51025377f,
        0.63795298f,
        107.46140881f,
        -0.06707810f,
        0.16873374f,
        1.30278515f,
        1.19336163f,
        1.27471597f,
        1.24404782f,
        1.30148822f,
        1.30805170f,
        0.02230689f,
        1.59605914f,
        -8.44790512f,
        5.71204879f,
        14.15995391f,
        0.68405883f,
        112.62830408f,
        -0.03155596f,
        0.18057720f,
        1.59649602f,
        2.55132264f,
        2.64998168f,
        2.54024659f,
        2.59453541f,
        2.52188164f,
        0.00001845f,
        0.34491624f,
        -1.28854583f,
        1.53397574f,
        2.82252157f,
        0.13539730f,
        23.29706764f,
        -0.00457596f,
        0.00221697f,
        0.34494679f,
        0.22599253f,
        0.23695905f,
        0.24043613f,
        0.24774727f,
        0.25520104f,
        -0.00015862f,
        0.11531685f,
        -0.60629917f,
        0.46909643f,
        1.07539560f,
        0.03952962f,
        6.89973258f,
        -0.00001399f,
        0.00027103f,
        0.11533836f,
        0.11930655f,
        0.12626010f,
        0.12529370f,
        0.12648205f,
        0.12830967f,
        -0.00003040f,
        0.07548889f,
        -0.35012113f,
        0.31196371f,
        0.66208485f,
        0.02711104f,
        4.81907793f,
        -0.00021333f,
        0.00033730f,
        0.07550953f,
        0.10420054f,
        0.10978887f,
        0.10783631f,
        0.10934075f,
        0.11116461f,
        -0.00869508f,
        0.82694811f,
        -3.22721398f,
        3.21864481f,
        6.44585880f,
        0.40524157f,
        65.86180859f,
        -0.15037870f,
        0.03176586f,
        0.82709873f,
        1.05545425f,
        1.03081400f,
        0.87939058f,
        0.85445721f,
        0.82420360f,
        0.00590099f,
        1.37319392f,
        -5.58439020f,
        4.45810734f,
        10.04249754f,
        0.68018606f,
        112.14167742f,
        -0.08792610f,
        0.19623118f,
        1.37335653f,
        1.33048795f,
        1.32586817f,
        1.20781305f,
        1.18653947f,
        1.17800346f,
        0.02495945f,
        1.67455220f,
        -8.94307363f,
        5.85044580f,
        14.79351943f,
        0.73004261f,
        117.04681033f,
        -0.05288645f,
        0.22078129f,
        1.67505778f,
        2.78913240f,
        2.61523712f,
        2.19327609f,
        2.03475071f,
        1.93809541f,
        -0.00005671f,
        0.36408737f,
        -1.36034248f,
        1.62105198f,
        2.98139447f,
        0.14668130f,
        24.52780540f,
        -0.01244928f,
        0.00807590f,
        0.36411625f,
        0.20316692f,
        0.20656235f,
        0.20860794f,
        0.21360697f,
        0.22133955f,
        0.00005485f,
        0.12295690f,
        -0.64227138f,
        0.49705477f,
        1.13932615f,
        0.04618328f,
        7.65724379f,
        -0.00199982f,
        0.00388786f,
        0.12296758f,
        0.06893228f,
        0.07005855f,
        0.06999083f,
        0.07133903f,
        0.07369634f,
        -0.00000069f,
        0.08128447f,
        -0.37241840f,
        0.33193863f,
        0.70435703f,
        0.03307312f,
        5.49628924f,
        -0.00240860f,
        0.00308177f,
        0.08129156f,
        0.05051355f,
        0.05124329f,
        0.05078466f,
        0.05180292f,
        0.05332370f,
        -0.00356264f,
        0.84565799f,
        -3.29593809f,
        3.29688454f,
        6.59282262f,
        0.41256516f,
        67.17214082f,
        -0.14604738f,
        0.03745186f,
        0.84573598f,
        0.70677922f,
        0.74483024f,
        0.71133470f,
        0.76911223f,
        0.77196414f,
        0.00237071f,
        1.40520846f,
        -5.71954114f,
        4.56367402f,
        10.28321516f,
        0.69373364f,
        114.48345820f,
        -0.09283436f,
        0.19451080f,
        1.40531953f,
        1.05065081f,
        1.08892084f,
        1.07515951f,
        1.12616334f,
        1.14622604f,
        0.01021902f,
        1.71035213f,
        -9.16784134f,
        5.91992851f,
        15.08776986f,
        0.74104652f,
        119.18165531f,
        -0.06656919f,
        0.20323405f,
        1.71055528f,
        1.60356262f,
        1.67429772f,
        1.63458291f,
        1.73763650f,
        1.75325836f,
        -0.00000867f,
        0.37267750f,
        -1.39307607f,
        1.65978606f,
        3.05286213f,
        0.14995043f,
        25.04815911f,
        -0.01259695f,
        0.00825300f,
        0.37270623f,
        0.20439971f,
        0.20767374f,
        0.21138901f,
        0.21771274f,
        0.22582126f,
        0.00001934f,
        0.12585609f,
        -0.65742466f,
        0.50944892f,
        1.16687358f,
        0.04720081f,
        7.82587981f,
        -0.00207113f,
        0.00390562f,
        0.12586638f,
        0.06712599f,
        0.06851458f,
        0.06957394f,
        0.07187450f,
        0.07468583f,
        -0.00000137f,
        0.08319874f,
        -0.38132632f,
        0.33983890f,
        0.72116521f,
        0.03380291f,
        5.61352136f,
        -0.00243277f,
        0.00313259f,
        0.08320539f,
        0.04886794f,
        0.04981893f,
        0.05039479f,
        0.05224302f,
        0.05406280f,
    };

    const float FEATURE_SCALE[NUM_FEATURES] = {
        0.01190003f,
        0.56284047f,
        3.89680021f,
        2.12141278f,
        5.73525526f,
        0.28603469f,
        49.77972330f,
        0.20552433f,
        0.09765324f,
        0.56284024f,
        1.00552018f,
        0.96632706f,
        0.98370631f,
        0.97882856f,
        1.01588081f,
        0.01673807f,
        0.74877594f,
        3.25451070f,
        2.61176818f,
        5.38113031f,
        0.44652822f,
        80.26196238f,
        0.21896091f,
        0.30042624f,
        0.74873974f,
        1.26327134f,
        1.18525178f,
        1.24857187f,
        1.21412362f,
        1.28043170f,
        0.02359843f,
        1.04556108f,
        8.22375806f,
        4.08878536f,
        11.75145436f,
        0.46936622f,
        82.04060954f,
        0.26702831f,
        0.28571308f,
        1.04539835f,
        2.00887661f,
        1.85498062f,
        1.95387814f,
        1.85415555f,
        1.94892161f,
        0.00395820f,
        0.24711062f,
        0.86839799f,
        1.00281849f,
        1.74300074f,
        0.11405150f,
        20.75084744f,
        0.02053871f,
        0.01549161f,
        0.24709969f,
        0.32496863f,
        0.31998277f,
        0.32219575f,
        0.32641082f,
        0.33468066f,
        0.00189173f,
        0.12125925f,
        0.68767646f,
        0.45525097f,
        1.10806674f,
        0.04551524f,
        8.14922021f,
        0.00089819f,
        0.00444566f,
        0.12125365f,
        0.14766118f,
        0.14046297f,
        0.14006730f,
        0.14325384f,
        0.14508114f,
        0.00149278f,
        0.07245608f,
        0.33587758f,
        0.30232801f,
        0.60116992f,
        0.03082324f,
        5.54018319f,
        0.00419237f,
        0.00500087f,
        0.07244996f,
        0.10693638f,
        0.09853505f,
        0.09586608f,
        0.09685115f,
        0.09906209f,
        0.01260024f,
        0.59543674f,
        4.12385568f,
        2.22441355f,
        6.06201626f,
        0.30198525f,
        52.44127619f,
        0.20407807f,
        0.10434099f,
        0.59542435f,
        1.10892752f,
        1.02632446f,
        0.98176530f,
        0.98034327f,
        1.00241497f,
        0.01815846f,
        0.79323417f,
        3.44608713f,
        2.77206689f,
        5.70518991f,
        0.47184023f,
        84.71008367f,
        0.22453027f,
        0.30480497f,
        0.79318244f,
        1.36942104f,
        1.26721055f,
        1.28723328f,
        1.27473697f,
        1.32966720f,
        0.02474373f,
        1.11195675f,
        8.70941195f,
        4.27911723f,
        12.46852834f,
        0.49585464f,
        86.85411339f,
        0.27825894f,
        0.28456402f,
        1.11175068f,
        2.13845798f,
        1.90275119f,
        1.89688929f,
        1.83298969f,
        1.93665654f,
        0.00408209f,
        0.26128438f,
        0.92014940f,
        1.06139041f,
        1.84692616f,
        0.12065947f,
        21.89432468f,
        0.02233536f,
        0.01787455f,
        0.26127603f,
        0.35410423f,
        0.35289284f,
        0.35630192f,
        0.36143849f,
        0.37079193f,
        0.00164626f,
        0.12739117f,
        0.72543492f,
        0.48035056f,
        1.16905905f,
        0.04819208f,
        8.58440838f,
        0.00347268f,
        0.00777014f,
        0.12739151f,
        0.14959968f,
        0.15049518f,
        0.15185123f,
        0.15490805f,
        0.15839656f,
        0.00109369f,
        0.07562212f,
        0.35290996f,
        0.31725277f,
        0.63089595f,
        0.03265589f,
        5.79974510f,
        0.00605156f,
        0.00740970f,
        0.07562242f,
        0.09700552f,
        0.09715587f,
        0.09828547f,
        0.09980157f,
        0.10251707f,
        0.01154591f,
        0.60992757f,
        4.22199851f,
        2.28070508f,
        6.21026727f,
        0.30895469f,
        53.66754403f,
        0.21023949f,
        0.10835946f,
        0.60993912f,
        0.98003426f,
        0.96383780f,
        0.96574586f,
        0.98667706f,
        1.02081720f,
        0.01673934f,
        0.81262880f,
        3.52920542f,
        2.84176774f,
        5.84570127f,
        0.48305792f,
        86.70918095f,
        0.22859926f,
        0.31126944f,
        0.81261259f,
        1.32453467f,
        1.27776927f,
        1.31678659f,
        1.31040141f,
        1.36826665f,
        0.02288896f,
        1.14093937f,
        8.91920007f,
        4.36898157f,
        12.77856600f,
        0.50831534f,
        88.96522681f,
        0.29444366f,
        0.29733454f,
        1.14091017f,
        1.92663397f,
        1.80220686f,
        1.91318115f,
        1.87313720f,
        1.99114332f,
        0.00418871f,
        0.26758850f,
        0.94255793f,
        1.08707194f,
        1.89249388f,
        0.12355412f,
        22.41864135f,
        0.02281689f,
        0.01818096f,
        0.26758127f,
        0.36561481f,
        0.36437329f,
        0.36679103f,
        0.37138785f,
        0.38047683f,
        0.00168583f,
        0.13045356f,
        0.74264184f,
        0.49215282f,
        1.19737964f,
        0.04932159f,
        8.78829836f,
        0.00357079f,
        0.00786920f,
        0.13045452f,
        0.15465443f,
        0.15518565f,
        0.15630761f,
        0.15902290f,
        0.16255779f,
        0.00111476f,
        0.07743773f,
        0.36147588f,
        0.32490976f,
        0.64627727f,
        0.03342795f,
        5.93444987f,
        0.00615714f,
        0.00754638f,
        0.07743860f,
        0.10019458f,
        0.10014488f,
        0.10094487f,
        0.10246637f,
        0.10518746f,
    };

    // ─── Hardware ─────────────────────────────────────────────────────────────
    Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
    Adafruit_MPU6050 mpu;
    Eloquent::ML::Port::XGBClassifier classifier;

    // ─── Ring buffers 200Hz – 3 luồng ────────────────────────────────────────
    float rawWindowA[RAW_WINDOW_SIZE][NUM_AXES]; // HPF 5Hz
    float rawWindowB[RAW_WINDOW_SIZE][NUM_AXES]; // HPF 0.5Hz
    float rawWindowC[RAW_WINDOW_SIZE][NUM_AXES]; // RAW – gravity

    // ─── Model windows 100Hz ─────────────────────────────────────────────────
    float modelWindowA[WINDOW_SIZE][NUM_AXES];
    float modelWindowB[WINDOW_SIZE][NUM_AXES];
    float modelWindowC[WINDOW_SIZE][NUM_AXES];

    float featureVector[NUM_FEATURES];

    // ─── Filter states ────────────────────────────────────────────────────────
    float prevInputA[NUM_AXES] = {0};
    float prevOutputA[NUM_AXES] = {0};
    float prevInputB[NUM_AXES] = {0};
    float prevOutputB[NUM_AXES] = {0};

    int rawSamplesCollected = 0;
    bool filterReady = false;
    uint32_t nextSampleMicros = 0;

    // ────────────────────────────────────────────────────────────────────────
    // Utilities
    // ────────────────────────────────────────────────────────────────────────
    float absf(float v) { return v < 0.0f ? -v : v; }

    void copyAxis(const float window[WINDOW_SIZE][NUM_AXES], int axis, float *out)
    {
        for (int i = 0; i < WINDOW_SIZE; i++)
            out[i] = window[i][axis];
    }

    void insertionSort(float *v, int n)
    {
        for (int i = 1; i < n; i++)
        {
            float key = v[i];
            int j = i - 1;
            while (j >= 0 && v[j] > key)
            {
                v[j + 1] = v[j];
                j--;
            }
            v[j + 1] = key;
        }
    }

    float percentile(const float *s, int n, float q)
    {
        float pos = (n - 1) * q;
        int lo = (int)floorf(pos), hi = (int)ceilf(pos);
        if (lo == hi)
            return s[lo];
        return s[lo] * (1.0f - (pos - lo)) + s[hi] * (pos - lo);
    }

    float fftMagnitudeBin(const float *v, int n, int bin)
    {
        float re = 0.0f, im = 0.0f;
        float f = (2.0f * PI_F * bin) / n;
        for (int i = 0; i < n; i++)
        {
            re += v[i] * cosf(f * i);
            im -= v[i] * sinf(f * i);
        }
        return sqrtf(re * re + im * im);
    }

    // ─── 15 features / axis ───────────────────────────────────────────────────
    void extractAxisFeatures(const float *v, float *out)
    {
        float sorted[WINDOW_SIZE];
        float mean = 0, minV = v[0], maxV = v[0];
        float mAbs = 0, rms = 0, diffSum = 0;

        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            mean += v[i];
            mAbs += absf(v[i]);
            rms += v[i] * v[i];
            sorted[i] = v[i];
            if (v[i] < minV)
                minV = v[i];
            if (v[i] > maxV)
                maxV = v[i];
            if (i > 0)
                diffSum += absf(v[i] - v[i - 1]);
        }

        mean /= WINDOW_SIZE;
        mAbs /= WINDOW_SIZE;
        rms = sqrtf(rms / WINDOW_SIZE);

        float var = 0;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            float c = v[i] - mean;
            var += c * c;
        }
        float std = sqrtf(var / WINDOW_SIZE);

        insertionSort(sorted, WINDOW_SIZE);

        out[0] = mean;
        out[1] = std;
        out[2] = minV;
        out[3] = maxV;
        out[4] = maxV - minV;
        out[5] = mAbs;
        out[6] = diffSum;
        out[7] = percentile(sorted, WINDOW_SIZE, 0.25f);
        out[8] = percentile(sorted, WINDOW_SIZE, 0.75f);
        out[9] = rms;

        // FFT bins 1–5 (bỏ bin 0 = DC component)
        for (int b = 1; b <= 5; b++)
        {
            out[9 + b] = fftMagnitudeBin(v, WINDOW_SIZE, b);
        }
        // out[10]=bin1, out[11]=bin2, out[12]=bin3, out[13]=bin4, out[14]=bin5
    }

    // ─── Extract triple stream → 270 features ────────────────────────────────
    void extractFeaturesTriple(
        const float wA[WINDOW_SIZE][NUM_AXES],
        const float wB[WINDOW_SIZE][NUM_AXES],
        const float wC[WINDOW_SIZE][NUM_AXES],
        float *features)
    {
        float axis[WINDOW_SIZE];
        // Luồng A 0–89
        for (int a = 0; a < NUM_AXES; a++)
        {
            copyAxis(wA, a, axis);
            extractAxisFeatures(axis, features + a * FEATURES_PER_AXIS);
        }
        // Luồng B 90–179
        int offB = NUM_AXES * FEATURES_PER_AXIS;
        for (int a = 0; a < NUM_AXES; a++)
        {
            copyAxis(wB, a, axis);
            extractAxisFeatures(axis, features + offB + a * FEATURES_PER_AXIS);
        }
        // Luồng C 180–269
        int offC = 2 * NUM_AXES * FEATURES_PER_AXIS;
        for (int a = 0; a < NUM_AXES; a++)
        {
            copyAxis(wC, a, axis);
            extractAxisFeatures(axis, features + offC + a * FEATURES_PER_AXIS);
        }
    }

    void scaleFeatures(float *f)
    {
        for (int i = 0; i < NUM_FEATURES; i++)
            f[i] = (f[i] - FEATURE_MEAN[i]) / (FEATURE_SCALE[i] + EPSILON);
    }

    // ─── HPF filters ──────────────────────────────────────────────────────────
    float highPassA(float x, int axis)
    {
        if (!filterReady)
        {
            prevInputA[axis] = x;
            prevOutputA[axis] = 0;
            return 0;
        }
        float y = HPF_A_ALPHA * (prevOutputA[axis] + x - prevInputA[axis]);
        prevInputA[axis] = x;
        prevOutputA[axis] = y;
        return y;
    }

    float highPassB(float x, int axis)
    {
        if (!filterReady)
        {
            prevInputB[axis] = x;
            prevOutputB[axis] = 0;
            return 0;
        }
        float y = HPF_B_ALPHA * (prevOutputB[axis] + x - prevInputB[axis]);
        prevInputB[axis] = x;
        prevOutputB[axis] = y;
        return y;
    }

    // ─── Preprocess 1 sample → ghi vào 3 buffers ─────────────────────────────
    void preprocessSample(const sensors_event_t &a, const sensors_event_t &g)
    {
        const float raw[NUM_AXES] = {
            a.acceleration.x, a.acceleration.y, a.acceleration.z,
            g.gyro.x, g.gyro.y, g.gyro.z};

        for (int axis = 0; axis < NUM_AXES; axis++)
        {
            // Luồng A – HPF 5Hz + deadband
            float fA = highPassA(raw[axis], axis);
            float dbA = axis < 3 ? ACCEL_DEADBAND_A : GYRO_DEADBAND_A;
            if (absf(fA) < dbA)
                fA = 0.0f;
            rawWindowA[rawSamplesCollected][axis] = fA;

            // Luồng B – HPF 0.5Hz, không deadband
            rawWindowB[rawSamplesCollected][axis] = highPassB(raw[axis], axis);

            // Luồng C – RAW, normalize gravity sau ở downsampleWindow
            rawWindowC[rawSamplesCollected][axis] = raw[axis];
        }
    }

    bool hasEnoughMotion()
    {
        int active = 0;
        for (int i = 0; i < RAW_WINDOW_SIZE; i++)
            for (int a = 0; a < NUM_AXES; a++)
                if (rawWindowA[i][a] != 0.0f)
                    active++;
        return active >= MIN_ACTIVE_VALUES;
    }

    void downsampleWindow()
    {
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            int idx = i * 2;
            for (int a = 0; a < NUM_AXES; a++)
            {
                modelWindowA[i][a] = rawWindowA[idx][a];
                modelWindowB[i][a] = rawWindowB[idx][a];
                modelWindowC[i][a] = rawWindowC[idx][a];
            }
        }
        // Normalize gravity cho luồng C: trừ mean cửa sổ cho 3 accel axes
        // Giữ biến đổi tư thế, bỏ DC offset tuyệt đối
        for (int a = 0; a < 3; a++)
        {
            float mean = 0.0f;
            for (int i = 0; i < WINDOW_SIZE; i++)
                mean += modelWindowC[i][a];
            mean /= WINDOW_SIZE;
            for (int i = 0; i < WINDOW_SIZE; i++)
                modelWindowC[i][a] -= mean;
        }
    }

    void showStatus(const char *title, const char *detail, int cls = -1)
    {
        display.clearDisplay();
        display.setTextColor(WHITE);
        display.setTextSize(1);
        display.setCursor(0, 0);
        display.println(title);
        display.println();
        display.println(detail);
        if (cls >= 0)
        {
            display.println();
            display.print("LED: ");
            display.println(cls == 1 ? "ON" : "OFF");
        }
        display.display();
    }

    void runInference()
    {
        if (!hasEnoughMotion())
        {
            digitalWrite(LED_PIN, LOW);
            Serial.println("No motion");
            showStatus("Status", "No motion");
            return;
        }

        downsampleWindow();
        extractFeaturesTriple(modelWindowA, modelWindowB, modelWindowC, featureVector);
        scaleFeatures(featureVector);

        const int cls = classifier.predict(featureVector);
        digitalWrite(LED_PIN, cls == 1 ? HIGH : LOW);

        const char *label = (cls >= 0 && cls < 3) ? GESTURE_LABELS[cls] : "Unknown";
        Serial.print("Predicted: ");
        Serial.print(cls);
        Serial.print(" -> ");
        Serial.println(label);
        showStatus("Gesture", label, cls);
    }

    void slideWindow()
    {
        memmove(rawWindowA, rawWindowA + RAW_WINDOW_STRIDE,
                (RAW_WINDOW_SIZE - RAW_WINDOW_STRIDE) * sizeof(rawWindowA[0]));
        memmove(rawWindowB, rawWindowB + RAW_WINDOW_STRIDE,
                (RAW_WINDOW_SIZE - RAW_WINDOW_STRIDE) * sizeof(rawWindowB[0]));
        memmove(rawWindowC, rawWindowC + RAW_WINDOW_STRIDE,
                (RAW_WINDOW_SIZE - RAW_WINDOW_STRIDE) * sizeof(rawWindowC[0]));
        rawSamplesCollected = RAW_WINDOW_SIZE - RAW_WINDOW_STRIDE;
    }
}

void setup()
{
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    if (!mpu.begin())
    {
        Serial.println("MPU6050 not found");
        while (true)
            delay(10);
    }
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);

    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
    {
        Serial.println("SSD1306 failed");
        while (true)
            delay(10);
    }
    display.clearDisplay();
    display.display();
    showStatus("Status", "Triple stream ready");
    nextSampleMicros = micros();
}

void loop()
{
    const uint32_t now = micros();
    if ((int32_t)(now - nextSampleMicros) < 0)
        return;

    nextSampleMicros += 1000000UL / RAW_SAMPLE_RATE_HZ;
    if ((int32_t)(now - nextSampleMicros) > 0)
        nextSampleMicros = now + (1000000UL / RAW_SAMPLE_RATE_HZ);

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    preprocessSample(a, g);
    rawSamplesCollected++;
    if (!filterReady)
        filterReady = true;
    if (rawSamplesCollected < RAW_WINDOW_SIZE)
        return;

    runInference();
    slideWindow();
}
