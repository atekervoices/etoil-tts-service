import os
import sys
import torch
import numpy as np
import re
import json
import asyncio
import time
import librosa
import tempfile
import soundfile as sf
from typing import List, Optional
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from vllm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub import snapshot_download

# Configuration
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
MODEL_NAME = os.environ.get("MODEL_NAME", "crestai/spark-tts-nexvox_v8")
TOKENIZER_REPO = os.environ.get("TOKENIZER_REPO", "unsloth/Spark-TTS-0.5B")
TOKENIZER_CACHE_DIR = os.environ.get("TOKENIZER_CACHE_DIR", "Spark-TTS-0.5B")
SPARK_TTS_REPO_PATH = os.environ.get("SPARK_TTS_REPO_PATH", "Spark-TTS")

# Set NCCL environment variables for multi-GPU communication
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_NET_GDR_LEVEL"] = "0"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_TREE_THRESHOLD"] = "0"
os.environ["NCCL_RING_THRESHOLD"] = "8388608"

# Audio configuration
AUDIO_SAMPLERATE = 16000
AUDIO_BITS_PER_SAMPLE = 16
AUDIO_CHANNELS = 1

# Default parameters
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048
DEFAULT_SPEAKER_ID = 243  # Runyankore female

# Speaker IDs mapping
SPEAKER_IDS = {
    "acholi_female": 241,
    "ateso_female": 242,
    "runyankore_female": 243,
    "lugbara_female": 245,
    "swahili_male": 246,
    "luganda_female": 248,
}

# Precomputed global tokens for each speaker
GLOBAL_IDS_BY_SPEAKER = {
    'kin_female_1': [3395, 2293, 1918, 4078, 3608, 1978, 384, 2717, 3663, 2587,
                  1850, 3174, 464, 3616, 3314, 1884, 2679, 293, 3770, 1407, 726,
                  1077, 4089, 3702, 3242, 1996, 3566, 320, 2337, 1485, 797,
                  1612],
    'nyn_female_248': [2631, 753, 3640, 4079, 1623, 3293, 389, 2636, 3217, 2161,
                    1649, 3634, 501, 3360, 3131, 4093, 1279, 1335, 1159, 253,
                    2414, 1424, 4021, 3194, 2767, 926, 3269, 1677, 3509, 3227,
                    3998, 3928],
    'lgg_female_245': [3819, 2546, 120, 4057, 630, 2406, 3284, 432, 3618, 2870, 37,
                    3377, 1016, 3380, 3191, 2993, 252, 3152, 495, 1785, 3500,
                    1252, 2298, 759, 1465, 1880, 3553, 216, 3828, 1725, 2275,
                    1541],
    'teo_female_241': [1355, 2785, 500, 3547, 1057, 1532, 18, 3460, 1370, 3007, 79,
                    1650, 215, 3288, 3190, 941, 244, 3476, 1062, 762, 2415,
                    1718, 1007, 881, 2664, 3974, 3313, 1550, 3110, 1099, 1431,
                    841],
    'ach_female_242': [2691, 1265, 1592, 4078, 3253, 2541, 2274, 2435, 3142, 2722,
                    1591, 3362, 741, 3608, 2293, 340, 124, 1057, 1004, 1145,
                    459, 1400, 4090, 3954, 1942, 1725, 3304, 1856, 3094, 457,
                    2713, 2377],
    'swa_male_246': [2711, 1205, 633, 3566, 3658, 675, 3347, 1678, 2644, 2302, 761,
                  3632, 496, 3848, 3190, 1364, 2167, 2298, 3481, 3576, 3302,
                  2931, 3252, 2386, 2810, 31, 3302, 613, 1220, 3531, 2973,
                  2342],
    'swa_female_1': [3395, 3834, 1659, 3821, 3418, 2195, 67, 3070, 2762, 2307,
                  4007, 3538, 1022, 3620, 2284, 2025, 3511, 865, 2476, 2172,
                  2410, 1287, 3767, 3376, 2894, 2974, 3278, 1867, 1587, 2139,
                  1630, 4068],
   'swa_male_3': [2327, 1202, 1657, 4072, 2681, 958, 1113, 2957, 3337, 2891, 1898,
                3156, 741, 3616, 3314, 2393, 251, 932, 885, 757, 726, 353, 3833,
                3382, 3764, 2577, 3569, 3820, 1066, 1735, 1462, 2052],
    'fat_male_1': [1307, 309, 3644, 4031, 3656, 2026, 2146, 910, 3494, 2422, 560,
                3893, 630, 3388, 1255, 293, 252, 1377, 2849, 566, 838, 1424,
                4085, 3120, 488, 1805, 3828, 1673, 3527, 961, 1880, 1281],
    'fat_female_2': [2630, 1272, 2876, 3822, 2422, 1002, 2181, 1613, 3144, 1411,
                  1973, 3353, 740, 3624, 2293, 1696, 2230, 1454, 1381, 1277, 5,
                  2212, 4089, 3455, 3045, 317, 3304, 1736, 3558, 450, 3032,
                  2312],
    'fat_male_3': [2323, 546, 3452, 4030, 1297, 3052, 1441, 540, 3158, 1785, 2614,
                3698, 725, 3624, 2171, 421, 253, 1389, 1573, 314, 1861, 2145,
                4081, 3122, 189, 1801, 3321, 2635, 2531, 898, 1821, 1817],
    'fat_female_4': [3339, 226, 1596, 4045, 170, 1768, 80, 1289, 3398, 2900, 930,
                  3354, 724, 3620, 2229, 1108, 374, 1322, 849, 247, 524, 2466,
                  4069, 3515, 2726, 1476, 3529, 844, 3506, 2271, 2904, 1305],
    'hau_male_8': [539, 1087, 568, 3768, 1644, 1508, 1071, 3146, 2050, 3779, 2981,
                2331, 481, 3612, 3238, 467, 183, 760, 1170, 316, 4008, 442,
                1019, 122, 3704, 1327, 3572, 2459, 1337, 1472, 2227, 875],
    'hau_male_6': [2819, 1073, 1652, 4023, 1606, 1021, 2175, 3338, 3522, 3778,
                3050, 3659, 471, 3372, 1265, 2598, 251, 820, 1300, 58, 1666,
                3696, 1006, 1342, 662, 1835, 3316, 704, 1037, 1666, 2535, 920],
    'hau_female_1': [342, 2559, 629, 4089, 886, 1016, 1293, 1628, 2048, 1735, 2044,
                  3400, 213, 3357, 1200, 2461, 2427, 1017, 2264, 3581, 646, 742,
                  1646, 1086, 1467, 799, 3320, 3230, 301, 450, 1790, 3408],
    'hau_female_3': [913, 1266, 828, 4057, 3559, 2268, 511, 3138, 2371, 3458, 938,
                  3397, 244, 3441, 2226, 3542, 1149, 809, 1369, 2174, 326, 2488,
                  1005, 383, 986, 2271, 3288, 1608, 2475, 2122, 2489, 662],
    'hau_male_4': [1603, 1076, 829, 4022, 2881, 494, 2154, 2118, 3394, 2690, 3044,
                2631, 466, 3368, 1208, 2745, 191, 881, 1, 317, 1955, 4020, 1023,
                1341, 1686, 558, 3316, 896, 3225, 2883, 2987, 601],
    'hau_male_2': [1623, 3129, 317, 3641, 545, 1786, 3451, 2185, 3395, 3267, 3796,
                2379, 618, 3372, 3192, 1686, 127, 889, 118, 1341, 1865, 2489,
                1003, 2426, 1561, 3375, 3192, 2446, 1159, 898, 3254, 601],
    'hau_female_7': [546, 248, 376, 4067, 3025, 1530, 3558, 1729, 2625, 2390, 2040,
                  3362, 240, 3612, 3312, 1614, 1215, 2841, 3662, 3004, 323,
                  2008, 1514, 310, 1514, 2557, 3568, 2957, 1234, 1922, 2957,
                  1809],
    'hau_female_5': [350, 1273, 1596, 4058, 1379, 1238, 2398, 2049, 3651, 2691,
                  1001, 3585, 229, 3380, 1190, 1754, 1082, 790, 2313, 2173,
                  1310, 2209, 1001, 378, 3019, 1788, 3273, 1608, 3515, 1303,
                  1701, 1314],
    'ibo_female_1': [2583, 2161, 568, 3823, 2112, 448, 2150, 3394, 2053, 1667,
                  1973, 3610, 730, 3132, 3127, 2944, 499, 1958, 724, 1264, 417,
                  249, 935, 1593, 3737, 1845, 3540, 1740, 570, 1692, 2479,
                  1045],
    'ibo_female_3': [1710, 2290, 564, 3049, 638, 3542, 2141, 1349, 66, 3778, 3048,
                  1672, 746, 3616, 3250, 2653, 1151, 937, 2460, 2236, 1618,
                  1687, 358, 59, 2590, 3702, 3276, 3916, 638, 715, 3454, 2576],
    'ibo_male_6': [1515, 2545, 569, 3811, 1344, 1420, 2207, 1762, 2050, 2694, 2027,
                2582, 423, 3368, 3191, 3224, 247, 2644, 273, 1264, 644, 2805,
                403, 314, 1833, 2869, 3280, 652, 2221, 1996, 491, 2582],
    'ibo_male_4': [2887, 3128, 824, 3362, 3877, 3560, 1118, 2051, 2058, 1922, 2020,
                1351, 352, 3620, 3127, 3396, 2234, 806, 2433, 1084, 4022, 3641,
                746, 1594, 1524, 2333, 3172, 1621, 2169, 1921, 1067, 1191],
    'ibo_male_8': [1879, 2102, 564, 3894, 1672, 2813, 47, 2055, 2114, 4034, 3049,
                653, 570, 3372, 3242, 3611, 123, 808, 133, 1077, 3978, 3873,
                687, 1337, 1972, 3871, 3124, 2710, 141, 961, 3195, 583],
    'ibo_female_7': [2603, 2289, 2481, 3799, 1341, 718, 2414, 2369, 2178, 3786,
                  3050, 3607, 727, 3608, 3126, 3002, 255, 877, 1350, 229, 2270,
                  3524, 2042, 1343, 267, 1871, 3264, 2690, 2124, 206, 3799,
                  2984],
    'ibo_female_5': [2644, 3317, 568, 3832, 3628, 2001, 93, 3522, 2307, 3523, 992,
                  2582, 470, 3384, 3254, 3036, 2226, 1897, 3722, 2296, 1430,
                  2043, 1016, 894, 1611, 1853, 3540, 1672, 314, 1692, 2477,
                  2128],
    'ibo_male_2': [2691, 2354, 820, 3378, 1929, 2812, 63, 1095, 1346, 3014, 3049,
                1417, 634, 3628, 2219, 3611, 123, 816, 138, 1078, 3722, 3877,
                735, 313, 953, 3887, 3124, 1686, 78, 1985, 3451, 646],
    'kik_female_2': [3107, 2293, 829, 4060, 2470, 2009, 97, 68, 3722, 3443, 615,
                  3441, 741, 3616, 2279, 2644, 2175, 1321, 166, 382, 1115, 1077,
                  4085, 3834, 2951, 2780, 3546, 2885, 3490, 1351, 3860, 2125],
    'kik_male_4': [3650, 3126, 828, 3886, 2555, 4092, 230, 2505, 3659, 3667, 811,
                3173, 160, 3376, 2279, 2960, 255, 357, 369, 1085, 1675, 1717,
                4072, 3953, 1638, 2092, 3325, 904, 2082, 2439, 600, 2892],
    'kik_male_1': [3930, 3637, 825, 3885, 3175, 4026, 1124, 217, 3596, 2882, 784,
                3177, 241, 3388, 3191, 1864, 1343, 1045, 1185, 3388, 87, 2876,
                3989, 3958, 2805, 2861, 3708, 347, 2161, 467, 2826, 1612],
    'kik_female_3': [2198, 2290, 829, 4076, 1723, 2809, 33, 1176, 3661, 3602, 875,
                  3429, 1017, 3600, 3310, 2568, 1151, 1301, 486, 382, 1095,
                  1077, 4085, 3826, 3990, 3564, 3565, 1801, 3314, 1419, 2824,
                  1100],
    'lug_female_4': [1128, 3312, 1599, 3044, 3039, 1675, 2150, 1769, 3911, 3843,
                  2966, 3732, 757, 3684, 3305, 1629, 1147, 257, 2933, 1593, 462,
                  2587, 3809, 3616, 581, 3764, 3545, 1924, 586, 1871, 1012,
                  3408],
    'lug_male_3': [3263, 3704, 2878, 4094, 3124, 2983, 3093, 501, 3973, 3923, 1899,
                3314, 917, 3372, 3247, 587, 3507, 3926, 633, 3957, 329, 2821,
                3815, 3504, 866, 1854, 4086, 1927, 537, 321, 2645, 1417],
    'lug_female_5': [34, 3312, 1913, 3817, 2465, 966, 2219, 2947, 3723, 3715, 1831,
                  3162, 486, 3124, 3318, 409, 1271, 792, 1916, 2272, 705, 2661,
                  2814, 3632, 852, 3440, 3528, 968, 555, 1612, 2554, 2052],
    'lug_male_2': [191, 2098, 1580, 3901, 3097, 263, 3072, 721, 3719, 3587, 1893,
                3237, 258, 3604, 3194, 2004, 59, 389, 1147, 2341, 412, 1347,
                2551, 1400, 615, 1067, 3462, 2891, 198, 1675, 1896, 1114],
    'lug_female_8': [2323, 2805, 1919, 4058, 3412, 1421, 89, 1887, 3142, 2375,
                  2987, 3298, 984, 3628, 3317, 741, 3443, 853, 553, 1460, 1428,
                  66, 3831, 3122, 3482, 1996, 3795, 3781, 2162, 80, 458, 3621],
    'lug_male_1': [1387, 2098, 1597, 3901, 3712, 1370, 3106, 709, 3778, 1347, 2914,
                3476, 518, 3624, 2218, 1954, 63, 400, 2107, 2612, 1945, 1686,
                3831, 1328, 602, 2079, 3240, 1927, 1218, 1601, 1704, 2157],
    'lug_male_6': [234, 2100, 2423, 3897, 2192, 3678, 2315, 823, 3468, 3399, 4007,
                3709, 519, 3868, 1134, 2299, 187, 720, 2360, 2100, 1692, 2470,
                4075, 1342, 680, 1567, 3837, 1607, 968, 640, 3702, 3435],
    'lug_female_7': [1186, 3312, 831, 4076, 2986, 1755, 2203, 937, 3974, 2883,
                  2967, 3735, 757, 3600, 3317, 1629, 2235, 1553, 1913, 2681,
                  714, 2406, 3582, 2608, 852, 2736, 3820, 1856, 863, 1871, 1912,
                  3408],
    'luo_male_3': [3715, 3105, 894, 4025, 3555, 2012, 1238, 3210, 3663, 2827, 1835,
                3478, 293, 3376, 3175, 2976, 1279, 673, 1634, 1663, 491, 1205,
                4026, 3702, 951, 2316, 3302, 660, 2357, 1615, 372, 1673],
    'luo_male_4': [3255, 3375, 703, 3770, 3460, 969, 3141, 1211, 3402, 3703, 3638,
                3387, 949, 3384, 3309, 210, 1279, 2736, 441, 30, 485, 2194,
                3750, 3259, 113, 2830, 3530, 903, 1106, 1920, 1968, 1371],
    'luo_female_3': [3715, 2251, 1211, 3814, 3153, 667, 3137, 1501, 3161, 3627,
                  3883, 3386, 710, 3376, 2218, 740, 2495, 3941, 120, 1278, 835,
                  595, 3510, 3191, 1970, 3021, 3797, 809, 1139, 2450, 2880,
                  1364],
    'luo_female_4': [3367, 2793, 1405, 3834, 3640, 1999, 230, 815, 3238, 3627,
                  3355, 3635, 743, 3604, 2295, 1365, 1279, 2865, 2362, 186, 520,
                  3360, 3774, 3379, 801, 1996, 3819, 1607, 2354, 68, 914,
                  3713],
    'luo_male_1': [3863, 1338, 573, 3614, 1686, 2812, 1249, 1386, 3721, 3925, 795,
                3174, 613, 3608, 2278, 2404, 127, 1073, 1569, 1086, 1061, 3382,
                4009, 4021, 2403, 1852, 3386, 649, 3376, 87, 920, 2893],
    'luo_female_1': [3223, 2243, 1471, 3558, 2133, 408, 1120, 2069, 2118, 2937,
                  1963, 3626, 451, 3616, 2227, 1000, 1211, 3416, 1147, 507,
                  1451, 1137, 4023, 1399, 945, 2502, 3530, 801, 2401, 2062, 800,
                  1433],
    'luo_female_2': [3735, 1078, 701, 4009, 3478, 3004, 1267, 2185, 3914, 2315,
                  1956, 3435, 160, 3604, 2281, 2662, 511, 678, 1891, 1342, 83,
                  3701, 4025, 3002, 1201, 1566, 3578, 716, 1136, 323, 998,
                  1869],
    'luo_male_2': [2833, 3105, 957, 4089, 3258, 2956, 1234, 3226, 3727, 3595, 1847,
                3734, 277, 3440, 3179, 2032, 1535, 468, 629, 1407, 1499, 1204,
                3833, 3957, 887, 3336, 3285, 916, 3621, 2703, 1140, 1929],
    'nyn_male_7': [3683, 3186, 2431, 4053, 1661, 863, 1161, 410, 3394, 3842, 3906,
                3488, 2021, 3616, 3226, 2661, 1211, 2725, 1906, 1396, 1670,
                1573, 3811, 3617, 949, 3362, 3812, 3993, 303, 2635, 3705,
                3108],
    'nyn_female_5': [1127, 3569, 1582, 4092, 3370, 2947, 1044, 742, 3983, 3907,
                  1955, 3552, 952, 3384, 2206, 989, 2483, 786, 3809, 2593, 348,
                  263, 4066, 3184, 1927, 1908, 3786, 3009, 886, 1821, 1637,
                  2121],
    'nyn_male_6': [2935, 3121, 1598, 3902, 2215, 1749, 1094, 1489, 3970, 3970,
                3975, 3541, 1862, 3604, 3162, 2916, 2239, 677, 2681, 3644, 1801,
                1403, 2523, 1576, 902, 3387, 3244, 2881, 799, 778, 2724, 2396],
    'nyn_male_8': [2870, 3122, 575, 3876, 3475, 1911, 2262, 1474, 3906, 2562, 1937,
                3817, 322, 3368, 3243, 2917, 1151, 550, 2417, 3642, 1558, 314,
                3814, 1460, 167, 3355, 3957, 3971, 1182, 2563, 2925, 1160],
    'nyn_female_4': [2199, 3313, 1595, 4068, 2729, 3978, 1125, 475, 3971, 3907,
                  3970, 3796, 757, 3380, 2281, 366, 1215, 529, 2041, 2929, 409,
                  566, 3538, 1073, 934, 2721, 3802, 1857, 416, 525, 3508,
                  3154],
    'nyn_female_2': [2737, 2482, 1338, 3834, 3007, 1743, 63, 1418, 3907, 3843,
                  2855, 3745, 2014, 3381, 3299, 2666, 1275, 549, 2678, 371, 650,
                  2582, 3824, 2340, 916, 1827, 3558, 3968, 287, 2319, 3701,
                  2260],
    'nyn_female_1': [1199, 3568, 1855, 4093, 3241, 3907, 2129, 425, 4042, 3843,
                  1874, 3797, 676, 3368, 2206, 985, 2167, 769, 2937, 3360, 348,
                  1543, 4066, 2416, 899, 2993, 3790, 966, 946, 1822, 2661,
                  3588],
    'nyn_male_3': [2786, 1073, 1342, 3902, 3415, 1391, 2094, 1449, 3843, 3843,
                3867, 3792, 519, 3848, 3223, 2264, 1087, 657, 2357, 3380, 330,
                1771, 2795, 2425, 518, 1331, 3834, 2885, 1371, 1283, 2904,
                2270],
    'twi_male_3': [3083, 2171, 636, 3569, 380, 649, 2199, 1306, 3141, 2699, 1983,
                3160, 989, 3620, 3190, 1791, 252, 1533, 3345, 250, 3788, 2481,
                3965, 3518, 564, 1549, 3288, 358, 729, 457, 1715, 3337],
    'twi_female_2': [3142, 194, 828, 3785, 1972, 1940, 1413, 1027, 3078, 3724,
                  1001, 3109, 730, 3624, 3238, 1677, 309, 1963, 2763, 250, 94,
                  3424, 4095, 3775, 2812, 2249, 3524, 361, 3509, 216, 1924,
                  2313],
    'twi_female_4': [3755, 3313, 1596, 4089, 2154, 906, 2113, 1158, 3235, 3762,
                  435, 3897, 720, 3616, 1405, 1918, 180, 3172, 1618, 3558, 1102,
                  2355, 4083, 2609, 1791, 1360, 3532, 878, 2552, 2447, 1820,
                  2570],
    'twi_male_1': [2262, 1128, 2940, 3561, 353, 3933, 2480, 14, 2117, 2540, 2031,
                3369, 710, 3848, 1132, 2090, 252, 2220, 1657, 1278, 1961, 3367,
                4017, 2663, 118, 2061, 3293, 1659, 1239, 1922, 2685, 1545],
    'ach_male_4': [3006, 3377, 571, 3878, 1459, 1606, 3142, 1041, 3722, 3649, 2978,
                4058, 513, 3840, 3131, 1960, 1083, 1557, 1462, 3383, 586, 602,
                2811, 3504, 939, 2850, 3608, 3970, 430, 1542, 2324, 1374],
    'ach_female_6': [3519, 3571, 1855, 4073, 946, 902, 2139, 1694, 3714, 3842,
                  4067, 3540, 742, 3380, 3306, 1305, 3191, 536, 2917, 2366, 589,
                  2326, 3825, 3121, 801, 3992, 3784, 1857, 683, 841, 1012,
                  2128],
    'ach_female_2': [2491, 3314, 2623, 4068, 1698, 1959, 3178, 648, 3969, 1794,
                  4048, 3796, 673, 3604, 1259, 1292, 2303, 1305, 3705, 3768,
                  328, 2347, 3809, 3124, 609, 2874, 3576, 2944, 106, 840, 868,
                  3136],
    'ach_male_1': [2410, 2098, 571, 3884, 3752, 2745, 3291, 596, 3651, 2882, 1936,
                3734, 517, 3624, 2155, 2980, 187, 1698, 849, 2353, 648, 2990,
                3031, 2681, 441, 3602, 3512, 3777, 1470, 710, 2198, 1436],
    'ach_male_5': [2071, 2081, 1919, 3901, 3241, 1703, 3255, 584, 3974, 2563, 3929,
                3558, 548, 3380, 3240, 851, 3511, 933, 566, 1594, 551, 1875,
                3555, 2101, 2600, 2564, 3769, 1929, 1139, 1803, 2004, 2592],
    'ach_male_3': [2487, 2098, 2623, 3848, 2910, 3587, 1353, 534, 3395, 2817, 3986,
                3496, 534, 3608, 2214, 2120, 1079, 1316, 564, 3616, 972, 2831,
                4081, 2584, 507, 3618, 3916, 4034, 751, 713, 2349, 3162],
    'ach_female_7': [3243, 3569, 1339, 4085, 3729, 970, 3082, 1417, 3970, 3907,
                  4003, 3732, 998, 3616, 3238, 701, 1211, 1572, 2922, 1654, 773,
                  1558, 4087, 2104, 886, 3985, 4052, 1857, 942, 841, 4084,
                  1168],
    'ach_female_8': [2155, 3313, 574, 4068, 2746, 1674, 1071, 2691, 3971, 3842,
                  4007, 3731, 2027, 3384, 3302, 1546, 2167, 1818, 2984, 566,
                  661, 3353, 3829, 3376, 1808, 3749, 3801, 2945, 543, 836, 2802,
                  1045],
    'swa_female_1': [2134, 3583, 1919, 4086, 3748, 735, 3145, 733, 3459, 3843,
                  4075, 3554, 1018, 3360, 2216, 1902, 3575, 854, 2643, 1727,
                  519, 775, 3558, 3254, 1901],
    'swa_male_3': [1554, 2161, 372, 4088, 1575, 958, 21, 2894, 3402, 3851, 1898,
                3145, 997, 3364, 3314, 1565, 251, 868, 885, 499, 710, 1314,
                4089, 3381, 2544, 2833, 3569, 3800, 1049, 1411, 2490, 1288],
    'swa_female_6': [3459, 3816, 2923, 4095, 882, 2639, 1169, 1835, 3523, 3330,
                  3993, 3538, 988, 3620, 3255, 1974, 3127, 852, 381, 2316, 344,
                  2569, 3511, 3123, 468, 1996, 3785, 704, 1761, 1540, 1443,
                  3930],
    'swa_male_8': [2223, 3123, 1467, 4060, 3384, 3674, 155, 1708, 2625, 2819, 3923,
                3473, 321, 3616, 2150, 1721, 1279, 720, 1833, 2357, 713, 3759,
                2786, 1912, 550, 2579, 3550, 2633, 1401, 590, 757, 3173],
    'swa_female_2': [3595, 4087, 1659, 4014, 3482, 2199, 71, 3007, 2755, 2627,
                  4051, 3794, 1023, 3600, 3304, 2987, 3451, 856, 2730, 1084,
                  2662, 1563, 3811, 2165, 3661, 1950, 3278, 1867, 1595, 2379,
                  927, 2772],
    'swa_female_5': [3162, 3827, 1919, 4057, 2408, 71, 1059, 767, 3463, 2371, 3985,
                  3554, 698, 3620, 3256, 922, 3507, 852, 2705, 2212, 1437, 1635,
                  3299, 3440, 3416, 1978, 3787, 969, 177, 1437, 731, 2375],
    'swa_male_4': [2134, 3121, 3967, 3956, 2649, 1539, 3143, 415, 3736, 2322, 1872,
                3569, 336, 3448, 2157, 1781, 3507, 2705, 1396, 2609, 408, 601,
                3555, 3248, 689, 1537, 4054, 1893, 503, 1353, 2916, 2402],
    'yor_female_1': [1591, 2296, 56, 4090, 2420, 978, 1137, 2386, 3459, 3283, 3056,
                  3367, 742, 3376, 3282, 3976, 1335, 1830, 3090, 1919, 550, 51,
                  2814, 1210, 2694, 3036, 3285, 3584, 1590, 1693, 377, 421],
    'yor_female_7': [3739, 3313, 564, 4077, 1283, 1533, 1053, 2385, 3843, 2883,
                  3063, 3712, 501, 3384, 2118, 2961, 1081, 865, 98, 2686, 4070,
                  570, 1655, 1326, 1923, 2686, 3324, 832, 1114, 1543, 2493,
                  2484],
    'yor_male_2': [1895, 1076, 824, 3455, 792, 977, 1087, 3158, 3330, 3011, 2010,
                2374, 614, 3612, 3191, 2633, 1138, 946, 375, 3124, 3733, 2661,
                479, 1326, 2951, 3135, 3316, 1475, 1103, 1921, 2651, 1129],
    'yor_male_6': [1826, 2108, 372, 3815, 1045, 490, 2093, 323, 2114, 3527, 3002,
                1305, 729, 3628, 3171, 1626, 1267, 997, 1110, 2234, 1686, 2530,
                751, 57, 1824, 557, 3312, 840, 1100, 2369, 1131, 854],
    'yor_male_4': [3923, 3120, 1592, 3389, 3629, 2018, 2143, 3147, 2631, 3971,
                2017, 2389, 272, 3360, 3191, 3656, 3130, 1958, 1677, 2364, 3727,
                1954, 1018, 1635, 1969, 3128, 3428, 1472, 1451, 1863, 2481,
                149],
    'yor_male_8': [3667, 1073, 572, 3640, 2841, 2810, 2103, 263, 2311, 1987, 2029,
                2377, 530, 3608, 3303, 2395, 58, 801, 2050, 1593, 2506, 1383,
                937, 630, 1204, 3884, 3381, 3285, 1254, 387, 3155, 351],
    'yor_female_3': [714, 2273, 824, 4071, 2103, 728, 60, 3141, 1026, 4035, 2030,
                  1612, 470, 3380, 3185, 1945, 59, 996, 268, 1274, 730, 1457,
                  862, 575, 929, 1738, 3272, 1608, 1390, 139, 442, 1664],
    'yor_female_5': [1383, 1265, 2364, 4086, 209, 1485, 2085, 3423, 3098, 3411,
                  3055, 3381, 986, 3380, 3302, 2186, 2806, 2923, 2568, 1241,
                  613, 602, 4073, 1075, 2869, 620, 3778, 2572, 566, 932, 1636,
                  2261],
    'wol_male_1': [2087, 3538, 1016, 3282, 1094, 2382, 3158, 2068, 1153, 2316,
                2026, 3162, 747, 3600, 3130, 1653, 4091, 1530, 2293, 3291, 2521,
                3097, 3062, 1075, 1129, 3854, 3275, 330, 3141, 1610, 1458,
                2674],
    'wol_female_1': [1775, 1264, 1376, 4042, 1209, 2439, 1122, 1116, 3457, 1154,
                  1399, 3612, 759, 3840, 3251, 743, 1399, 2101, 2140, 3146, 686,
                  1091, 2805, 2623, 1866, 4041, 3275, 2697, 3536, 1357, 2642,
                  3959],
    'pcm_male_1': [1575, 1086, 1700, 3711, 2905, 1417, 2164, 1614, 195, 1131, 2495,
                3112, 565, 3840, 3314, 1302, 3259, 2302, 1640, 3101, 2778, 454,
                1276, 2239, 2490, 1037, 3275, 2699, 2467, 3969, 569, 3955],
    'pcm_female_2': [3259, 2254, 1432, 3527, 1934, 1167, 105, 3493, 1930, 2694,
                  953, 3141, 487, 3376, 3250, 3671, 3135, 3446, 1558, 2703,
                  3538, 333, 500, 383, 2950, 1229, 3279, 2114, 2627, 3595, 3858,
                  1778],
    'pcm_female_4': [2419, 1520, 381, 3829, 1918, 135, 1058, 1418, 1222, 1303,
                  2347, 3110, 635, 3600, 3314, 2561, 1523, 3185, 2744, 1506,
                  137, 1114, 1256, 2418, 933, 1840, 3523, 901, 418, 2190, 1908,
                  3634],
    'pcm_female_5': [2323, 2256, 1524, 4074, 1443, 958, 1033, 1220, 1220, 3019,
                  2927, 3161, 466, 3620, 3312, 1881, 382, 3812, 2133, 3069,
                  3011, 534, 2047, 305, 695, 2958, 3568, 2655, 459, 1987, 4009,
                  516],
    'pcm_female_3': [3607, 245, 564, 4078, 2015, 2742, 3173, 2438, 1922, 2723,
                  1509, 3153, 437, 2828, 3251, 1898, 1143, 2144, 2206, 3768,
                  3224, 1726, 747, 801, 3787, 1854, 3326, 1478, 2388, 3215,
                  2715, 2855],
    'pcm_male_4': [1907, 1072, 569, 3902, 2862, 2471, 2113, 431, 1481, 1779, 1364,
                3156, 809, 3640, 3314, 1677, 51, 3168, 3676, 2618, 1669, 1367,
                1007, 1968, 1725, 3332, 3106, 3470, 2721, 2507, 3453, 2960],
    'pcm_female_1': [59, 464, 2388, 4095, 1832, 3085, 20, 2369, 2185, 2968, 1885,
                  3110, 745, 3612, 3314, 1165, 115, 2338, 324, 1681, 4, 3596,
                  3045, 1588, 1465, 3023, 3782, 4079, 1208, 452, 2727, 3598],
    'pcm_male_2': [2151, 1312, 496, 3721, 2382, 1643, 69, 21, 2888, 539, 612, 3455,
                33, 2364, 3189, 3713, 499, 3505, 390, 1074, 2388, 2578, 743,
                2872, 2453, 1801, 3571, 2507, 2448, 68, 2459, 3877],
    'pcm_male_3': [2099, 2364, 1646, 4089, 521, 748, 1054, 2948, 3464, 1923, 2999,
                3142, 242, 3372, 3252, 2682, 2295, 759, 1860, 1790, 1490, 10,
                1016, 305, 2849, 1839, 3826, 3533, 1039, 1539, 3833, 1425],
    'pcm_male_5': [3867, 3133, 629, 4074, 2522, 2990, 1146, 3981, 3846, 2051, 1718,
                3193, 55, 3381, 3251, 1319, 2555, 176, 2789, 2431, 3793, 378,
                3494, 3639, 2511, 1341, 3311, 1218, 3398, 2950, 635, 3687]
}

app = FastAPI()

# Global model variables
vllm_model = None
audio_tokenizer = None
device = None


class AudioRequest(BaseModel):
    text: str
    voice: str = "runyankore_female"
    speaker_id: Optional[int] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


class VoiceCloningRequest(BaseModel):
    text: str
    reference_audio_path: str
    reference_text: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def chunk_text(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks based on sentence boundaries.
    
    This approach preserves natural sentence flow and intonation for TTS.
    
    Args:
        text: The input string to chunk
        max_chunk_size: Maximum character length per chunk (soft limit)
    
    Returns:
        List of text chunks, each containing one or more complete sentences
    """
    # Split on sentence-ending punctuation (. ! ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_length = len(sentence)
        
        # Start new chunk if adding this sentence would exceed limit
        if current_chunk and (current_length + sentence_length + 1) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1
    
    # Add the final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def chunk_text_simple(text: str) -> List[str]:
    """
    Split text into individual sentences.
    
    Recommended for TTS - provides maximum control with one sentence per chunk.
    
    Args:
        text: The input string to chunk
    
    Returns:
        List of individual sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_with_count(text: str, sentences_per_chunk: int = 3) -> List[str]:
    """
    Split text into chunks containing a specific number of sentences.
    
    Args:
        text: The input string to chunk
        sentences_per_chunk: Number of sentences to include in each chunk
    
    Returns:
        List of text chunks
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks: List[str] = []
    
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    
    return chunks


def extract_speaker_from_reference(
    audio_path: str,
    audio_tokenizer,
    reference_text: str = None,
    device="cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract global and semantic tokens from a reference audio file.
    Returns: (global_ids, semantic_ids)
    """
    # Load audio and resample to 16kHz if needed
    wav, sr = sf.read(audio_path)
    
    # Resample if not 16kHz
    if sr != 16000:
        print(f"Resampling audio from {sr}Hz to 16000Hz...")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        
        # Save resampled audio to a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)  # Close the file descriptor
        sf.write(temp_path, wav, 16000)
        audio_path_to_use = temp_path
    else:
        audio_path_to_use = audio_path
    
    try:
        # Tokenize reference audio using the file path
        global_ids, semantic_ids = audio_tokenizer.tokenize(audio_path_to_use)
        
        # Convert to tensors if they aren't already
        if not isinstance(global_ids, torch.Tensor):
            global_ids = torch.tensor(global_ids).long()
        if not isinstance(semantic_ids, torch.Tensor):
            semantic_ids = torch.tensor(semantic_ids).long()
        
        # Ensure they're 1D tensors
        if global_ids.dim() > 1:
            global_ids = global_ids.squeeze()
        if semantic_ids.dim() > 1:
            semantic_ids = semantic_ids.squeeze()
            
        return global_ids, semantic_ids
        
    finally:
        # Clean up temporary file if we created one
        if sr != 16000 and os.path.exists(temp_path):
            os.unlink(temp_path)


def text_to_speech_cloned(
    text: str,
    audio_tokenizer,
    model,
    reference_audio_path: str,
    reference_text: str = None,
    temperature: float = 0.7,
    device="cuda"
):
    '''Create a wav array using zero-shot voice cloning from reference audio.'''
    try:
        print(f"Starting voice cloning for text: '{text[:50]}...'")
        print(f"Reference audio path: {reference_audio_path}")
        
        texts = chunk_text_simple(text)
        texts = [t.strip() for t in texts if len(t.strip()) > 0]
        print(f"Text split into {len(texts)} chunks: {texts}")
        
        if not texts:
            raise ValueError("No valid text chunks found after processing")
        
        sampling_params = SamplingParams(temperature=temperature, max_tokens=2048)
        
        # 1. Extract speaker identity from reference
        print("Extracting speaker features from reference audio...")
        try:
            global_ids_ref, semantic_ids_ref = extract_speaker_from_reference(
                reference_audio_path, audio_tokenizer, reference_text, device
            )
            print(f"Successfully extracted speaker features")
            print(f"Global IDs shape: {global_ids_ref.shape}, Semantic IDs shape: {semantic_ids_ref.shape}")
        except Exception as e:
            raise ValueError(f"Failed to extract speaker features from reference audio: {str(e)}")
        
        # Convert to list for prompt formatting
        global_ids_list = global_ids_ref.cpu().tolist()
        if isinstance(global_ids_list, int):
            global_ids_list = [global_ids_list]
        
        print(f"Extracted {len(global_ids_list)} global tokens from reference")
        
        prompts = []
        for i, chunk in enumerate(texts):
            # Build prompt with reference global tokens
            prompt = f"<|task_tts|><|start_content|>{chunk}<|end_content|><|start_global_token|>"
            prompt += ''.join([f'<|bicodec_global_{t}|>' for t in global_ids_list]) 
            prompt += '<|end_global_token|><|start_semantic_token|>'
            prompts.append(prompt)
            print(f"Generated prompt {i+1}/{len(texts)}: {prompt[:100]}...")
        
        print("Generating speech with model...")
        try:
            outputs = model.generate(
                prompts=prompts,
                sampling_params=sampling_params
            )
            print(f"Model generation completed. Generated {len(outputs)} outputs")
        except Exception as e:
            raise ValueError(f"Model generation failed: {str(e)}")
        
        speech_segments = []
        
        for i, output in enumerate(outputs):
            print(f"Processing output {i+1}/{len(outputs)}")
            predicted_tokens = output.outputs[0].text
            print(f"Raw model output: {predicted_tokens[:200]}...")
            
            semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_tokens)
            print(f"Found {len(semantic_matches)} semantic tokens")
            
            if not semantic_matches:
                raise ValueError(f"No semantic tokens found in output {i+1}. Raw output: {predicted_tokens}")
            
            try:
                pred_semantic_ids = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0)
                pred_global_ids = torch.tensor([global_ids_list]).long()
                
                print(f"Detokenizing audio: semantic shape={pred_semantic_ids.shape}, global shape={pred_global_ids.shape}")
                
                wav_np = audio_tokenizer.detokenize(
                    pred_global_ids.to(device),
                    pred_semantic_ids.to(device)
                )
                
                print(f"Generated audio segment {i+1}: shape={wav_np.shape}")
                speech_segments.append(wav_np)
                
            except Exception as e:
                raise ValueError(f"Audio detokenization failed for segment {i+1}: {str(e)}")
        
        if not speech_segments:
            raise ValueError("No speech segments were generated")
        
        result_wav = np.concatenate(speech_segments)
        print(f"Successfully concatenated {len(speech_segments)} segments. Final shape: {result_wav.shape}")
        
        return result_wav
        
    except Exception as e:
        print(f"ERROR in text_to_speech_cloned: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def initialize_models():
    """Initialize vLLM model and audio tokenizer."""
    global vllm_model, audio_tokenizer, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Add Spark-TTS to path if exists
    if os.path.exists(SPARK_TTS_REPO_PATH):
        sys.path.append(SPARK_TTS_REPO_PATH)
        print(f"Added {SPARK_TTS_REPO_PATH} to Python path")
    else:
        print(f"Warning: {SPARK_TTS_REPO_PATH} not found. Clone it with:")
        print(f"git clone https://github.com/SparkAudio/Spark-TTS")
    
    # Load vLLM model with single GPU to avoid tensor parallelism issues
    print(f"Loading Spark TTS model: {MODEL_NAME}...")
    vllm_model = LLM(
        MODEL_NAME,
        enforce_eager=False,
        gpu_memory_utilization=0.5,
        tensor_parallel_size=1  # Use single GPU to avoid multi-GPU issues
    )
    print("✅ Model loaded successfully!")
    
    # Download tokenizer if needed
    if not os.path.exists(TOKENIZER_CACHE_DIR) or not os.path.exists(f"{TOKENIZER_CACHE_DIR}/config.yaml"):
        print(f"Downloading tokenizer from {TOKENIZER_REPO}...")
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            local_dir=TOKENIZER_CACHE_DIR,
        )
        print(f"✅ Tokenizer downloaded to {TOKENIZER_CACHE_DIR}")
    else:
        print(f"✅ Tokenizer already exists at {TOKENIZER_CACHE_DIR}")
    
    # Initialize audio tokenizer
    try:
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        print("Initializing audio tokenizer...")
        audio_tokenizer = BiCodecTokenizer(TOKENIZER_CACHE_DIR, device)
        print("✅ Audio tokenizer initialized!")
    except ImportError:
        print("Error: Could not import BiCodecTokenizer. Make sure Spark-TTS repo is available.")
        raise


def generate_audio_segment(text: str, speaker_id: int, temperature: float) -> np.ndarray:
    """Generate audio for a single text segment with strict memory limits."""
    global_tokens = GLOBAL_IDS_BY_SPEAKER[speaker_id]
    
    # Clear CUDA cache before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create prompt
    prompt = f"<|task_tts|><|start_content|>{speaker_id}: {text}<|end_content|><|start_global_token|>"
    prompt += ''.join([f'<|bicodec_global_{t}|>' for t in global_tokens])
    prompt += '<|end_global_token|><|start_semantic_token|>'
    
    # Generate with vLLM
    sampling_params = SamplingParams(temperature=temperature, max_tokens=DEFAULT_MAX_TOKENS)
    outputs = vllm_model.generate(prompts=[prompt], sampling_params=sampling_params)
    
    # Extract semantic tokens
    predicted_tokens = outputs[0].outputs[0].text
    semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicted_tokens)
    
    if not semantic_matches:
        raise ValueError("No semantic tokens found in the generated output.")
    
    # Strict semantic token limit to prevent memory issues
    if len(semantic_matches) > 800:
        semantic_matches = semantic_matches[:800]
        print(f"Limited semantic tokens to 800 for memory safety")
    
    # Convert to tensors
    pred_semantic_ids = (
        torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)
    )
    pred_global_ids = torch.tensor([global_tokens]).long()
    
    # Decode to audio with aggressive memory management
    with torch.no_grad():  # Disable gradient computation
        wav_np = audio_tokenizer.detokenize(
            pred_global_ids.to(device), pred_semantic_ids.to(device)
        )
    
    # Aggressive cleanup
    del pred_semantic_ids, pred_global_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all CUDA operations complete
        # Force garbage collection
        import gc
        gc.collect()
    
    return wav_np


def convert_to_pcm16_bytes(audio_np: np.ndarray) -> bytes:
    """Convert numpy audio array to PCM16 bytes."""
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


async def generate_audio_chunks_async(
    text: str,
    speaker_id: int,
    temperature: float
):
    """Async generator that yields audio chunks for streaming."""
    loop = asyncio.get_running_loop()
    
    try:
        # Split text into sentences
        sentences = chunk_text_simple(text)
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            print(f"Generating audio for sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
            
            # Generate audio in thread pool to avoid blocking
            audio_np = await loop.run_in_executor(
                None,
                generate_audio_segment,
                sentence,
                speaker_id,
                temperature
            )
            
            # Convert to PCM bytes
            pcm_bytes = convert_to_pcm16_bytes(audio_np)
            
            print(f"Generated audio shape: {audio_np.shape}, PCM bytes length: {len(pcm_bytes)}")
            
            if len(pcm_bytes) > 0:
                # Yield immediately for real-time streaming
                yield pcm_bytes
                print(f"Streamed audio chunk {i+1}/{len(sentences)} immediately")
            else:
                print("Warning: Generated empty audio data")
                
    except Exception as e:
        print(f"Error during audio generation: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    print("Initializing Spark TTS models...")
    initialize_models()
    print("Server ready!")


@app.websocket("/v1/audio/speech/stream/ws")
async def websocket_audio_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio generation.
    
    Protocol:
    - Client sends JSON: {"input": "text", "voice": "voice_name", "speaker_id": 248, "continue": true/false, "segment_id": "id"}
    - Server sends: {"type": "start", "segment_id": "id"} followed by binary audio chunks
    - Server sends: {"type": "end", "segment_id": "id"} when segment complete
    """
    await websocket.accept()
    print("WebSocket connection established")
    
    # Set up ping/pong for connection health monitoring
    ping_task = None
    last_activity = time.time()
    
    async def ping_loop():
        """Send periodic pings to keep connection alive."""
        nonlocal last_activity
        while True:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                if time.time() - last_activity > 60:  # No activity for 1 minute
                    print("Connection idle, sending ping")
                    await websocket.send_json({"type": "ping"})
                    last_activity = time.time()
            except Exception:
                break
    
    try:
        ping_task = asyncio.create_task(ping_loop())
        
        while True:
            try:
                # Set a reasonable timeout for receiving messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)  # 5 minutes
                last_activity = time.time()
                message = json.loads(data)
                
                text = message.get("input", "")
                voice = message.get("voice", "luganda_female")
                speaker_id = message.get("speaker_id")
                temperature = message.get("temperature", DEFAULT_TEMPERATURE)
                continue_stream = message.get("continue", True)
                segment_id = message.get("segment_id", "default")
                
                # Resolve speaker ID
                if speaker_id is None:
                    speaker_id = SPEAKER_IDS.get(voice, DEFAULT_SPEAKER_ID)
                
                if not text and not continue_stream:
                    print("Received end signal, closing stream")
                    break
                
                if text:
                    # Send start message
                    await websocket.send_json({
                        "type": "start",
                        "segment_id": segment_id,
                        "speaker_id": speaker_id
                    })
                    
                    # Stream audio chunks continuously for the session
                    chunk_count = 0
                    try:
                        async_generator = generate_audio_chunks_async(
                            text=text,
                            speaker_id=speaker_id,
                            temperature=temperature
                        )
                        
                        # Process all chunks without timeout - let the session stay open
                        async for audio_chunk in async_generator:
                            chunk_count += 1
                            print(f"Immediately sending audio chunk {chunk_count}: {len(audio_chunk)} bytes")
                            await websocket.send_bytes(audio_chunk)
                            print(f"Audio chunk {chunk_count} sent and played immediately")
                            last_activity = time.time()
                                
                    except Exception as e:
                        print(f"Error during audio generation: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": f"Audio generation error: {str(e)}"
                        })
                        continue
                    
                    print(f"Finished streaming {chunk_count} audio chunks, sending end message")
                    # Send end message but keep session open for more sentences
                    await websocket.send_json({
                        "type": "end",
                        "segment_id": segment_id
                    })
                    
            except asyncio.TimeoutError:
                print("Client timeout, closing connection")
                break
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except json.JSONDecodeError:
                print("Invalid JSON received")
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ping_task:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
        print("WebSocket connection closed")


@app.websocket("/v1/audio/speech/clone/ws")
async def websocket_voice_cloning(websocket: WebSocket):
    """
    WebSocket endpoint for voice cloning streaming.
    
    Protocol:
    - Client sends JSON: {"input": "text", "reference_audio_path": "path/to/audio.wav", "reference_text": "optional", "temperature": 0.7, "segment_id": "id"}
    - Server sends: {"type": "start", "segment_id": "id"} followed by binary audio chunks
    - Server sends: {"type": "end", "segment_id": "id"} when segment complete
    """
    await websocket.accept()
    print("Voice cloning WebSocket connection established")
    
    # Set up ping/pong for connection health monitoring
    ping_task = None
    last_activity = time.time()
    
    async def ping_loop():
        """Send periodic pings to keep connection alive."""
        nonlocal last_activity
        while True:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                if time.time() - last_activity > 60:  # No activity for 1 minute
                    print("Connection idle, sending ping")
                    await websocket.send_json({"type": "ping"})
                    last_activity = time.time()
            except Exception:
                break
    
    try:
        ping_task = asyncio.create_task(ping_loop())
        
        while True:
            try:
                # Set a reasonable timeout for receiving messages
                data = await asyncio.wait_for(websocket.receive_text(), timeout=300)  # 5 minutes
                last_activity = time.time()
                message = json.loads(data)
                
                text = message.get("input", "")
                reference_audio_path = message.get("reference_audio_path", "")
                reference_text = message.get("reference_text")
                temperature = message.get("temperature", DEFAULT_TEMPERATURE)
                segment_id = message.get("segment_id", "default")
                
                if not text:
                    print("Received empty text, skipping")
                    continue
                
                if not reference_audio_path:
                    await websocket.send_json({
                        "type": "error",
                        "message": "reference_audio_path is required for voice cloning"
                    })
                    continue
                
                # Validate reference audio file exists
                if not os.path.exists(reference_audio_path):
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Reference audio file not found: {reference_audio_path}"
                    })
                    continue
                
                # Send start message
                await websocket.send_json({
                    "type": "start",
                    "segment_id": segment_id,
                    "reference_audio": reference_audio_path
                })
                
                # Generate cloned audio
                try:
                    loop = asyncio.get_running_loop()
                    
                    # Generate cloned audio in thread pool
                    result_wav = await loop.run_in_executor(
                        None,
                        text_to_speech_cloned,
                        text,
                        audio_tokenizer,
                        vllm_model,
                        reference_audio_path,
                        reference_text,
                        temperature,
                        device
                    )
                    
                    # Convert to PCM bytes
                    pcm_bytes = convert_to_pcm16_bytes(result_wav)
                    
                    print(f"Generated cloned audio: {len(result_wav)} samples, {len(pcm_bytes)} bytes")
                    
                    if len(pcm_bytes) > 0:
                        # Send audio data
                        await websocket.send_bytes(pcm_bytes)
                        print(f"Cloned audio sent for segment {segment_id}")
                    else:
                        print("Warning: Generated empty cloned audio data")
                        
                except Exception as e:
                    print(f"Error during voice cloning: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Voice cloning error: {str(e)}"
                    })
                    continue
                
                # Send end message
                await websocket.send_json({
                    "type": "end",
                    "segment_id": segment_id
                })
                    
            except asyncio.TimeoutError:
                print("Client timeout, closing connection")
                break
            except WebSocketDisconnect:
                print("Client disconnected")
                break
            except json.JSONDecodeError:
                print("Invalid JSON received")
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ping_task:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
        print("Voice cloning WebSocket connection closed")


@app.post("/v1/audio/speech/stream")
async def http_audio_stream(request: AudioRequest):
    """
    HTTP endpoint for streaming audio as raw PCM bytes.
    """
    print(f"Received HTTP streaming request for: '{request.text[:50]}...'")
    
    # Resolve speaker ID
    speaker_id = request.speaker_id
    if speaker_id is None:
        speaker_id = SPEAKER_IDS.get(request.voice, DEFAULT_SPEAKER_ID)
    
    async def stream_pcm():
        async for chunk in generate_audio_chunks_async(
            text=request.text,
            speaker_id=speaker_id,
            temperature=request.temperature
        ):
            yield chunk
    
    return StreamingResponse(
        stream_pcm(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(AUDIO_SAMPLERATE),
            "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
            "X-Channels": str(AUDIO_CHANNELS),
        }
    )


@app.post("/v1/audio/speech/clone/upload")
async def voice_cloning_upload(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: Optional[str] = Form(None),
    temperature: float = Form(DEFAULT_TEMPERATURE)
):
    """
    Voice cloning endpoint that accepts reference audio as file upload.
    
    Args:
        text: Text to synthesize
        reference_audio: Audio file for voice cloning
        reference_text: Optional transcript of reference audio
        temperature: Controls randomness (0.0-1.0)
    
    Returns:
        StreamingResponse with PCM audio data
    """
    print(f"Received voice cloning upload request for: '{text[:50]}...'")
    print(f"Reference audio file: {reference_audio.filename}, size: {reference_audio.size}")
    print(f"Reference text: {reference_text}")
    print(f"Temperature: {temperature}")
    
    # Validate text
    if not text or not text.strip():
        error_msg = "Text parameter is required and cannot be empty"
        print(f"ERROR: {error_msg}")
        return {"error": error_msg}
    
    # Validate file
    if not reference_audio or not reference_audio.filename:
        error_msg = "Reference audio file is required"
        print(f"ERROR: {error_msg}")
        return {"error": error_msg}
    
    # Validate file type
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    file_ext = os.path.splitext(reference_audio.filename)[1].lower()
    if file_ext not in allowed_extensions:
        error_msg = f"Invalid audio file format: {file_ext}. Allowed formats: {', '.join(allowed_extensions)}"
        print(f"ERROR: {error_msg}")
        return {"error": error_msg}
    
    # Validate temperature
    if not 0.0 <= temperature <= 1.0:
        error_msg = f"Temperature must be between 0.0 and 1.0, got: {temperature}"
        print(f"ERROR: {error_msg}")
        return {"error": error_msg}
    
    # Create temporary file for uploaded audio
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    
    # Flag to track if cleanup should be deferred
    defer_cleanup = False
    
    try:
        # Save uploaded audio to temporary file
        print(f"Saving uploaded audio to: {temp_path}")
        content = await reference_audio.read()
        
        if not content:
            error_msg = "Uploaded audio file is empty"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
        
        print(f"Received audio content: {len(content)} bytes")
        print(f"Content type: {reference_audio.content_type}")
        
        # Save the file
        with open(temp_path, "wb") as f:
            f.write(content)
        
        print(f"Audio file saved, size: {len(content)} bytes")
        
        # Verify the file exists and is readable
        if not os.path.exists(temp_path):
            error_msg = f"Failed to save temporary file: {temp_path}"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        print(f"Saved file size: {file_size} bytes")
        
        if file_size == 0:
            error_msg = "Saved audio file is empty"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
        
        # Test if the file can be read by soundfile
        try:
            test_wav, test_sr = sf.read(temp_path)
            print(f"Successfully read test audio: shape={test_wav.shape}, sr={test_sr}")
        except Exception as e:
            error_msg = f"Cannot read saved audio file: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {"error": error_msg}
        
        # Defer cleanup to the streaming function
        defer_cleanup = True
        
        async def stream_cloned_pcm():
            loop = asyncio.get_running_loop()
            
            try:
                print("Starting voice cloning process with uploaded audio...")
                
                # Generate cloned audio in thread pool
                result_wav = await loop.run_in_executor(
                    None,
                    text_to_speech_cloned,
                    text,
                    audio_tokenizer,
                    vllm_model,
                    temp_path,
                    reference_text,
                    temperature,
                    device
                )
                
                print(f"Voice cloning completed. Generated audio shape: {result_wav.shape}")
                
                # Convert to PCM bytes
                pcm_bytes = convert_to_pcm16_bytes(result_wav)
                
                print(f"Converted to PCM: {len(result_wav)} samples, {len(pcm_bytes)} bytes")
                
                if len(pcm_bytes) > 0:
                    yield pcm_bytes
                    print(f"Successfully yielded {len(pcm_bytes)} bytes of audio data")
                else:
                    error_msg = "Warning: Generated empty cloned audio data"
                    print(f"ERROR: {error_msg}")
                    raise ValueError(error_msg)
                    
            except Exception as e:
                error_msg = f"Error during voice cloning: {str(e)}"
                print(f"ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                # Return error as JSON instead of raising exception
                yield json.dumps({"error": error_msg}).encode()
            
            finally:
                # Clean up temporary file AFTER voice cloning completes
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
        
        return StreamingResponse(
            stream_cloned_pcm(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": str(AUDIO_SAMPLERATE),
                "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
                "X-Channels": str(AUDIO_CHANNELS),
                "X-Voice-Cloning": "true"
            }
        )
        
    except Exception as e:
        error_msg = f"Error processing upload: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return {"error": error_msg}
        
    finally:
        # Only clean up if not deferred to streaming function
        if not defer_cleanup and os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"Cleaned up temporary file: {temp_path}")


@app.post("/v1/audio/speech/clone/debug")
async def voice_cloning_debug(
    text: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
    reference_text: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None)
):
    """
    Debug endpoint to test form data parsing without processing.
    """
    print("=== DEBUG ENDPOINT CALLED ===")
    print(f"Text: {text}")
    print(f"Reference audio: {reference_audio}")
    if reference_audio:
        print(f"  - Filename: {reference_audio.filename}")
        print(f"  - Size: {reference_audio.size}")
        print(f"  - Content type: {reference_audio.content_type}")
    print(f"Reference text: {reference_text}")
    print(f"Temperature: {temperature}")
    print("=== END DEBUG ===")
    
    return {
        "message": "Debug endpoint received data",
        "text": text,
        "reference_audio_filename": reference_audio.filename if reference_audio else None,
        "reference_audio_size": reference_audio.size if reference_audio else None,
        "reference_text": reference_text,
        "temperature": temperature
    }


@app.post("/v1/audio/speech/clone")
async def voice_cloning_http(request: VoiceCloningRequest):
    """
    HTTP endpoint for voice cloning using reference audio.
    
    Args:
        request: VoiceCloningRequest containing text, reference audio path, and parameters
    
    Returns:
        StreamingResponse with PCM audio data
    """
    print(f"Received voice cloning request for: '{request.text[:50]}...'")
    print(f"Reference audio: {request.reference_audio_path}")
    
    # Validate reference audio file exists
    if not os.path.exists(request.reference_audio_path):
        error_msg = f"Reference audio file not found: {request.reference_audio_path}"
        print(f"ERROR: {error_msg}")
        return {"error": error_msg}
    
    async def stream_cloned_pcm():
        loop = asyncio.get_running_loop()
        
        try:
            print("Starting voice cloning process...")
            
            # Generate cloned audio in thread pool
            result_wav = await loop.run_in_executor(
                None,
                text_to_speech_cloned,
                request.text,
                audio_tokenizer,
                vllm_model,
                request.reference_audio_path,
                request.reference_text,
                request.temperature,
                device
            )
            
            print(f"Voice cloning completed. Generated audio shape: {result_wav.shape}")
            
            # Convert to PCM bytes
            pcm_bytes = convert_to_pcm16_bytes(result_wav)
            
            print(f"Converted to PCM: {len(result_wav)} samples, {len(pcm_bytes)} bytes")
            
            if len(pcm_bytes) > 0:
                yield pcm_bytes
                print(f"Successfully yielded {len(pcm_bytes)} bytes of audio data")
            else:
                error_msg = "Warning: Generated empty cloned audio data"
                print(f"ERROR: {error_msg}")
                raise ValueError(error_msg)
                
        except Exception as e:
            error_msg = f"Error during voice cloning: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            # Return error as JSON instead of raising exception
            yield json.dumps({"error": error_msg}).encode()
    
    return StreamingResponse(
        stream_cloned_pcm(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(AUDIO_SAMPLERATE),
            "X-Bit-Depth": str(AUDIO_BITS_PER_SAMPLE),
            "X-Channels": str(AUDIO_CHANNELS),
            "X-Voice-Cloning": "true"
        }
    )


@app.get("/")
async def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Spark TTS Streaming API with Voice Cloning",
        "model": MODEL_NAME,
        "sample_rate": AUDIO_SAMPLERATE,
        "available_voices": list(SPEAKER_IDS.keys()),
        "features": ["text-to-speech", "voice-cloning", "streaming"],
        "endpoints": {
            "websocket": "/v1/audio/speech/stream/ws",
            "http": "/v1/audio/speech/stream",
            "voice_cloning_websocket": "/v1/audio/speech/clone/ws",
            "voice_cloning_http": "/v1/audio/speech/clone",
            "voice_cloning_upload": "/v1/audio/speech/clone/upload",
            "voices": "/v1/voices"
        },
        "example_usage": {
            "websocket": {
                "connect": "ws://localhost:8000/v1/audio/speech/stream/ws",
                "send": {
                    "input": "Your text here",
                    "voice": "luganda_female",
                    "segment_id": "segment_1"
                }
            },
            "http": {
                "url": "POST /v1/audio/speech/stream",
                "body": {
                    "text": "Your text here",
                    "voice": "luganda_female",
                    "temperature": 0.7
                }
            },
            "voice_cloning_websocket": {
                "connect": "ws://localhost:8000/v1/audio/speech/clone/ws",
                "send": {
                    "input": "Your text here",
                    "reference_audio_path": "/path/to/reference.wav",
                    "reference_text": "Optional transcript of reference audio",
                    "temperature": 0.7,
                    "segment_id": "clone_segment_1"
                }
            },
            "voice_cloning_http": {
                "url": "POST /v1/audio/speech/clone/upload",
                "body": {
                    "text": "Your text here",
                    "reference_audio": "/path/to/reference.wav",
                    "reference_text": "Optional transcript of reference audio",
                    "temperature": 0.7
                }
            }
        },
        "voice_cloning_info": {
            "description": "Zero-shot voice cloning using reference audio",
            "requirements": [
                "Reference audio file must exist and be accessible",
                "Audio will be automatically resampled to 16kHz",
                "Supports WAV, MP3, and other audio formats"
            ],
            "parameters": {
                "text": "Text to synthesize (required)",
                "reference_audio": "Reference audio file (required)",
                "reference_text": "Optional transcript of reference audio",
                "temperature": "Controls randomness (0.0-1.0, default: 0.7)"
            }
        }
    }


@app.get("/v1/voices")
async def list_voices():
    """List available voices."""
    return {
        "voices": [
            {"id": name, "speaker_id": sid, "language": name.split("_")[0]}
            for name, sid in SPEAKER_IDS.items()
        ]
    }


if __name__ == "__main__":
    print("Starting Spark TTS FastAPI server with WebSocket support...")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {CUDA_VISIBLE_DEVICES}")
    uvicorn.run(
        "spark_tts_streaming:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        timeout_keep_alive=300

    )


