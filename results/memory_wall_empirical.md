# Empirical Memory Wall Sweep Results

- **Hardware Profile:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (6.0 GB VRAM)
- **Model:** `Qwen/Qwen2.5-0.5B` (`float16`)
- **Max New Tokens:** 256

| Language   |   K Paths |   Prompt Tokens |   Max Gen Tokens |   Total Trajectory Tokens |   Peak VRAM (GB) |   KV & Act (GB) | Status    |
|:-----------|----------:|----------------:|-----------------:|--------------------------:|-----------------:|----------------:|:----------|
| English    |         1 |              58 |              256 |                       314 |           0.939  |          0.0188 | COMPLETED |
| English    |         2 |              58 |              256 |                       628 |           0.9527 |          0.0324 | COMPLETED |
| English    |         4 |              58 |              256 |                      1256 |           0.9779 |          0.0577 | COMPLETED |
| English    |         8 |              58 |              256 |                      2512 |           1.0275 |          0.1072 | COMPLETED |
| English    |        12 |              58 |              256 |                      3768 |           1.0785 |          0.1582 | COMPLETED |
| English    |        16 |              58 |              256 |                      5024 |           1.1294 |          0.2091 | COMPLETED |
| English    |        20 |              58 |              256 |                      6280 |           1.1803 |          0.2601 | COMPLETED |
| English    |        24 |              58 |              256 |                      7536 |           1.2279 |          0.3076 | COMPLETED |
| Hindi      |         1 |             198 |              256 |                       454 |           0.9407 |          0.0204 | COMPLETED |
| Hindi      |         2 |             198 |              256 |                       908 |           0.9569 |          0.0367 | COMPLETED |
| Hindi      |         4 |             198 |              256 |                      1816 |           0.985  |          0.0647 | COMPLETED |
| Hindi      |         8 |             198 |              256 |                      3632 |           1.0403 |          0.1201 | COMPLETED |
| Hindi      |        12 |             198 |              256 |                      5448 |           1.1001 |          0.1799 | COMPLETED |
| Hindi      |        16 |             198 |              256 |                      7264 |           1.155  |          0.2348 | COMPLETED |
| Hindi      |        20 |             198 |              256 |                      9080 |           1.2139 |          0.2936 | COMPLETED |
| Hindi      |        24 |             198 |              256 |                     10896 |           1.2648 |          0.3446 | COMPLETED |
| Telugu     |         1 |             347 |              256 |                       603 |           0.9548 |          0.0346 | COMPLETED |
| Telugu     |         2 |             347 |              256 |                      1206 |           0.9812 |          0.0609 | COMPLETED |
| Telugu     |         4 |             347 |              256 |                      2412 |           1.0311 |          0.1109 | COMPLETED |
| Telugu     |         8 |             347 |              256 |                      4824 |           1.1336 |          0.2133 | COMPLETED |
| Telugu     |        12 |             347 |              256 |                      7236 |           1.2419 |          0.3217 | COMPLETED |
| Telugu     |        16 |             347 |              256 |                      9648 |           1.3368 |          0.4165 | COMPLETED |
| Telugu     |        20 |             347 |              256 |                     12060 |           1.4409 |          0.5206 | COMPLETED |
| Telugu     |        24 |             347 |              256 |                     14472 |           1.5404 |          0.6201 | COMPLETED |