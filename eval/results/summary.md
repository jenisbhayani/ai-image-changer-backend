# Background Removal Evaluation Summary

| Pipeline | p50 Latency | p95 Latency | Peak RAM | Uncertain % | Mean Halo Score | Mean Components |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **legacy** | 6575.7 ms | 16115.7 ms | 238.3 MB | 2.10% | 97.41 | 1.07 |
| **fast** | 4843.9 ms | 6565.5 ms | 238.6 MB | 12.41% | 95.38 | 1.13 |
| **balanced** | 4967.5 ms | 8825.4 ms | 238.6 MB | 12.34% | 105.58 | 1.13 |
| **high** | 7998.9 ms | 8756.3 ms | 238.6 MB | 12.49% | 79.09 | 1.47 |

### Per-Category Breakdown

| Category | Pipeline | Latency (ms) | Uncertain % | Halo Score | Components |
| :--- | :--- | :---: | :---: | :---: | :---: |
| food | legacy | 16115.7 | 0.62% | 234.93 | 1 |
| full_body | legacy | 6142.4 | 1.07% | 58.53 | 1 |
| group | legacy | 6660.3 | 0.93% | 56.11 | 1 |
| hair_fine | legacy | 6196.0 | 2.52% | 35.33 | 1 |
| large_4k_photo | legacy | 7994.1 | 7.23% | 159.16 | 1 |
| low_contrast | legacy | 7454.6 | 1.70% | 101.03 | 1 |
| low_light | legacy | 6928.4 | 0.37% | 73.15 | 1 |
| pet_fur | legacy | 5876.3 | 1.75% | 18.30 | 1 |
| portrait_closeup | legacy | 6727.9 | 0.41% | 110.21 | 1 |
| poster_graphic_text | legacy | 6542.1 | 1.69% | 127.42 | 1 |
| product_busy | legacy | 6575.7 | 3.75% | 41.09 | 1 |
| product_clean | legacy | 10057.6 | 1.90% | 233.74 | 1 |
| small_subject | legacy | 5396.3 | 3.01% | 24.48 | 1 |
| transparent_glass | legacy | 5762.1 | 2.66% | 104.96 | 2 |
| vehicle | legacy | 5417.5 | 1.92% | 82.66 | 1 |
| food | fast | 5116.1 | 2.39% | 207.23 | 1 |
| full_body | fast | 4447.8 | 3.93% | 65.05 | 2 |
| group | fast | 5197.8 | 1.13% | 63.63 | 1 |
| hair_fine | fast | 4666.2 | 9.86% | 37.70 | 1 |
| large_4k_photo | fast | 4755.6 | 34.83% | 147.83 | 1 |
| low_contrast | fast | 5928.5 | 2.66% | 100.36 | 1 |
| low_light | fast | 4407.3 | 94.14% | 65.10 | 1 |
| pet_fur | fast | 4216.6 | 8.51% | 16.83 | 1 |
| portrait_closeup | fast | 5338.7 | 0.75% | 103.55 | 1 |
| poster_graphic_text | fast | 4340.1 | 4.11% | 119.79 | 1 |
| product_busy | fast | 4646.0 | 2.10% | 82.60 | 1 |
| product_clean | fast | 4843.9 | 1.29% | 267.76 | 1 |
| small_subject | fast | 6388.5 | 7.35% | 21.11 | 1 |
| transparent_glass | fast | 6565.5 | 6.24% | 34.78 | 2 |
| vehicle | fast | 6140.9 | 6.92% | 97.32 | 1 |
| food | balanced | 5040.8 | 2.39% | 207.23 | 1 |
| full_body | balanced | 4960.5 | 3.93% | 65.05 | 2 |
| group | balanced | 5177.7 | 1.13% | 63.63 | 1 |
| hair_fine | balanced | 4490.0 | 9.86% | 37.70 | 1 |
| large_4k_photo | balanced | 4223.0 | 34.83% | 147.83 | 1 |
| low_contrast | balanced | 6135.9 | 2.69% | 101.09 | 1 |
| low_light | balanced | 5144.7 | 92.99% | 217.49 | 1 |
| pet_fur | balanced | 5708.1 | 8.51% | 16.83 | 1 |
| portrait_closeup | balanced | 8825.4 | 0.75% | 103.55 | 1 |
| poster_graphic_text | balanced | 6636.2 | 4.11% | 119.79 | 1 |
| product_busy | balanced | 4685.2 | 2.10% | 82.60 | 1 |
| product_clean | balanced | 4967.5 | 1.29% | 267.76 | 1 |
| small_subject | balanced | 4317.9 | 7.35% | 21.11 | 1 |
| transparent_glass | balanced | 4304.7 | 6.24% | 34.78 | 2 |
| vehicle | balanced | 4557.9 | 6.92% | 97.32 | 1 |
| food | high | 8189.3 | 3.54% | 24.12 | 1 |
| full_body | high | 8093.8 | 6.75% | 73.71 | 2 |
| group | high | 8443.5 | 1.13% | 63.41 | 1 |
| hair_fine | high | 7870.2 | 10.83% | 41.23 | 1 |
| large_4k_photo | high | 7998.9 | 21.40% | 123.44 | 1 |
| low_contrast | high | 8165.2 | 3.31% | 98.46 | 1 |
| low_light | high | 8756.3 | 99.81% | 28.24 | 6 |
| pet_fur | high | 7935.1 | 5.84% | 12.65 | 1 |
| portrait_closeup | high | 7846.5 | 0.75% | 103.22 | 1 |
| poster_graphic_text | high | 7444.4 | 5.04% | 119.41 | 1 |
| product_busy | high | 7859.0 | 2.44% | 81.16 | 1 |
| product_clean | high | 8234.9 | 1.30% | 266.31 | 1 |
| small_subject | high | 7668.5 | 9.93% | 24.49 | 1 |
| transparent_glass | high | 7793.8 | 6.17% | 34.99 | 2 |
| vehicle | high | 8102.0 | 9.06% | 91.46 | 1 |
