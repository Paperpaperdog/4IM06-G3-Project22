# Raw test metrics（直接评估输出）
来源：`results/{mask,cnn}/n6_*_size{N}/metrics.json` · **test** · 每类 support=1000

## Table A — Overall

| Method | Size | accuracy | macro_P | macro_R | macro_F1 | weighted_F1 |
|--------|------|----------|---------|---------|----------|-------------|
| Mask | 32 | 0.605167 | 0.590422 | 0.605167 | 0.592195 | 0.592195 |
| Mask | 64 | 0.627833 | 0.607052 | 0.627833 | 0.605422 | 0.605422 |
| Mask | 96 | 0.642167 | 0.624621 | 0.642167 | 0.617637 | 0.617637 |
| Mask | 128 | 0.647833 | 0.628747 | 0.647833 | 0.620080 | 0.620080 |
| CNN | 32 | 0.702833 | 0.000000 | 0.000000 | 0.695158 | 0.695158 |
| CNN | 64 | 0.774167 | 0.000000 | 0.000000 | 0.745566 | 0.745566 |
| CNN | 96 | 0.780833 | 0.000000 | 0.000000 | 0.774324 | 0.774324 |
| CNN | 128 | 0.827833 | 0.000000 | 0.000000 | 0.824519 | 0.824519 |

## Table B — Per-class precision / recall / F1

| Method | Size | class | precision | recall | F1 | support |
|--------|------|-------|-----------|--------|-----|--------|
| Mask | 32 | original | 0.428135 | 0.280000 | 0.338573 | 1000 |
| Mask | 32 | JPEG_Q80 | 0.524590 | 0.672000 | 0.589215 | 1000 |
| Mask | 32 | downsample_x8 | 0.376590 | 0.296000 | 0.331467 | 1000 |
| Mask | 32 | downsample_x16 | 0.424294 | 0.496000 | 0.457354 | 1000 |
| Mask | 32 | upsample_x4 | 0.904808 | 0.941000 | 0.922549 | 1000 |
| Mask | 32 | upsample_x8 | 0.884112 | 0.946000 | 0.914010 | 1000 |
| Mask | 64 | original | 0.470410 | 0.310000 | 0.373719 | 1000 |
| Mask | 64 | JPEG_Q80 | 0.559080 | 0.705000 | 0.623618 | 1000 |
| Mask | 64 | downsample_x8 | 0.320388 | 0.165000 | 0.217822 | 1000 |
| Mask | 64 | downsample_x16 | 0.456199 | 0.677000 | 0.545089 | 1000 |
| Mask | 64 | upsample_x4 | 0.898674 | 0.949000 | 0.923152 | 1000 |
| Mask | 64 | upsample_x8 | 0.937561 | 0.961000 | 0.949136 | 1000 |
| Mask | 96 | original | 0.474151 | 0.321000 | 0.382826 | 1000 |
| Mask | 96 | JPEG_Q80 | 0.550798 | 0.759000 | 0.638352 | 1000 |
| Mask | 96 | downsample_x8 | 0.386207 | 0.168000 | 0.234146 | 1000 |
| Mask | 96 | downsample_x16 | 0.475534 | 0.690000 | 0.563035 | 1000 |
| Mask | 96 | upsample_x4 | 0.906578 | 0.951000 | 0.928258 | 1000 |
| Mask | 96 | upsample_x8 | 0.954455 | 0.964000 | 0.959204 | 1000 |
| Mask | 128 | original | 0.494346 | 0.306000 | 0.378011 | 1000 |
| Mask | 128 | JPEG_Q80 | 0.563191 | 0.713000 | 0.629303 | 1000 |
| Mask | 128 | downsample_x8 | 0.390187 | 0.167000 | 0.233894 | 1000 |
| Mask | 128 | downsample_x16 | 0.485822 | 0.771000 | 0.596057 | 1000 |
| Mask | 128 | upsample_x4 | 0.896165 | 0.958000 | 0.926051 | 1000 |
| Mask | 128 | upsample_x8 | 0.942774 | 0.972000 | 0.957164 | 1000 |
| CNN | 32 | original | 0.631271 | 0.755000 | 0.687614 | 1000 |
| CNN | 32 | JPEG_Q80 | 0.678112 | 0.790000 | 0.729792 | 1000 |
| CNN | 32 | downsample_x8 | 0.434608 | 0.432000 | 0.433300 | 1000 |
| CNN | 32 | downsample_x16 | 0.505935 | 0.341000 | 0.407407 | 1000 |
| CNN | 32 | upsample_x4 | 0.968687 | 0.959000 | 0.963819 | 1000 |
| CNN | 32 | upsample_x8 | 0.958206 | 0.940000 | 0.949016 | 1000 |
| CNN | 64 | original | 0.747500 | 0.897000 | 0.815455 | 1000 |
| CNN | 64 | JPEG_Q80 | 0.878981 | 0.828000 | 0.852729 | 1000 |
| CNN | 64 | downsample_x8 | 0.569767 | 0.147000 | 0.233704 | 1000 |
| CNN | 64 | downsample_x16 | 0.531392 | 0.821000 | 0.645187 | 1000 |
| CNN | 64 | upsample_x4 | 0.995829 | 0.955000 | 0.974987 | 1000 |
| CNN | 64 | upsample_x8 | 0.909672 | 0.997000 | 0.951336 | 1000 |
| CNN | 96 | original | 0.871186 | 0.771000 | 0.818037 | 1000 |
| CNN | 96 | JPEG_Q80 | 0.772047 | 0.928000 | 0.842870 | 1000 |
| CNN | 96 | downsample_x8 | 0.539846 | 0.420000 | 0.472441 | 1000 |
| CNN | 96 | downsample_x16 | 0.570755 | 0.605000 | 0.587379 | 1000 |
| CNN | 96 | upsample_x4 | 0.918819 | 0.996000 | 0.955854 | 1000 |
| CNN | 96 | upsample_x8 | 0.973764 | 0.965000 | 0.969362 | 1000 |
| CNN | 128 | original | 0.911538 | 0.948000 | 0.929412 | 1000 |
| CNN | 128 | JPEG_Q80 | 0.965440 | 0.866000 | 0.913021 | 1000 |
| CNN | 128 | downsample_x8 | 0.612329 | 0.447000 | 0.516763 | 1000 |
| CNN | 128 | downsample_x16 | 0.583465 | 0.741000 | 0.652863 | 1000 |
| CNN | 128 | upsample_x4 | 0.988741 | 0.966000 | 0.977238 | 1000 |
| CNN | 128 | upsample_x8 | 0.919890 | 0.999000 | 0.957814 | 1000 |

## Table C — Confusion matrices

### Mask · size=32

| true \\ pred | original | JPEG_Q80 | downsample_x8 | downsample_x16 | upsample_x4 | upsample_x8 |
|---|---|---|---|---|---|---|
| original | 280 | 407 | 81 | 211 | 18 | 3 |
| JPEG_Q80 | 113 | 672 | 44 | 69 | 35 | 67 |
| downsample_x8 | 144 | 120 | 296 | 391 | 25 | 24 |
| downsample_x16 | 117 | 78 | 279 | 496 | 19 | 11 |
| upsample_x4 | 0 | 4 | 35 | 1 | 941 | 19 |
| upsample_x8 | 0 | 0 | 51 | 1 | 2 | 946 |

### Mask · size=64

| true \\ pred | original | JPEG_Q80 | downsample_x8 | downsample_x16 | upsample_x4 | upsample_x8 |
|---|---|---|---|---|---|---|
| original | 310 | 413 | 70 | 172 | 30 | 5 |
| JPEG_Q80 | 95 | 705 | 62 | 68 | 28 | 42 |
| downsample_x8 | 146 | 81 | 165 | 567 | 38 | 3 |
| downsample_x16 | 108 | 51 | 150 | 677 | 10 | 4 |
| upsample_x4 | 0 | 11 | 30 | 0 | 949 | 10 |
| upsample_x8 | 0 | 0 | 38 | 0 | 1 | 961 |

### Mask · size=96

| true \\ pred | original | JPEG_Q80 | downsample_x8 | downsample_x16 | upsample_x4 | upsample_x8 |
|---|---|---|---|---|---|---|
| original | 321 | 452 | 46 | 142 | 35 | 4 |
| JPEG_Q80 | 95 | 759 | 30 | 72 | 24 | 20 |
| downsample_x8 | 143 | 101 | 168 | 547 | 34 | 7 |
| downsample_x16 | 118 | 58 | 129 | 690 | 4 | 1 |
| upsample_x4 | 0 | 8 | 27 | 0 | 951 | 14 |
| upsample_x8 | 0 | 0 | 35 | 0 | 1 | 964 |

### Mask · size=128

| true \\ pred | original | JPEG_Q80 | downsample_x8 | downsample_x16 | upsample_x4 | upsample_x8 |
|---|---|---|---|---|---|---|
| original | 306 | 440 | 47 | 151 | 46 | 10 |
| JPEG_Q80 | 85 | 713 | 45 | 87 | 35 | 35 |
| downsample_x8 | 137 | 87 | 167 | 578 | 24 | 7 |
| downsample_x16 | 91 | 23 | 112 | 771 | 3 | 0 |
| upsample_x4 | 0 | 3 | 32 | 0 | 958 | 7 |
| upsample_x8 | 0 | 0 | 25 | 0 | 3 | 972 |

### CNN · size=32

| true \\ pred | original | JPEG_Q80 | downsample_x8 | downsample_x16 | upsample_x4 | upsample_x8 |
|---|---|---|---|---|---|---|
| original | 755 | 176 | 49 | 16 | 4 | 0 |
| JPEG_Q80 | 125 | 790 | 25 | 11 | 15 | 34 |
| downsample_x8 | 137 | 116 | 432 | 305 | 6 | 4 |
| downsample_x16 | 93 | 74 | 488 | 341 | 4 | 0 |
| upsample_x4 | 35 | 3 | 0 | 0 | 959 | 3 |
| upsample_x8 | 51 | 6 | 0 | 1 | 2 | 940 |

### CNN · size=64

| true \\ pred | original | JPEG_Q80 | downsample_x8 | downsample_x16 | upsample_x4 | upsample_x8 |
|---|---|---|---|---|---|---|
| original | 897 | 57 | 7 | 16 | 1 | 22 |
| JPEG_Q80 | 109 | 828 | 7 | 10 | 0 | 46 |
| downsample_x8 | 125 | 28 | 147 | 698 | 1 | 1 |
| downsample_x16 | 68 | 13 | 97 | 821 | 0 | 1 |
| upsample_x4 | 1 | 15 | 0 | 0 | 955 | 29 |
| upsample_x8 | 0 | 1 | 0 | 0 | 2 | 997 |

### CNN · size=96

| true \\ pred | original | JPEG_Q80 | downsample_x8 | downsample_x16 | upsample_x4 | upsample_x8 |
|---|---|---|---|---|---|---|
| original | 771 | 194 | 10 | 3 | 20 | 2 |
| JPEG_Q80 | 37 | 928 | 6 | 1 | 20 | 8 |
| downsample_x8 | 58 | 51 | 420 | 451 | 10 | 10 |
| downsample_x16 | 19 | 29 | 342 | 605 | 3 | 2 |
| upsample_x4 | 0 | 0 | 0 | 0 | 996 | 4 |
| upsample_x8 | 0 | 0 | 0 | 0 | 35 | 965 |

### CNN · size=128

| true \\ pred | original | JPEG_Q80 | downsample_x8 | downsample_x16 | upsample_x4 | upsample_x8 |
|---|---|---|---|---|---|---|
| original | 948 | 22 | 12 | 3 | 2 | 13 |
| JPEG_Q80 | 72 | 866 | 16 | 3 | 4 | 39 |
| downsample_x8 | 19 | 5 | 447 | 523 | 4 | 2 |
| downsample_x16 | 0 | 3 | 255 | 741 | 1 | 0 |
| upsample_x4 | 1 | 0 | 0 | 0 | 966 | 33 |
| upsample_x8 | 0 | 1 | 0 | 0 | 0 | 999 |

