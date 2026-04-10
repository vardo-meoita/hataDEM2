## r.hataDEM2 GRASS GIS extension

*r.hataDEM2* calculates a radio path-loss coverage raster from a single base
station (BS) using the Ericsson model 9999 formula. The output raster gives
the path loss in dB at every pixel of the current computational region.

## How to install

```
grass --exec g.extension extension=r.hataDEM2 url=https://github.com/vardo-meoita/hataDEM2
```

Optionally make sure you have an OpenCL SDK and runtime installed, and
instead run the following command to compile an OpenCL-enabled version:

```
env WITH_OPENCL=1 grass --exec g.extension extension=r.hataDEM2 url=https://github.com/vardo-meoita/hataDEM2
```

### Path loss formula

$$
L_b = H_{OA} + m_k + \sqrt{(\alpha \cdot K_{DFR})^2 + J_{DFR}^2}
$$

where $H_{OA}$ is the Okamura-Hata open-area path loss, $m_k$ is a
per-pixel land-use clutter correction supplied as the **m_k** raster,
$K_{DFR}$ is the knife-edge diffraction loss (ITU-R P.526-16, Sec. 4.1),
$J_{DFR}$ is the spherical Earth diffraction loss (ITU-R P.526-16, Sec. 3.2),
and $\alpha = 1$.

### Okamura-Hata open-area formula

$$
H_{OA} = A_0 + A_1 \log_{10}(d)
        + A_2 \log_{10}(H_{EBK})
        + A_3 \log_{10}(d)\,\log_{10}(H_{EBK})
        - 3.2\bigl(\log_{10}(11.75\,h_m)\bigr)^2
        + 44.49\,\log_{10}(F)
        - 4.78\bigl(\log_{10}(F)\bigr)^2
$$

where $d$ is the BS-to-MS distance in km, $H_{EBK}$ is the effective BS
antenna height in m (derived from the DEM and the physical mast heights),
$h_m$ is the mobile antenna height in m (**rx_ant_height**), and $F$ is the
carrier frequency in MHz.

### Tuning parameters

The four tuning parameters **a0** ($A_0$), **a1** ($A_1$), **a2** ($A_2$),
and **a3** ($A_3$) control the shape of the path-loss curve.

**Default values** are taken directly from the TEMS CellPlanner 9.1 Common
Features Technical Reference Manual, Sec. 2.2:

| Parameter | Default | Role |
|---|---|---|
| **a0** | 36.2 | Absolute level — shifts the entire curve up or down |
| **a1** | 30.7 | Distance slope |
| **a2** | -12.0 | Effective antenna height adjustment |
| **a3** | 0.1 | Interaction between distance and effective height |

#### Alternative parameter sets

Simi et al. ("Minimax LS Algorithm for Automatic Propagation Model Tuning")
derive $A_0$ and $A_1$ automatically from field measurements while keeping
$A_2$ and $A_3$ at their default values of -12.0 and 0.1. Their results for
Serbia GSM measurements are reproduced below.

**Open area** (Table 1 in the paper):

| Algorithm | a0 | a1 | STD (dB) | RMS error (dB) | Max error (dB) |
|---|---|---|---|---|---|
| LS | 45.95 | 100.6 | 6.04 | 36.5 | 20.2 |
| IRLS | 47.31 | 101.0 | 6.04 | 38.0 | 18.4 |

**Suburban area** (Table 2 in the paper):

| Algorithm | a0 | a1 | STD (dB) | RMS error (dB) | Max error (dB) |
|---|---|---|---|---|---|
| LS | 43.20 | 68.93 | 4.96 | 24.52 | 17.36 |
| IRLS | 44.17 | 80.55 | 4.96 | 27.1 | 14.56 |

Note that these parameter sets produced unsatisfactory results in the authors'
own testing of *r.hataDEM2* and are provided here only for reference.
Site-specific measurement campaigns are the recommended basis for tuning.

### Clutter compensation

The **m_k** raster supplies a per-pixel land-use clutter correction added
directly to the path loss. Each pixel value is the correction in dB for the
land-use category at the mobile station location.

The TEMS CellPlanner 9.1 Reference Manual gives example $m_k$ values for the
WCDMA band (2050 MHz):

| Land use type | mk (dB) |
|---|---|
| Suburban | 20.6 – 24.2 |
| Urban | 25.6 – 29.2 |
| Marsh | 14.0 |
| Open land | 4.4 |
| Pine forest | 21.9 |
| Half-open | 17.6 |
| Forest | 17.9 |
| Water | 3.4 |
| Village | 15.7 |

The TEMS CellPlanner 9.1 software ships with the following default clutter
table for GSM-900:

| Land use type | mk (dB) |
|---|---|
| Buildings | 14.67 |
| Densely forested | 18.0 |
| Open | 6.0 |
| Open vegetated | 14.0 |
| Sparsely forested | 9.2 |
| Water | 2.0 |

These tables are illustrative starting points. The values vary with frequency,
terrain, and regional morphology. Users are strongly encouraged to derive
their own clutter correction values from empirical path-loss measurements.

## NOTES

- The computation covers the entire current GRASS computational region.
  Use *g.region* to limit the extent before running the module.
- The DEM must have a valid (non-null) value at the base station pixel.
  Pixels where either **input_dem** or **m_k** is null are written as null
  in the output.
- Distances shorter than 0.02 km are clamped to that value.
- All parameters are saved in the output map history (*r.info* **-h**).
- Parallel computation is supported via **nprocs** (OpenMP). An optional
  GPU path is available when the module is built with `WITH_OPENCL=1`.

## EXAMPLES

Compute path loss for a base station at UTM coordinates
(412500 E, 5068000 N), BS antenna height 30 m, using 4 threads:

```sh
r.hataDEM2 input_dem=dem m_k=clutter output=path_loss \
    coordinates=412500,5068000 ant_height=30 rx_ant_height=1.5 \
    frequency=900 nprocs=4
```

Restrict computation to a study area before running:

```sh
g.region vector=study_area
r.hataDEM2 input_dem=dem m_k=clutter output=path_loss \
    coordinates=412500,5068000 ant_height=30
```

## REFERENCES

TEMS CellPlanner 9.1 Common Features Technical Reference
Manual, Sec. 2.2 9999 Propagation Model.

Hata, M. (1980). Empirical formula for propagation loss in land mobile radio
services. *IEEE Transactions on Vehicular Technology*, 29(3), 317-325.

ITU-R Recommendation P.526-16 (2025). Propagation by diffraction.
International Telecommunication Union, Geneva.

## SEE ALSO

*[g.region](g.region.md),
[r.hataDEM](r.hataDEM.md),
[r.info](r.info.md)*

## AUTHORS

tifil

Andrej Vilhar, Jozef Stefan Institute

Tomaz Javornik, Jozef Stefan Institute

Andrej Hrovat, Jozef Stefan Institute

Igor Ozimek, Jozef Stefan Institute
