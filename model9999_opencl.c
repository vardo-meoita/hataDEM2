/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 *
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek, Jozef Stefan Institute (modifications and corrections)
 *
 * PURPOSE:      OpenCL-accelerated path loss computation for Ericsson model
 *               9999.  Offloads the entire per-pixel loop to a GPU (or an
 *               OpenCL CPU device) via a single 2-D NDRange kernel.
 *
 *               Top-level path loss formula (kernel mirrors model9999.c):
 *                 Lb = H_OA + m_k + sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )
 *
 *               where:
 *                 H_OA  - Okamura-Hata open-area path loss
 *                 m_k   - per-pixel land-use clutter correction (dB)
 *                 K_DFR - knife-edge diffraction (ITU-R P.526-16 Sec.4.1,
 *                         equations 26 and 31)
 *                 J_DFR - spherical Earth diffraction (ITU-R P.526-16 Sec.3.2,
 *                         full 6-step procedure + Sec.3.1.1.2 sub-model)
 *                 ALPHA - 1.0 (hard-coded per Ericsson model 9999 spec)
 *
 *               The DEM and clutter-correction rasters are streamed to the
 *               device in configurable-size batches to avoid exhausting
 *               host-pinned or device-visible memory on large maps.
 *
 * COPYRIGHT:    (C) 2009-2018 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#ifdef USE_OPENCL

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <grass/gis.h>
#include <grass/raster.h>
#include <grass/glocale.h>
#include "local_proto.h"

/* Controls how many bytes are transferred to / from the OpenCL device per
 * clEnqueueWriteBuffer / clEnqueueReadBuffer call.  4 MiB keeps individual
 * DMA transfers small enough to avoid bus-stall timeouts on slow PCIe links
 * while large enough to amortise per-call overhead on all commodity GPUs. */
#define STREAM_BATCH_BYTES (4 * 1024 * 1024)

/* OpenCL kernel source string
 *
 * The kernel computes the Ericsson model 9999 path loss for every pixel of
 * the output raster in parallel.  Each work item processes one (col, row)
 * pair:  get_global_id(0) -> column, get_global_id(1) -> row.
 *
 * Sub-models implemented inside the kernel (single-precision float):
 *   - Okamura-Hata open-area loss H_OA
 *   - Bresenham line walk to find the dominant terrain obstacle
 *   - Knife-edge diffraction K_DFR  (ITU-R P.526-16 Sec.4.1, eq. 26 & 31)
 *   - Spherical Earth diffraction J_DFR (ITU-R P.526-16 Sec.3.2,
 *     6-step procedure; Sec.3.1.1.2 beta=1 inner model) */
static const char *model9999_kernel_src =
    "/* M_PI_F: some OpenCL implementations do not expose M_PI in device  */\n"
    "/* code; define it explicitly to ensure portability.                 */\n"
    "#define M_PI_F           3.14159265358979323846f\n"
    "\n"
    "/* ALPHA scales K_DFR in the quadratic diffraction combination:      */\n"
    "/*   sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )                            */\n"
    "/* Hard-coded to 1.0 per the Ericsson model 9999 specification.      */\n"
    "#define ALPHA_F              1.0f\n"
    "\n"
    "/* Minimum BS-to-MS distance.  The Okamura-Hata formula contains     */\n"
    "/* log10(d) and is undefined at d = 0; any distance below this value */\n"
    "/* is clamped before the model is evaluated.                          */\n"
    "#define MIN_DISTANCE_KM_F    0.02f\n"
    "\n"
    "/* Exact SI speed of light (m/s), used to derive signal wavelength.  */\n"
    "#define SPEED_OF_LIGHT_F     299792458.0f\n"
    "\n"
    "/* Effective Earth radius ae = 6371 * (4/3) km, standard atmosphere. */\n"
    "#define AE_KM_F              8494.6666666f\n"
    "#define AE_M_F               8494666.6666f\n"
    "\n"

    /* DEM array index safety clamp
     *
     * Prevents out-of-bounds access when the Bresenham walk approaches
     * the raster boundary.  In practice the walk stays within valid
     * coordinates because both endpoints are valid pixels; this is a
     * last-resort guard against floating-point rounding in coordinate
     * conversion or pathological caller inputs. */
    "int clamp_idx(int v, int limit)\n"
    "{\n"
    "    if (v < 0)      return 0;\n"
    "    if (v >= limit) return limit - 1;\n"
    "    return v;\n"
    "}\n"
    "\n"

    /* Walks the DEM from (bs_row, bs_col) to (ms_row, ms_col) using
     * Bresenham's integer line algorithm.  For each pixel the terrain
     * height above the straight line-of-sight (LOS) chord that connects
     * z_bs_trans to z_ms_trans is computed.  Returns the maximum height
     * above LOS found anywhere along the path, and sets *out_dist_bs_m
     * to the Euclidean distance (metres) from the BS to that pixel.
     *
     * NaN DEM pixels are skipped (treated as no obstruction).
     *
     * The loop iteration count is capped at max_steps to prevent runaway
     * loops if the input coordinates are inconsistent. */
    "static float bresenham_dominant_obstacle(\n"
    "    __global const float *dem, int ncols, int nrows,\n"
    "    int bs_row, int bs_col, int ms_row, int ms_col,\n"
    "    float z_bs_trans, float z_ms_trans, float scale,\n"
    "    float d_total_m, float *out_dist_bs_m, int max_steps)\n"
    "{\n"
    "    int dcol    = abs(ms_col - bs_col);\n"
    "    int drow    = abs(ms_row - bs_row);\n"
    "    int scol    = (ms_col >= bs_col) ? 1 : -1;\n"
    "    int srow    = (ms_row >= bs_row) ? 1 : -1;\n"
    "    int err     = dcol - drow;\n"
    "\n"
    "    /* Bresenham visits exactly max(dcol, drow) + 1 pixels.          */\n"
    "    int n_total = (dcol > drow ? dcol : drow) + 1;\n"
    "    if (n_total > max_steps) n_total = max_steps;\n"
    "\n"
    "    int   row   = bs_row;\n"
    "    int   col   = bs_col;\n"
    "    float max_h = -1.0e30f;\n"
    "    *out_dist_bs_m = 0.0f;\n"
    "\n"
    "    for (int n = 0; n < n_total; n++) {\n"
    "        /* Euclidean distance from the BS pixel centre to this pixel. */\n"
    "        float dc   = (float)(col - bs_col);\n"
    "        float dr   = (float)(row - bs_row);\n"
    "        float dist = sqrt(dc * dc + dr * dr) * scale;\n"
    "\n"
    "        /* Normalised position along the path: 0 at BS, 1 at MS.     */\n"
    "        float t     = (d_total_m > 0.0f) ? dist / d_total_m : 0.0f;\n"
    "\n"
    "        /* Height of the LOS chord at this pixel.                     */\n"
    "        float los_h = z_bs_trans + (z_ms_trans - z_bs_trans) * t;\n"
    "\n"
    "        float dem_h = dem[clamp_idx(row, nrows) * ncols\n"
    "                        + clamp_idx(col, ncols)];\n"
    "\n"
    "        /* Skip NaN terrain pixels (GRASS null cells).                */\n"
    "        if (!isnan(dem_h)) {\n"
    "            float h_above = dem_h - los_h;\n"
    "            if (h_above > max_h) {\n"
    "                max_h          = h_above;\n"
    "                *out_dist_bs_m = dist;\n"
    "            }\n"
    "        }\n"
    "\n"
    "        if (row == ms_row && col == ms_col) break;\n"
    "\n"
    "        /* Advance to next pixel.  When e2 satisfies both conditions  */\n"
    "        /* simultaneously the algorithm takes a diagonal step.        */\n"
    "        int e2 = 2 * err;\n"
    "        if (e2 > -drow) { err -= drow; col += scol; }\n"
    "        if (e2 <  dcol) { err += dcol; row += srow; }\n"
    "    }\n"
    "    return max_h;\n"
    "}\n"
    "\n"

    /* ITU-R P.526-16 3.1.1.2 helper functions (beta = 1)
     *
     * These functions implement the smooth-Earth diffraction sub-model.
     * The beta = 1 simplification is valid for horizontal polarisation
     * at any frequency, and for vertical polarisation above 20 MHz over
     * land, because in those regimes the surface admittance K << 0.001
     * and has no influence on the result (P.526-16, 3.1.1.1). */

    "/* ITU-R P.526-16, eq. (14a) with beta = 1:                          */\n"
    "/*   X = 2.188 * f^(1/3) * ae^(-2/3) * d                            */\n"
    "static float normalise_distance_f(float d_km, float ae_km, float f_MHz)\n"
    "{\n"
    "    return 2.188f\n"
    "         * pow(f_MHz, 1.0f / 3.0f)\n"
    "         * pow(ae_km, -2.0f / 3.0f)\n"
    "         * d_km;\n"
    "}\n"
    "\n"

    "/* ITU-R P.526-16, eq. (15a) with beta = 1:                          */\n"
    "/*   Y = 9.575e-3 * f^(2/3) * ae^(-1/3) * h                         */\n"
    "static float normalise_height_f(float h_m, float ae_km, float f_MHz)\n"
    "{\n"
    "    return 9.575e-3f\n"
    "         * pow(f_MHz, 2.0f / 3.0f)\n"
    "         * pow(ae_km, -1.0f / 3.0f)\n"
    "         * h_m;\n"
    "}\n"
    "\n"

    "/* ITU-R P.526-16, eqs. (17a) and (17b):                             */\n"
    "/*   F(X) = 11 + 10*log10(X) - 17.6*X         for X >= 1.6  (17a)  */\n"
    "/*   F(X) = -20*log10(X) - 5.6488*X^1.425     for X <  1.6  (17b)  */\n"
    "static float distance_function_F(float X)\n"
    "{\n"
    "    if (X >= 1.6f)\n"
    "        return 11.0f + 10.0f * log10(X) - 17.6f * X;\n"
    "    else\n"
    "        return -20.0f * log10(X) - 5.6488f * pow(X, 1.425f);\n"
    "}\n"
    "\n"

    "/* ITU-R P.526-16, eqs. (18) and (18a), B = beta*Y = Y (beta=1):    */\n"
    "/*   G(Y) = 17.6*(B-1.1)^0.5 - 5*log10(B-1.1) - 8   for B >  2    */\n"
    "/*   G(Y) = 20*log10(B + 0.1*B^3)                    for B <= 2    */\n"
    "/*                                                                     */\n"
    "/* The fmax guard on (B - 1.1) prevents sqrt() and log10() from      */\n"
    "/* receiving a non-positive argument due to floating-point rounding   */\n"
    "/* near the branch boundary at B = 2.                                 */\n"
    "static float height_gain_function_G(float Y)\n"
    "{\n"
    "    float B = Y; /* beta = 1 */\n"
    "    if (B > 2.0f) {\n"
    "        float Bm11 = fmax(B - 1.1f, 1e-30f);\n"
    "        return 17.6f * sqrt(Bm11)\n"
    "             - 5.0f  * log10(Bm11)\n"
    "             - 8.0f;\n"
    "    } else {\n"
    "        return 20.0f * log10(B + 0.1f * B * B * B);\n"
    "    }\n"
    "}\n"
    "\n"

    "/* ITU-R P.526-16, 3.1.1.2 master equation (13):                    */\n"
    "/*   20*log10(E/E0) = F(X) + G(Y1) + G(Y2)                           */\n"
    "/*                                                                     */\n"
    "/* Returns Ah = -(F(X) + G(Y1) + G(Y2)) in dB.  Positive Ah means   */\n"
    "/* diffraction loss; negative Ah means the formula yields gain (the   */\n"
    "/* caller must treat that as zero loss).                               */\n"
    "/*                                                                     */\n"
    "/* ae_km is supplied explicitly so this helper can be called with     */\n"
    "/* either the standard effective Earth radius AE_KM_F (over-horizon   */\n"
    "/* step, 3.2 step 1) or the modified radius aem (interpolation step, */\n"
    "/* 3.2 step 5).                                                       */\n"
    "static float apply_section_311_f(float d_km, float h1_m, float h2_m,\n"
    "                                  float f_MHz, float ae_km)\n"
    "{\n"
    "    float X  = normalise_distance_f(d_km, ae_km, f_MHz);\n"
    "    float Y1 = normalise_height_f(h1_m, ae_km, f_MHz);\n"
    "    float Y2 = normalise_height_f(h2_m, ae_km, f_MHz);\n"
    "    return -(distance_function_F(X)\n"
    "           + height_gain_function_G(Y1)\n"
    "           + height_gain_function_G(Y2));\n"
    "}\n"
    "\n"

    /* ITU-R P.526-16 3.2 — six-step procedure for diffraction loss at
     * 10 MHz and above.  All geometry steps work in SI units (metres);
     * the 3.1.1 sub-model call uses practical units (km / m / MHz) as
     * required by eqs. (14a) and (15a).
     *
     * Step 1 (eq. 21) — marginal LoS distance d_los.
     * Step 2 (eqs. 22-22e) — smallest clearance height h.
     * Step 3 (eq. 23) — required clearance h_req for zero loss.
     * Step 4 (eq. 24) — modified effective Earth radius aem.
     * Step 5          — 3.1.1 with aem to obtain Ah.
     * Step 6 (eq. 25) — interpolated loss A = (1 - h/h_req) * Ah. */
    "static float spherical_earth_diffraction_f(\n"
    "    float d_km, float h_t_m, float h_r_m, float f_MHz)\n"
    "{\n"
    "    float ae_m     = AE_M_F;\n"
    "    float d_m      = d_km * 1000.0f;\n"
    "    float lambda_m = SPEED_OF_LIGHT_F / (f_MHz * 1.0e6f);\n"
    "\n"
    "    /* Step 1 (eq. 21): marginal line-of-sight distance.             */\n"
    "    /* d_los = 2 * sqrt(ae) * (sqrt(h_t) + sqrt(h_r))               */\n"
    "    /* If d >= d_los the path is over the radio horizon; apply       */\n"
    "    /* 3.1.1 directly with the standard effective Earth radius.     */\n"
    "    float sqrt_ht = sqrt(h_t_m);\n"
    "    float sqrt_hr = sqrt(h_r_m);\n"
    "    float d_los   = 2.0f * sqrt(ae_m) * (sqrt_ht + sqrt_hr);\n"
    "\n"
    "    if (d_m >= d_los)\n"
    "        return apply_section_311_f(d_km, h_t_m, h_r_m, f_MHz, AE_KM_F);\n"
    "\n"
    "    /* Step 2 (eqs. 22-22e): smallest clearance height h.            */\n"
    "    /* All quantities in SI units (metres).                           */\n"
    "    float m = (d_m * d_m) / (4.0f * ae_m * (h_t_m + h_r_m));\n"
    "    float c = (h_t_m - h_r_m) / (h_t_m + h_r_m);\n"
    "    float b;\n"
    "    if (m < 1.0e-10f) {\n"
    "        /* For very short paths m -> 0 the general expression for b  */\n"
    "        /* has a 0*Inf form.  The limiting value is b = c (verified   */\n"
    "        /* by L'Hopital expansion of eq. 22c).                        */\n"
    "        b = c;\n"
    "    } else {\n"
    "        /* Argument of arccos in eq. (22c).  Clamped to [-1, 1] to   */\n"
    "        /* guard against floating-point rounding outside acos domain. */\n"
    "        float q = 3.0f * c * sqrt(3.0f * m)\n"
    "                / (2.0f * pow(m + 1.0f, 1.5f));\n"
    "        if (q >  1.0f) q =  1.0f;\n"
    "        if (q < -1.0f) q = -1.0f;\n"
    "        b = 2.0f * sqrt((m + 1.0f) / (3.0f * m))\n"
    "          * cos(M_PI_F / 3.0f + acos(q) / 3.0f);\n"
    "    }\n"
    "    float d1 = 0.5f * d_m * (1.0f + b);  /* eq. (22a) */\n"
    "    float d2 = d_m - d1;                  /* eq. (22b) */\n"
    "    /* eq. (22): clearance height h. */\n"
    "    float h  = ((h_t_m - d1 * d1 / (2.0f * ae_m)) * d2\n"
    "              + (h_r_m - d2 * d2 / (2.0f * ae_m)) * d1) / d_m;\n"
    "\n"
    "    /* Step 3 (eq. 23): required clearance for zero diffraction loss. */\n"
    "    /* h_req = 0.552 * sqrt(d1 * d2 * lambda / d)                    */\n"
    "    float h_req = 0.552f * sqrt(d1 * d2 * lambda_m / d_m);\n"
    "    if (h > h_req)\n"
    "        return 0.0f;\n"
    "\n"
    "    /* Step 4 (eq. 24): modified effective Earth radius aem.          */\n"
    "    /* aem = 0.5 * (d / (sqrt(h_t) + sqrt(h_r)))^2                   */\n"
    "    float sqrt_sum = sqrt_ht + sqrt_hr;\n"
    "    float aem_m    = 0.5f * (d_m / sqrt_sum) * (d_m / sqrt_sum);\n"
    "\n"
    "    /* Step 5: 3.1.1 with aem.  Negative Ah means gain, which is    */\n"
    "    /* non-physical for a diffraction mechanism; treat as zero loss.  */\n"
    "    float Ah = apply_section_311_f(d_km, h_t_m, h_r_m, f_MHz,\n"
    "                                    aem_m / 1000.0f);\n"
    "    if (Ah < 0.0f)\n"
    "        return 0.0f;\n"
    "\n"
    "    /* Step 6 (eq. 25): interpolated loss.                            */\n"
    "    /* A = (1 - h / h_req) * Ah                                       */\n"
    "    /* Interpolates between full loss (h=0, grazing incidence on the  */\n"
    "    /* modified Earth) and zero loss (h=h_req, sufficient clearance). */\n"
    "    return (1.0f - h / h_req) * Ah;\n"
    "}\n"
    "\n"

    /* main compute kernel
     *
     * One work item per output raster pixel (col = gid0, row = gid1).
     * Assembles the full Ericsson model 9999 path loss for that pixel:
     *
     *   Lb = H_OA + m_k + sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )
     *
     * where link_correction bundles the frequency and mobile-height terms
     * that are constant across the raster (precomputed on the host). */
    "__kernel void model9999_kernel(\n"
    "    __global const float *dem,\n"
    "    __global const float *mk,\n"
    "    __global       float *path_loss,\n"
    "    int   ncols,\n"
    "    int   nrows,\n"
    "    float bs_row,\n"
    "    float bs_col,\n"
    "    float ant_height_bs,\n"
    "    float ant_height_ms,\n"
    "    float scale,\n"
    "    float A0, float A1, float A2, float A3,\n"
    "    float link_correction,\n"
    "    float freq_mhz,\n"
    "    int   max_steps)\n"
    "{\n"
    "    int gid0 = (int)get_global_id(0); /* column index */\n"
    "    int gid1 = (int)get_global_id(1); /* row index    */\n"
    "\n"
    "    /* Guard: work items outside the raster bounds do nothing.        */\n"
    "    if (gid0 >= ncols || gid1 >= nrows)\n"
    "        return;\n"
    "\n"
    "    int ms_row   = gid1;\n"
    "    int ms_col   = gid0;\n"
    "\n"
    "    /* Round the floating-point BS coordinates to the nearest pixel.  */\n"
    "    int bs_row_i = (int)(bs_row + 0.5f);\n"
    "    int bs_col_i = (int)(bs_col + 0.5f);\n"
    "\n"
    "    /* Terrain elevations (m ASL).  NaN signals a GRASS null cell.    */\n"
    "    float z_bs   = dem[bs_row_i * ncols + bs_col_i];\n"
    "    float z_ms   = dem[gid1     * ncols + gid0];\n"
    "    float mk_val = mk[gid1      * ncols + gid0];\n"
    "\n"
    "    /* Any NaN input -> output null (0.0f is used as the null sentinel */\n"
    "    /* and is decoded back to GRASS null by stream_gpu_to_raster).     */\n"
    "    if (isnan(z_bs) || isnan(z_ms) || isnan(mk_val)) {\n"
    "        path_loss[gid1 * ncols + gid0] = 0.0f;\n"
    "        return;\n"
    "    }\n"
    "\n"
    "    /* BS-to-MS distance (km), clamped to prevent log10(0).            */\n"
    "    float dc   = (float)(ms_col - bs_col_i);\n"
    "    float dr   = (float)(ms_row - bs_row_i);\n"
    "    float d_km = sqrt(dc * dc + dr * dr) * scale / 1000.0f;\n"
    "    if (d_km < MIN_DISTANCE_KM_F)\n"
    "        d_km = MIN_DISTANCE_KM_F;\n"
    "\n"
    "    /* Antenna tip elevations (m ASL).                                  */\n"
    "    float z_bs_trans = z_bs + ant_height_bs;\n"
    "    float z_ms_trans = z_ms + ant_height_ms;\n"
    "\n"
    "    /* H_EBK: model 9999 effective BS antenna height (m).               */\n"
    "    /*                                                                   */\n"
    "    /* H_EBK = ZoTransBS - ZoTransMS                                    */\n"
    "    /*       = (z_bs + ant_height_bs) - (z_ms + ant_height_ms)          */\n"
    "    /*                                                                   */\n"
    "    /* When the mobile is at higher terrain than the BS, H_EBK would    */\n"
    "    /* go negative or very small.  Clamp to ant_height_bs so that       */\n"
    "    /* log10(H_EBK) remains well-defined and physically meaningful.     */\n"
    "    float H_EBK = z_bs_trans - z_ms_trans;\n"
    "    if (H_EBK < ant_height_bs)\n"
    "        H_EBK = ant_height_bs;\n"
    "\n"
    "    /* Okamura-Hata open-area path loss H_OA (dB).                      */\n"
    "    /*                                                                   */\n"
    "    /* H_OA = A0 + A1*log10(d) + A2*log10(H_EBK)                       */\n"
    "    /*           + A3*log10(d)*log10(H_EBK) + link_correction           */\n"
    "    /*                                                                   */\n"
    "    /* link_correction bundles: 44.49*log10(F) - 4.78*(log10(F))^2      */\n"
    "    /*                         - 3.2*(log10(11.75*h_m))^2               */\n"
    "    /* and is precomputed once on the host by calc_hata_link_correction. */\n"
    "    float log_d = log10(d_km);\n"
    "    float log_H = log10(H_EBK);\n"
    "    float H_OA  = A0\n"
    "                + A1 * log_d\n"
    "                + A2 * log_H\n"
    "                + A3 * log_d * log_H\n"
    "                + link_correction;\n"
    "\n"
    "    /* Dominant terrain obstacle: Bresenham walk from BS to MS.         */\n"
    "    float d_total_m  = d_km * 1000.0f;\n"
    "    float d_bs_obs_m = 0.0f;\n"
    "    float h_obs = bresenham_dominant_obstacle(\n"
    "        dem, ncols, nrows,\n"
    "        bs_row_i, bs_col_i, ms_row, ms_col,\n"
    "        z_bs_trans, z_ms_trans, scale,\n"
    "        d_total_m, &d_bs_obs_m, max_steps);\n"
    "\n"
    "    /* Knife-edge diffraction loss K_DFR (dB).                          */\n"
    "    /*                                                                   */\n"
    "    /* Fresnel-Kirchhoff parameter nu (ITU-R P.526-16, eq. 26):         */\n"
    "    /*   nu = h * sqrt( 2*(d1+d2) / (lambda * d1 * d2) )               */\n"
    "    /* where d1 = d_bs_obs_m, d2 = d_ms_obs_m.                          */\n"
    "    /*                                                                   */\n"
    "    /* If the obstacle lies at or beyond one terminal, the geometry is  */\n"
    "    /* degenerate; nu is set below the -0.78 threshold so K_DFR = 0.   */\n"
    "    /*                                                                   */\n"
    "    /* Loss J(nu) (ITU-R P.526-16, eq. 31), valid for nu > -0.78:      */\n"
    "    /*   J(nu) = 6.9 + 20*log10( sqrt((nu-0.1)^2 + 1) + nu - 0.1 )    */\n"
    "    float lambda_m   = SPEED_OF_LIGHT_F / (freq_mhz * 1.0e6f);\n"
    "    float d_ms_obs_m = d_total_m - d_bs_obs_m;\n"
    "    float nu;\n"
    "    if (d_bs_obs_m <= 0.0f || d_ms_obs_m <= 0.0f) {\n"
    "        nu = -1.0f; /* degenerate geometry -> no diffraction penalty */\n"
    "    } else {\n"
    "        nu = h_obs * sqrt(2.0f * (d_bs_obs_m + d_ms_obs_m)\n"
    "                         / (lambda_m * d_bs_obs_m * d_ms_obs_m));\n"
    "    }\n"
    "    float K_DFR;\n"
    "    if (nu <= -0.78f) {\n"
    "        K_DFR = 0.0f;\n"
    "    } else {\n"
    "        float x = nu - 0.1f;\n"
    "        K_DFR = 6.9f + 20.0f * log10(sqrt(x * x + 1.0f) + x);\n"
    "    }\n"
    "\n"
    "    /* Spherical Earth diffraction loss J_DFR (dB).                     */\n"
    "    /*                                                                   */\n"
    "    /* H_EBK is used as the transmitter height rather than the physical  */\n"
    "    /* mast height ant_height_bs.  H_EBK already accounts for the       */\n"
    "    /* terrain elevation difference between the BS and MS sites.  Using  */\n"
    "    /* only the mast height would ignore a BS sited on a hill above the  */\n"
    "    /* MS terrain and produce spurious over-horizon diffraction loss.    */\n"
    "    float J_DFR = spherical_earth_diffraction_f(\n"
    "        d_km, H_EBK, ant_height_ms, freq_mhz);\n"
    "\n"
    "    /* Assemble model 9999 path loss.                                   */\n"
    "    /*                                                                   */\n"
    "    /* Lb = H_OA + m_k + sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )          */\n"
    "    /*                                                                   */\n"
    "    /* hypot() computes sqrt(a^2+b^2) with protection against           */\n"
    "    /* intermediate overflow for individually large loss values.         */\n"
    "    float Lb = H_OA + mk_val + hypot(ALPHA_F * K_DFR, J_DFR);\n"
    "\n"
    "    path_loss[gid1 * ncols + gid0] = Lb;\n"
    "}\n";


/* Reads a sub-region [row_min .. row_min+sub_nrows-1,
 *                     col_min .. col_min+sub_ncols-1]
 * of an open GRASS FCELL raster file (identified by file descriptor infd)
 * and uploads it to the OpenCL buffer gpu_buf in STREAM_BATCH_BYTES-sized
 * chunks.  The full row width full_ncols is needed because
 * Rast_allocate_f_buf() / Rast_get_f_row() always operate on complete rows.
 *
 * Returns 0 on success, or the (negative) OpenCL error code on failure. */
static int stream_raster_to_gpu(cl_command_queue queue, cl_mem gpu_buf,
                                int infd, int full_ncols, int row_min,
                                int col_min, int sub_nrows, int sub_ncols)
{
    float *slab;
    FCELL *full_row;
    int batch;
    int r;
    cl_int err;

    /* Suppress "unused parameter" warning: full_ncols is implicit in the
     * allocation and behaviour of Rast_allocate_f_buf(). */
    (void)full_ncols;

    /* How many complete rows fit in one STREAM_BATCH_BYTES chunk? */
    batch = STREAM_BATCH_BYTES / (int)(sub_ncols * sizeof(float));
    if (batch < 1)
        batch = 1;
    if (batch > sub_nrows)
        batch = sub_nrows;

    slab = (float *)G_malloc((size_t)batch * (size_t)sub_ncols * sizeof(float));
    full_row = Rast_allocate_f_buf();

    r = 0;
    while (r < sub_nrows) {
        int rows_this = batch;
        int i;

        if (r + rows_this > sub_nrows)
            rows_this = sub_nrows - r;

        /* Fill the staging slab: read each complete row then copy the
         * [col_min .. col_min+sub_ncols-1] window into the slab. */
        for (i = 0; i < rows_this; i++) {
            Rast_get_f_row(infd, full_row, row_min + r + i);
            memcpy(slab + (size_t)i * (size_t)sub_ncols, full_row + col_min,
                   (size_t)sub_ncols * sizeof(FCELL));
            /* FCELL == float, so the memcpy is type-safe. GRASS null
             * cells are stored as NaN in FCELL buffers; they are
             * preserved here and handled in the kernel via isnan(). */
        }

        err = clEnqueueWriteBuffer(
            queue, gpu_buf, CL_TRUE, /* blocking */
            (size_t)r * (size_t)sub_ncols * sizeof(float),
            (size_t)rows_this * (size_t)sub_ncols * sizeof(float), slab, 0,
            NULL, NULL);

        if (err != CL_SUCCESS) {
            G_free(slab);
            G_free(full_row);
            return (int)err;
        }

        r += rows_this;
    }

    G_free(slab);
    G_free(full_row);
    return 0;
}

/* Reads the path-loss output buffer gpu_buf back from the OpenCL device
 * and writes it row by row to the open GRASS FCELL raster file outfd.
 *
 * Cells where the kernel wrote 0.0f (the null sentinel used for NaN DEM
 * pixels, or for pixels that are outside the computation window) are
 * converted back to GRASS null values.  Cells that are NaN in the GPU
 * buffer (which should not occur in normal operation but may result from
 * exceptional floating-point conditions in the kernel) are also nulled.
 *
 * Transfer is performed in STREAM_BATCH_BYTES-sized chunks to limit the
 * amount of host memory pinned at any one time.
 *
 * Returns 0 on success, or the (negative) OpenCL error code on failure. */
static int stream_gpu_to_raster(cl_command_queue queue, cl_mem gpu_buf,
                                int outfd, int nrows, int ncols)
{
    FCELL *slab;
    FCELL *full_row;
    int batch;
    int r;
    cl_int err;

    batch = STREAM_BATCH_BYTES / (ncols * (int)sizeof(float));
    if (batch < 1)
        batch = 1;
    if (batch > nrows)
        batch = nrows;

    slab = (FCELL *)G_malloc((size_t)batch * (size_t)ncols * sizeof(FCELL));
    full_row = (FCELL *)G_malloc((size_t)ncols * sizeof(FCELL));

    r = 0;
    while (r < nrows) {
        int rows_this = batch;
        int i, j;

        if (r + rows_this > nrows)
            rows_this = nrows - r;

        /* Blocking read: wait for the device to fill the slab before
         * we inspect and write the values to disk. */
        err = clEnqueueReadBuffer(queue, gpu_buf, CL_TRUE, /* blocking */
                                  (size_t)r * (size_t)ncols * sizeof(FCELL),
                                  (size_t)rows_this * (size_t)ncols *
                                      sizeof(FCELL),
                                  slab, 0, NULL, NULL);

        if (err != CL_SUCCESS) {
            G_free(slab);
            G_free(full_row);
            return (int)err;
        }

        for (i = 0; i < rows_this; i++) {
            for (j = 0; j < ncols; j++) {
                float val = (float)slab[(size_t)i * (size_t)ncols + j];

                /* 0.0f is the null sentinel written by the kernel for
                 * NaN DEM / m_k input pixels.  An unexpected NaN in the
                 * output is also mapped to GRASS null for safety. */
                if (val == 0.0f || isnan(val)) {
                    Rast_set_f_null_value(&full_row[j], 1);
                }
                else {
                    full_row[j] = (FCELL)val;
                }
            }
            Rast_put_f_row(outfd, full_row);
        }

        r += rows_this;
    }

    G_free(slab);
    G_free(full_row);
    return 0;
}

/* Top-level host function.  Orchestrates the full OpenCL pipeline:
 *   1. Platform and device discovery (GPU preferred, CPU fallback)
 *   2. Context and command-queue creation
 *   3. Kernel compilation (with build log on failure)
 *   4. DEM and m_k raster upload to device
 *   5. Zero-initialised output buffer allocation
 *   6. 2-D NDRange kernel launch (16x16 work-groups)
 *   7. Device synchronisation
 *   8. Output readback and GRASS raster write
 *   9. Resource cleanup via goto-pattern
 *
 * Returns 0 on success, -1 on any error.
 * Errors are reported with G_warning() so that main() can decide how to
 * proceed (typically by falling back to the CPU implementation). */
int calc_model9999_path_loss_ocl(int infd_dem, int infd_mk, int outfd,
                                 int nrows, int ncols, int bs_row, int bs_col,
                                 const struct Model9999Params *params)
{
    /* OpenCL runtime objects.  All set to NULL / 0 so the cleanup block
     * can call clRelease*() safely even when creation failed partway
     * through the initialisation sequence. */
    cl_int err = CL_SUCCESS;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_mem dem_buf = NULL;
    cl_mem mk_buf = NULL;
    cl_mem out_buf = NULL;
    int ret = 0;

    /* Scalar kernel arguments (double -> float down-conversion).
     * Declared here so they are in scope at the goto label. */
    float bs_row_f, bs_col_f;
    float ant_height_bs_f, ant_height_ms_f;
    float scale_f;
    float A0_f, A1_f, A2_f, A3_f;
    float link_correction_f;
    float freq_mhz_f;
    int max_steps;
    int arg;

    /* NDRange geometry (column-major: dim 0 = cols, dim 1 = rows). */
    size_t local_work_size[2] = {16, 16};
    size_t global_work_size[2];

    /* Size of one raster plane on the device (bytes). */
    size_t buf_size = (size_t)nrows * (size_t)ncols * sizeof(float);

    cl_uint num_platforms = 0;

    /* Step 1: Platform discovery */
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        G_warning(_("OpenCL: no platforms found (error %d)."), (int)err);
        return -1;
    }

    /* Step 2: Device selection — GPU preferred, CPU fallback */
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        /* No GPU available on this platform; try any CPU OpenCL device. */
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id,
                             NULL);
        if (err != CL_SUCCESS) {
            G_warning(_("OpenCL: no GPU or CPU device found (error %d)."),
                      (int)err);
            return -1;
        }
        G_message(_("OpenCL: GPU not available; falling back to CPU device."));
    }

    /* Step 3: Create context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clCreateContext failed (error %d)."), (int)err);
        ret = -1;
        goto cleanup;
    }

     /* Step 4: Create command queue */
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clCreateCommandQueue failed (error %d)."),
                  (int)err);
        ret = -1;
        goto cleanup;
    }

    /* Step 5: Create and build program */
    program = clCreateProgramWithSource(context, 1, &model9999_kernel_src, NULL,
                                        &err);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clCreateProgramWithSource failed (error %d)."),
                  (int)err);
        ret = -1;
        goto cleanup;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        /* Retrieve and print the compiler's build log to help diagnose
         * kernel source errors. */
        size_t log_size = 0;
        char *build_log;

        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                              &log_size);
        build_log = (char *)G_malloc(log_size + 1);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                              log_size, build_log, NULL);
        build_log[log_size] = '\0';
        G_warning(_("OpenCL: clBuildProgram failed (error %d):\n%s"), (int)err,
                  build_log);
        G_free(build_log);
        ret = -1;
        goto cleanup;
    }

    /* Step 6: Create kernel object */
    kernel = clCreateKernel(program, "model9999_kernel", &err);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clCreateKernel failed (error %d)."), (int)err);
        ret = -1;
        goto cleanup;
    }

    /* Step 7: Allocate DEM buffer and upload from GRASS raster */
    dem_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, buf_size, NULL, &err);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clCreateBuffer (DEM) failed (error %d)."),
                  (int)err);
        ret = -1;
        goto cleanup;
    }
    {
        int rc = stream_raster_to_gpu(queue, dem_buf, infd_dem, ncols, 0, 0,
                                      nrows, ncols);
        if (rc != 0) {
            G_warning(_("OpenCL: DEM upload to device failed (error %d)."), rc);
            ret = -1;
            goto cleanup;
        }
    }

    /* Step 8: Allocate m_k buffer and upload from GRASS raster */
    mk_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, buf_size, NULL, &err);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clCreateBuffer (m_k) failed (error %d)."),
                  (int)err);
        ret = -1;
        goto cleanup;
    }
    {
        int rc = stream_raster_to_gpu(queue, mk_buf, infd_mk, ncols, 0, 0,
                                      nrows, ncols);
        if (rc != 0) {
            G_warning(_("OpenCL: m_k upload to device failed (error %d)."), rc);
            ret = -1;
            goto cleanup;
        }
    }

    /* Step 9: Allocate output buffer, zero-fill via clEnqueueFillBuffer
     *
     * Zero is also the null sentinel value: the kernel writes 0.0f for
     * any pixel with NaN DEM or m_k input.  The fill ensures that pixels
     * outside the active 2-D NDRange (padding to a multiple of 16) are
     * harmlessly zero rather than uninitialised. */
    out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, buf_size, NULL, &err);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clCreateBuffer (output) failed (error %d)."),
                  (int)err);
        ret = -1;
        goto cleanup;
    }
    {
        float zero = 0.0f;
        err = clEnqueueFillBuffer(queue, out_buf, &zero, sizeof(float), 0,
                                  buf_size, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            G_warning(_("OpenCL: clEnqueueFillBuffer failed (error %d)."),
                      (int)err);
            ret = -1;
            goto cleanup;
        }
    }

    /* Step 10: Precompute scalar kernel arguments
     *
     * All double-precision model parameters are down-converted to float
     * here once, before being passed as kernel arguments.  The conversion
     * is acceptable because the model coefficients and antenna heights are
     * typically specified to fewer significant digits than float provides. */
    bs_row_f = (float)bs_row;
    bs_col_f = (float)bs_col;
    ant_height_bs_f = (float)params->ant_height_bs_m;
    ant_height_ms_f = (float)params->ant_height_ms_m;
    scale_f = (float)params->scale_m;
    A0_f = (float)params->A0;
    A1_f = (float)params->A1;
    A2_f = (float)params->A2;
    A3_f = (float)params->A3;
    link_correction_f = (float)params->link_correction;
    freq_mhz_f = (float)params->freq_mhz;

    /* max_steps caps the Bresenham walk to the diagonal of the raster
     * plus a small margin.  This is always >= the true walk length for
     * any valid BS/MS pair within the raster. */
    max_steps = (int)ceil(sqrt((double)nrows * (double)nrows +
                               (double)ncols * (double)ncols)) +
                2;

    /* Step 11: Set all 17 kernel arguments
     *
     * Arguments match the kernel signature in order:
     *   0  dem          (__global const float *)
     *   1  mk           (__global const float *)
     *   2  path_loss    (__global float *)
     *   3  ncols        (int)
     *   4  nrows        (int)
     *   5  bs_row       (float)
     *   6  bs_col       (float)
     *   7  ant_height_bs (float)
     *   8  ant_height_ms (float)
     *   9  scale        (float)
     *   10 A0           (float)
     *   11 A1           (float)
     *   12 A2           (float)
     *   13 A3           (float)
     *   14 link_correction (float)
     *   15 freq_mhz     (float)
     *   16 max_steps    (int) */
    arg = 0;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dem_buf);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &mk_buf);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &out_buf);
    err |= clSetKernelArg(kernel, arg++, sizeof(int), &ncols);
    err |= clSetKernelArg(kernel, arg++, sizeof(int), &nrows);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &bs_row_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &bs_col_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &ant_height_bs_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &ant_height_ms_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &scale_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &A0_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &A1_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &A2_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &A3_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &link_correction_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(float), &freq_mhz_f);
    err |= clSetKernelArg(kernel, arg++, sizeof(int), &max_steps);

    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clSetKernelArg failed (error %d)."), (int)err);
        ret = -1;
        goto cleanup;
    }

    /* Step 12: Launch 2-D NDRange kernel
     *
     * Work-group size is 16x16 = 256 threads, a safe choice that fits
     * within the minimum guaranteed CL_DEVICE_MAX_WORK_GROUP_SIZE of 256
     * for OpenCL 1.2 devices.  The global size is padded to the next
     * multiple of 16 in both dimensions; out-of-bounds work items are
     * discarded by the bounds guard at the top of the kernel. */
    global_work_size[0] = ((size_t)ncols + 15u) / 16u * 16u;
    global_work_size[1] = ((size_t)nrows + 15u) / 16u * 16u;

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clEnqueueNDRangeKernel failed (error %d)."),
                  (int)err);
        ret = -1;
        goto cleanup;
    }

    /* Step 13: Wait for kernel completion */
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        G_warning(_("OpenCL: clFinish failed (error %d)."), (int)err);
        ret = -1;
        goto cleanup;
    }

    /* Step 14: Release DEM and m_k buffers
     *
     * Free device memory occupied by the two read-only inputs before
     * reading the larger output buffer back to the host.  This ensures
     * that we never hold three large raster planes simultaneously on a
     * device with limited memory. */
    clReleaseMemObject(dem_buf);
    dem_buf = NULL;
    clReleaseMemObject(mk_buf);
    mk_buf = NULL;

    {
        int rc = stream_gpu_to_raster(queue, out_buf, outfd, nrows, ncols);
        if (rc != 0) {
            G_warning(_("OpenCL: output readback from device failed"
                        " (error %d)."),
                      rc);
            ret = -1;
            goto cleanup;
        }
    }

    /* Step 16: Resource cleanup
     *
     * All OpenCL objects are released in reverse order of creation.
     * NULL checks make the block safe to jump into from any earlier
     * failure point, and re-entrant in case a future refactor introduces
     * multiple cleanup paths. */
cleanup:
    if (out_buf) {
        clReleaseMemObject(out_buf);
        out_buf = NULL;
    }
    if (mk_buf) {
        clReleaseMemObject(mk_buf);
        mk_buf = NULL;
    }
    if (dem_buf) {
        clReleaseMemObject(dem_buf);
        dem_buf = NULL;
    }
    if (kernel) {
        clReleaseKernel(kernel);
        kernel = NULL;
    }
    if (program) {
        clReleaseProgram(program);
        program = NULL;
    }
    if (queue) {
        clReleaseCommandQueue(queue);
        queue = NULL;
    }
    if (context) {
        clReleaseContext(context);
        context = NULL;
    }

    return ret;
}

#endif /* USE_OPENCL */
