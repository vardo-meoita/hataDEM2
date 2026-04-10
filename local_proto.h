/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek, Jozef Stefan Institute (modifications and corrections)
 *
 * PURPOSE:      Calculates radio coverage from a single base station
 *               according to the Ericsson model 9999.
 *
 *               Top-level path loss formula:
 *                 Lb = H_OA + m_k + sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )
 *
 *               where:
 *                 H_OA - Okamura-Hata open-area path loss
 *                 m_k - land-use clutter correction (dB), from raster
 *                 K_DFR - knife-edge diffraction loss (ITU-R P.526-16 Sec.4.1)
 *                 J_DFR - spherical Earth diffraction loss (ITU-R P.526-16 Sec.3.2)
 *                 ALPHA - tuning constant, hard-coded to 1.0
 *
 *               Formula source: TEMS(tm) CellPlanner 9.1 Common Features
 *               Technical Reference Manual, sec. 2.2 9999 Propagation Model.
 *
 * COPYRIGHT:    (C) 2009-2025 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#ifndef LOCAL_PROTO_H
#define LOCAL_PROTO_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

/* Diffraction combination tuning
 *
 * ALPHA scales the knife-edge term K_DFR inside the quadratic combination:
 *   sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )
 */
#define ALPHA                     1.0

/* Physical constants */

/* Speed of light, exact SI value (m/s).
 * Used to derive wavelength: lambda_m = SPEED_OF_LIGHT_M_PER_S / (f_MHz * 1e6)
 */
#define SPEED_OF_LIGHT_M_PER_S    299792458.0

/* Earth geometry
 *
 * ITU-R P.526-16 parametrises the effective Earth radius as a_e = k * a,
 * where a is the actual Earth radius and k is the effective Earth radius
 * factor that accounts for sub-refractive bending of radio waves under
 * normal atmospheric conditions (see e.g. eq. 16a and the definition of
 * a_e used throughout section 3).
 *
 * K_EARTH_RADIUS_FACTOR is the standard-atmosphere value of k = 4/3.
 * EFFECTIVE_EARTH_RADIUS_KM is the derived ae = k * a (km), which equals
 * 6371 * 4/3 = 8494.67 km. The factor form is preferred over a hard-coded
 * rounded value so that the derivation remains traceable to the recommendation.
 */
#define EARTH_RADIUS_KM           6371.0
#define K_EARTH_RADIUS_FACTOR     (4.0 / 3.0)
#define EFFECTIVE_EARTH_RADIUS_KM (EARTH_RADIUS_KM * K_EARTH_RADIUS_FACTOR)

/* Default Okamura-Hata tuning parameters
 *
 * TEMS(tm) CellPlanner 9.1 Common Features Technical Reference Manual, sec. 2.2
 * specifies tuning parameters for the Okamura-Hata model to be used to make the
 * model fit empirical data. It also specifies the default values for these
 * parameters which can be found below.
 *
 * A0: shifts the entire path-loss curve up or down in parallel (dB).
 * A1: controls the slope of the curve as a function of log(distance).
 * A2: controls the vertical separation between curves with different
 *     effective antenna heights.
 * A3: modifies the distance slope as a function of effective antenna height
 *     (interaction term). */
#define DEFAULT_A0                36.2
#define DEFAULT_A1                30.7
#define DEFAULT_A2                (-12.0)
#define DEFAULT_A3                0.1

/* Default antenna and link parameters */
#define DEFAULT_ANT_HEIGHT_BS_M   10.0  /* base-station TX antenna height (m) */
#define DEFAULT_ANT_HEIGHT_MS_M   1.5   /* mobile RX antenna height h_m (m)   */
#define DEFAULT_FREQ_MHZ          900.0 /* carrier frequency (MHz)            */

/* Minimum path distance guard
 *
 * The Okamura-Hata formula contains log(d) and is therefore undefined at
 * d = 0. Additionally, the Eriscsson model 9999 specification lists 0.02 km as
 * the minimum distance. the model is valid for.
 *
 * Considering the above two facts, any BS-to-MS distance below MIN_DISTANCE_KM
 * is clamped to this value before any model calculation is performed. */
#define MIN_DISTANCE_KM           0.02

/* Model parameter bundle
 *
 * Groups all tunable and fixed model quantities so that inner computation
 * functions receive a single const pointer rather than long argument lists.
 * The struct is populated once in main() from the GRASS option values and
 * then passed read-only to every model function. */
struct Model9999Params {
    /* Okamura-Hata tuning coefficients (see DEFAULT_* above for meanings) */
    double A0;
    double A1;
    double A2;
    double A3;

    /* Physical antenna heights above local ground level */
    double ant_height_bs_m; /* base-station (TX) mast height (m) */
    double ant_height_ms_m; /* mobile (RX) antenna height (m)    */

    /* Link parameters */
    double freq_mhz; /* carrier frequency (MHz) */

    /* Precomputed Okamura-Hata constant term.
     * Equals calc_hata_link_correction(freq_mhz, ant_height_ms_m).
     * Computed once in main() before the per-pixel loop and stored here
     * so that calc_model9999_path_loss() can pass it directly to
     * calc_okamura_hata_open_area() without recomputing it per pixel. */
    double link_correction;

    /* Spatial parameters */
    double scale_m; /* DEM pixel size / map resolution (m/pixel) */
};

/* Walks the DEM from the base-station pixel to the mobile-station pixel
 * using Bresenham's line algorithm and records the ground elevation and
 * Euclidean distance from the BS for every pixel visited.
 *
 * Bresenham's integer arithmetic guarantees that every pixel along the path
 * is visited exactly once with no gaps and no floating-point positional
 * drift. Distances stored in out_distances_m[] are true Euclidean distances
 * from the BS pixel centre to each visited pixel centre (NOT accumulated
 * Bresenham step distances), ensuring that the line-of-sight chord in
 * find_dominant_obstacle is parametrised correctly regardless of path azimuth.
 *
 * The profile runs from the BS pixel (index 0, distance 0) to the MS pixel
 * (last index, distance = total Euclidean path length). Both endpoints are
 * included; their displacements above the LOS chord are always negative by
 * construction and therefore never selected as the dominant obstacle.
 *
 * Note that TEMS(tm) CellPlanner 9.1 Common Features Technical Reference
 * Manual, sec. 2.2 makes no reference as to how the terrain profile is used in
 * calculating path loss. In this implementation we are using it to calculate
 * diffraction loss using the knife-edge approximation over a single dominant
 * obstacle.
 *
 * Parameters
 *   raster          2-D DEM array indexed as raster[row][col] (m ASL)
 *   bs_row, bs_col  BS pixel coordinates (0-based, row = north-to-south)
 *   ms_row, ms_col  MS pixel coordinates (0-based)
 *   scale_m         ground pixel size (m/pixel)
 *   out_heights_m   caller-allocated; receives ground elevation (m ASL)
 *   out_distances_m caller-allocated; receives Euclidean distance from BS (m)
 *   max_points      capacity of both output arrays; must be at least
 *                   max(|ms_col - bs_col|, |ms_row - bs_row|) + 1
 *
 * Returns the number of profile points written (>= 1), or -1 if max_points
 * is too small.
 */
int extract_terrain_profile(double **raster, int bs_row, int bs_col, int ms_row,
                            int ms_col, double scale_m, double *out_heights_m,
                            double *out_distances_m, int max_points);

/* Scans a terrain profile and identifies the sample with the greatest
 * positive displacement above the straight line-of-sight (LOS) chord that
 * connects the BS transmitter antenna tip to the MS receiver antenna tip.
 *
 * The LOS chord height at distance t from the BS is:
 *   los(t) = z_bs_trans_m + (z_ms_trans_m - z_bs_trans_m) * t / d_total_m
 *
 * A positive *out_h_obs_m means the terrain penetrates the LOS (NLOS
 * conditions); a negative value means the path has clearance.
 *
 * Parameters
 *   heights_m      : ground elevation profile (m ASL); length n_points
 *   distances_m    : Euclidean distance from BS for each sample (m);
 *                    length n_points
 *   n_points       : number of valid profile samples
 *   z_bs_trans_m   : BS antenna tip elevation (ground ASL + ant_height_bs) (m)
 *   z_ms_trans_m   : MS antenna tip elevation (ground ASL + ant_height_ms) (m)
 *   d_total_m      : total Euclidean BS-to-MS distance (m)
 *   out_h_obs_m    : output: maximum obstacle height above the LOS chord (m)
 *   out_d_bs_obs_m : output: Euclidean distance from BS to that obstacle (m)
 */
void find_dominant_obstacle(const double *heights_m, const double *distances_m,
                            int n_points, double z_bs_trans_m,
                            double z_ms_trans_m, double d_total_m,
                            double *out_h_obs_m, double *out_d_bs_obs_m);

/* Precomputes the part of H_OA that depends only on the link parameters
 * (frequency and mobile antenna height) and is therefore constant across
 * the entire raster computation:
 *
 *   link_correction = 44.49*log10(F) - 4.78*(log10(F))^2
 *                     - 3.2*(log10(11.75*h_m))^2
 *
 * Call this once before the per-pixel loop and pass the result to
 * calc_okamura_hata_open_area().
 *
 * Parameters
 *   F_MHz : carrier frequency (MHz)
 *   h_m_m : MS (mobile) antenna height (m)
 */
double calc_hata_link_correction(double F_MHz, double h_m_m);

/* Computes H_OA, the Okamura-Hata open-area path loss in dB:
 *
 *   H_OA = A0 + A1*log10(d)
 *             + A2*log10(H_EBK) + A3*log10(d)*log10(H_EBK)
 *             + link_correction
 *
 * Parameters
 *   d_km            : BS-to-MS Euclidean distance (km); must be >=
 * MIN_DISTANCE_KM H_EBK_m         : effective BS antenna height per model 9999
 * (m); must be > 0 link_correction : precomputed constant from
 * calc_hata_link_correction() A0..A3          : Okamura-Hata tuning
 * coefficients
 */
double calc_okamura_hata_open_area(double d_km, double H_EBK_m,
                                   double link_correction, double A0, double A1,
                                   double A2, double A3);

/* Computes the dimensionless Fresnel-Kirchhoff diffraction parameter nu (v)
 * per ITU-R P.526-16, equation (26):
 *
 *   nu = h * sqrt( 2/lambda * (1/d1 + 1/d2) )
 *      = h * sqrt( 2*(d1+d2) / (lambda * d1 * d2) )
 *
 * h, d1, d2, and lambda must all be in the same units (metres in this
 * implementation).
 *
 * Parameters
 *   h_m      : height of the obstacle tip above the LOS chord (m);
 *              positive if the obstacle is above the LOS, negative if below
 *   d1_m     : distance from the transmitter to the obstacle (m)
 *   d2_m     : distance from the obstacle to the receiver (m)
 *   lambda_m : signal wavelength (m) = SPEED_OF_LIGHT_M_PER_S / (f_MHz * 1e6)
 */
double calc_fresnel_kirchhoff_parameter(double h_m, double d1_m, double d2_m,
                                        double lambda_m);

/* Computes the knife-edge diffraction loss J(nu) in dB using the closed-form
 * approximation of ITU-R P.526-16, equation (31):
 *
 *   J(nu) = 6.9 + 20*log10( sqrt((nu - 0.1)^2 + 1) + nu - 0.1 )   [dB]
 *
 * This approximation is stated as valid for nu > -0.78 (P.526-16 eq. 31).
 * For nu <= -0.78 the obstacle clears the first Fresnel zone by a sufficient
 * margin that no diffraction penalty applies; the function returns 0.
 *
 * Note that TEMS(tm) CellPlanner 9.1 Common Features Technical Reference
 * Manual, sec. 2.2 does not specify the procedure for computing K_DFR.
 *
 * Parameter
 *   nu : Fresnel-Kirchhoff parameter (from calc_fresnel_kirchhoff_parameter)
 */
double calc_knife_edge_diffraction_loss(double nu);

/* Computes J_DFR, the spherical Earth diffraction loss in dB, using the
 * procedure of ITU-R P.526-16, section 3.2 ("Diffraction loss for any
 * distance at 10 MHz and above").
 *
 * Note that TEMS(tm) CellPlanner 9.1 Common Features Technical Reference
 * Manual, sec. 2.2 does not specify the procedure for computing J_DFR.
 *
 * The procedure comprises six steps:
 *   1. Compute the marginal line-of-sight distance d_los (eq. 21).
 *      If d >= d_los the path is over the horizon; apply Sec.3.1.1 directly.
 *   2. Compute the smallest clearance height h between the curved-Earth
 *      surface and the ray joining the antennas (eqs. 22-22e).
 *   3. Compute the required clearance h_req for zero diffraction loss
 *      (eq. 23). If h > h_req the loss is zero.
 *   4. Compute the modified effective Earth radius a_em that produces
 *      marginal LoS at the actual path distance (eq. 24).
 *   5. Apply Sec.3.1.1 with a_em to obtain Ah. If Ah < 0 the loss is zero.
 *   6. Interpolate: A = (1 - h/h_req) * Ah  (eq. 25).
 *
 * Section 3.1.1 is implemented as a static helper inside the .c file
 * (apply_section_3_1_1) and is not exposed in this header.
 *
 * For our frequency and polarisation regime (f >= 10 MHz, horizontal
 * polarisation or vertical polarisation above 20 MHz over land), the
 * surface admittance K is negligible (K << 0.001) and the beta correction
 * factor equals 1.0, simplifying eqs. (14a) and (15a).
 *
 * Parameters
 *   d_km  : total BS-to-MS Euclidean path length (km)
 *   h_t_m : transmitter (BS) antenna height above local ground (m)
 *   h_r_m : receiver (MS) antenna height above local ground (m)
 *   f_MHz : carrier frequency (MHz); must be >= 10 MHz
 */
double calc_spherical_earth_diffraction_loss(double d_km, double h_t_m,
                                             double h_r_m, double f_MHz);

/* Returns H_EBK, the effective BS antenna height:
 *
 *   H_EBK = (z_bs_asl_m + ant_height_bs_m) - (z_ms_asl_m + ant_height_ms_m)
 *         = ZoTransBS - ZoTransMS
 *
 * Physically this is the BS antenna tip height above the MS antenna tip on
 * a flat-Earth reference. The value is clamped to a minimum of
 * ant_height_bs_m: when the mobile is at a higher elevation than the BS,
 * H_EBK cannot sensibly fall below the physical height of the BS mast.
 *
 * Note that TEMS(tm) CellPlanner 9.1 Common Features Technical Reference
 * Manual, sec. 2.2 does not specify the procedure for computing H_EBK.
 *
 * Parameters
 *   z_bs_asl_m      : BS ground elevation above mean sea level (m)
 *   ant_height_bs_m : BS antenna height above local ground (m)
 *   z_ms_asl_m      : MS ground elevation above mean sea level (m)
 *   ant_height_ms_m : MS antenna height above local ground (m)
 */
double calc_effective_antenna_height(double z_bs_asl_m, double ant_height_bs_m,
                                     double z_ms_asl_m, double ant_height_ms_m);

/* Top-level Ericsson model 9999 path loss in dB
 *
 * The TEMS(tm) CellPlanner 9.1 Common Features Technical Reference
 * Manual, sec. 2.2 defines the following formula for calculating path loss:
 *
 *   Lb = H_OA + m_k + sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )
 *
 * Parameters
 *   d_km              : BS-to-MS Euclidean distance (km), already clamped to
 *                       >= MIN_DISTANCE_KM before this function is called
 *   H_EBK_m           : effective BS antenna height (m), from
 *                       calc_effective_antenna_height()
 *   m_k_dB            : land-use clutter correction at the MS pixel (dB),
 *                       read directly from the m_k input raster
 *   h_obs_above_los_m : height of dominant obstacle above the LOS chord (m),
 *                       from find_dominant_obstacle(); may be negative
 *   d_bs_to_obs_m     : Euclidean distance from BS to dominant obstacle (m),
 *                       from find_dominant_obstacle()
 *   params            : pointer to model, antenna, and frequency parameters
 */
double calc_model9999_path_loss(double d_km, double H_EBK_m, double m_k_dB,
                                double h_obs_above_los_m, double d_bs_to_obs_m,
                                const struct Model9999Params *params);

#ifdef USE_OPENCL
/* OpenCL host function: streams DEM and m_k rasters from disk into GPU
 * memory, launches the model9999_kernel 2-D NDRange, and streams the
 * resulting path-loss raster directly from GPU memory back to the GRASS
 * output raster on disk.
 *
 * Peak host RAM: two ~4 MiB staging slabs (one for upload, one for
 * readback). The full rasters are never materialised in host memory.
 *
 * Parameters
 *   infd_dem  : open GRASS file descriptor for the DEM raster
 *   infd_mk   : open GRASS file descriptor for the m_k clutter raster
 *   outfd     : open GRASS file descriptor for the output path-loss raster
 *   nrows     : number of raster rows
 *   ncols     : number of raster columns
 *   bs_row    : base-station pixel row (0-based, rounded to nearest integer)
 *   bs_col    : base-station pixel column (0-based, rounded to nearest integer)
 *   params    : model, antenna, and frequency parameters
 *
 * Returns 0 on success, -1 on any OpenCL error (errors are reported via
 * G_warning so that main() can decide whether to treat them as fatal).
 */
int calc_model9999_path_loss_ocl(int infd_dem, int infd_mk, int outfd,
                                 int nrows, int ncols, int bs_row, int bs_col,
                                 const struct Model9999Params *params);
#endif /* USE_OPENCL */

#endif /* LOCAL_PROTO_H */
