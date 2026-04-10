/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek, Jozef Stefan Institute (modifications and corrections)
 *
 * PURPOSE:      Ericsson model 9999 path loss assembly.
 *
 *               calc_effective_antenna_height() computes the effective BS
 *               antenna height H_EBK as defined by model 9999.
 *
 *               calc_model9999_path_loss() assembles the top-level formula:
 *
 *                 Lb = H_OA + m_k + sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )
 *
 *               by calling the Okamura-Hata, knife-edge diffraction, and
 *               spherical Earth diffraction sub-models in turn.
 *
 * COPYRIGHT:    (C) 2009-2025 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#include "local_proto.h"

double calc_effective_antenna_height(double z_bs_asl_m, double ant_height_bs_m,
                                     double z_ms_asl_m, double ant_height_ms_m)
{
    /* H_EBK is the height of the BS antenna tip above the MS antenna tip,
     * projected onto a flat-Earth reference:
     *
     *   H_EBK = (z_BS_ASL + ant_height_BS) - (z_MS_ASL + ant_height_MS)
     *         = ZoTransBS - ZoTransMS
     */
    double H_EBK =
        (z_bs_asl_m + ant_height_bs_m) - (z_ms_asl_m + ant_height_ms_m);

    /* Clamp to the physical BS mast height.  When the mobile is at a
     * higher terrain elevation than the BS, H_EBK would otherwise go
     * negative or fall below a meaningful minimum.  Using ant_height_bs_m
     * as the floor ensures the Okamura-Hata log(H_EBK) term remains
     * well-defined and physically interpretable. */
    if (H_EBK < ant_height_bs_m)
        H_EBK = ant_height_bs_m;

    return H_EBK;
}

double calc_model9999_path_loss(double d_km, double H_EBK_m, double m_k_dB,
                                double h_obs_above_los_m, double d_bs_to_obs_m,
                                const struct Model9999Params *params)
{
    /* Okamura-Hata open-area path loss H_OA (dB).
     *
     * params->link_correction was precomputed once before the per-pixel
     * loop by calc_hata_link_correction(freq_mhz, ant_height_ms_m) and
     * bundles the frequency term and the mobile-height correction that
     * are constant across the entire raster computation. */
    double H_OA = calc_okamura_hata_open_area(
        d_km, H_EBK_m, params->link_correction, params->A0, params->A1,
        params->A2, params->A3);

    /* Signal wavelength (m), derived from the carrier frequency.
     * Used by the Fresnel-Kirchhoff parameter calculation. */
    double lambda_m = SPEED_OF_LIGHT_M_PER_S / (params->freq_mhz * 1.0e6);

    /* Knife-edge diffraction loss K_DFR (dB) per ITU-R P.526-16 Sec.4.1.
     *
     * d_ms_to_obs_m is the distance from the dominant obstacle to the
     * mobile station; together with d_bs_to_obs_m it parametrises the
     * Fresnel-Kirchhoff diffraction parameter nu.
     *
     * If the obstacle lies exactly at one terminal (d_bs_to_obs_m == 0
     * or d_ms_to_obs_m <= 0), calc_fresnel_kirchhoff_parameter returns
     * -1.0 (well below the -0.78 threshold) and the loss evaluates to
     * zero, avoiding a division by zero. */
    double d_ms_to_obs_m = d_km * 1000.0 - d_bs_to_obs_m;

    double nu = calc_fresnel_kirchhoff_parameter(
        h_obs_above_los_m, d_bs_to_obs_m, d_ms_to_obs_m, lambda_m);
    double K_DFR = calc_knife_edge_diffraction_loss(nu);

    /* Spherical Earth diffraction loss J_DFR (dB) per ITU-R P.526-16
     * Sec.3.2.
     *
     * H_EBK_m is used as the transmitter height rather than the raw
     * physical mast height (params->ant_height_bs_m).  H_EBK already
     * accounts for the terrain elevation difference between the BS and
     * MS sites: it is the height of the BS antenna tip above the MS
     * antenna tip on a flat-Earth reference.
     *
     * Using the physical mast height instead would make the smooth-Earth
     * radio horizon depend only on the mast height (e.g. 3 m), ignoring
     * a BS that sits on a hill 200 m above the MS terrain.  That causes
     * the Sec.3.2 procedure to classify geometrically clear paths as
     * over-the-horizon and apply tens of dB of spurious diffraction loss.
     *
     * With H_EBK_m the horizon distance correctly reflects the terrain
     * geometry, and J_DFR is non-zero only when the effective height
     * advantage is insufficient to keep the path within the radio horizon. */
    double J_DFR = calc_spherical_earth_diffraction_loss(
        d_km, H_EBK_m, params->ant_height_ms_m, params->freq_mhz);

    /* Assemble the model 9999 path loss:
     *
     *   Lb = H_OA + m_k + sqrt( (ALPHA * K_DFR)^2 + J_DFR^2 )
     *
     * Formula source: TEMS(tm) CellPlanner 9.1 Common Features Technical
     * Reference Manual, sec. 2.2 9999 Propagation Model.
     *
     * hypot() computes sqrt(a^2 + b^2) with protection against
     * intermediate overflow for large individual loss values. */
    return H_OA + m_k_dB + hypot(ALPHA * K_DFR, J_DFR);
}
