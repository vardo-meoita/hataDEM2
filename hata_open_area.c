/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek, Jozef Stefan Institute (modifications and corrections)
 *
 * PURPOSE:      Okamura-Hata open-area path loss model (H_OA).
 *
 * COPYRIGHT:    (C) 2009-2025 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#include "local_proto.h"

/* The frequency term and the mobile antenna height correction both depend
 * solely on link-level constants (F and h_m) that do not change across the
 * raster computation. Precomputing their combined value once avoids
 * redundant log() and pow() calls inside the per-pixel loop.
 *
 *   freq_term   =  44.49*log10(F) - 4.78*(log10(F))^2
 *   mobile_corr =  3.2*(log10(11.75*h_m))^2
 *
 *   link_correction = freq_term - mobile_corr
 *
 * Formula source: TEMS(tm) CellPlanner 9.1 Common Features Technical
 * Reference Manual, sec. 2.2 9999 Propagation Model.
 */
double calc_hata_link_correction(double F_MHz, double h_m_m)
{
    double log_F = log10(F_MHz);

    /* Frequency-dependent path-loss term. The quadratic form in log(F)
     * captures the empirical non-linear frequency dependence of the
     * Okamura-Hata open-area model. */
    double freq_term = 44.49 * log_F - 4.78 * log_F * log_F;

    /* Mobile antenna height correction.
     * Accounts for the gain at the mobile due to its height above ground.
     * The factor 11.75 and the squared logarithm come from the empirical
     * Okamura-Hata open-area fit. */
    double mobile_corr = 3.2 * pow(log10(11.75 * h_m_m), 2.0);

    return freq_term - mobile_corr;
}

/* Per-pixel computation of H_OA. Only the distance- and
 * antenna-height-dependent terms are evaluated here; the constant
 * link_correction is supplied precomputed by calc_hata_link_correction().
 *
 *   H_OA = A0 + A1*log10(d) + A2*log10(H_EBK) + A3*log10(d)*log10(H_EBK)
 *          + link_correction
 *
 * Formula source: TEMS(tm) CellPlanner 9.1 Common Features Technical
 * Reference Manual, sec. 2.2 9999 Propagation Model.
 *
 * The A3 term introduces a coupled dependence: the slope of the loss curve
 * as a function of distance changes with effective antenna height.
 */
double calc_okamura_hata_open_area(double d_km, double H_EBK_m,
                                   double link_correction, double A0, double A1,
                                   double A2, double A3)
{
    double log_d = log10(d_km);
    double log_H_EBK = log10(H_EBK_m);

    double distance_term =
        A0 + A1 * log_d + A2 * log_H_EBK + A3 * log_d * log_H_EBK;

    return distance_term + link_correction;
}
