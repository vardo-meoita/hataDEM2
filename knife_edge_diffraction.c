/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek, Jozef Stefan Institute (modifications and corrections)
 *
 * PURPOSE:      Knife-edge diffraction loss per ITU-R P.526-16, section 4.1.
 *
 *               calc_fresnel_kirchhoff_parameter() computes the dimensionless
 *               diffraction parameter nu (eq. 26).
 *
 *               calc_knife_edge_diffraction_loss() converts nu to a loss in
 *               dB using the closed-form approximation (eq. 31).
 *
 * COPYRIGHT:    (C) 2009-2025 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#include "local_proto.h"

/*
 * calc_fresnel_kirchhoff_parameter
 *
 * ITU-R P.526-16, equation (26):
 *
 *   nu = h * sqrt( 2/lambda * (1/d1 + 1/d2) )
 *      = h * sqrt( 2*(d1+d2) / (lambda * d1 * d2) )
 *
 * h may be negative (obstacle below the LOS chord); the sign propagates
 * through to nu and is then handled by calc_knife_edge_diffraction_loss().
 *
 * If either d1_m or d2_m is not positive the geometry is degenerate
 * (obstacle at one of the terminals).  In that case nu is set to a value
 * well below the -0.78 threshold so that the loss evaluates to zero.
 */
double calc_fresnel_kirchhoff_parameter(double h_m, double d1_m, double d2_m,
                                        double lambda_m)
{
    if (d1_m <= 0.0 || d2_m <= 0.0)
        return -1.0;

    return h_m * sqrt(2.0 * (d1_m + d2_m) / (lambda_m * d1_m * d2_m));
}

/*
 * calc_knife_edge_diffraction_loss
 *
 * ITU-R P.526-16, equation (31) — closed-form approximation of J(nu):
 *
 *   J(nu) = 6.9 + 20*log10( sqrt((nu - 0.1)^2 + 1) + nu - 0.1 )   [dB]
 *
 * The approximation is stated as valid for nu > -0.78.  Below that threshold
 * the obstacle affords sufficient Fresnel-zone clearance that the diffraction
 * loss is negligible; the function returns 0.
 */
double calc_knife_edge_diffraction_loss(double nu)
{
    /* Threshold from ITU-R P.526-16, equation (31). */
    if (nu <= -0.78)
        return 0.0;

    /* The argument of log10 is sqrt((nu-0.1)^2 + 1) + (nu-0.1).
     * Introducing x = nu - 0.1 makes the expression easier to read and
     * avoids evaluating (nu - 0.1) twice. */
    double x = nu - 0.1;
    return 6.9 + 20.0 * log10(sqrt(x * x + 1.0) + x);
}
