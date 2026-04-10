/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek, Jozef Stefan Institute (modifications and corrections)
 *
 * PURPOSE:      Spherical Earth diffraction loss per ITU-R P.526-16,
 *               section 3.2 ("Diffraction loss for any distance at 10 MHz
 *               and above").
 *
 *               The six-step procedure in section 3.2 calls the numerical
 *               calculation of section 3.1.1 as a sub-procedure.
 *               apply_section_3_1_1() implements section 3.1.1.2 with
 *               beta = 1, which is valid for horizontal polarisation at any
 *               frequency, and for vertical polarisation above 20 MHz over
 *               land (ITU-R P.526-16, section 3.1.1.1).  For these cases
 *               the surface admittance K << 0.001 and has no influence on
 *               the result.
 *
 * COPYRIGHT:    (C) 2009-2025 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#include "local_proto.h"

/* Static helpers implementing ITU-R P.526-16 section 3.1.1.2
 *
 * All helpers use practical units unless noted otherwise:
 *   distance  - km
 *   height    - m
 *   frequency - MHz
 *   ae        - km */

/* ITU-R P.526-16, eq. (14a) with beta = 1:
 *
 *   X = 2.188 * f^(1/3) * ae^(-2/3) * d
 */
static double normalise_distance(double d_km, double ae_km, double f_MHz)
{
    return 2.188 * pow(f_MHz, 1.0 / 3.0) * pow(ae_km, -2.0 / 3.0) * d_km;
}

/* ITU-R P.526-16, eq. (15a) with beta = 1:
 *
 *   Y = 9.575e-3 * f^(2/3) * ae^(-1/3) * h
 */
static double normalise_height(double h_m, double ae_km, double f_MHz)
{
    return 9.575e-3 * pow(f_MHz, 2.0 / 3.0) * pow(ae_km, -1.0 / 3.0) * h_m;
}

/* ITU-R P.526-16, eqs. (17a) and (17b).
 *
 *   F(X) = 11 + 10*log10(X) - 17.6*X          for X >= 1.6   (17a)
 *   F(X) = -20*log10(X) - 5.6488 * X^1.425    for X <  1.6   (17b)
 */
static double distance_function_F(double X)
{
    if (X >= 1.6)
        return 11.0 + 10.0 * log10(X) - 17.6 * X;
    else
        return -20.0 * log10(X) - 5.6488 * pow(X, 1.425);
}

/* ITU-R P.526-16, eqs. (18) and (18a), with B = beta * Y = Y (beta = 1,
 * eq. 18b).
 *
 *   G(Y) = 17.6*(B-1.1)^0.5 - 5*log10(B-1.1) - 8   for B >  2   (18)
 *   G(Y) = 20*log10(B + 0.1*B^3)                     for B <= 2   (18a)
 *
 * The floor G(Y) >= 2 + 20*log10(K) from eq. (18) is not applied: for
 * f >= 10 MHz the surface admittance K << 0.001 and the floor is
 * effectively negative infinity, leaving G(Y) unconstrained. */
static double height_gain_function_G(double Y)
{
    double B = Y; /* beta = 1 */

    if (B > 2.0)
        return 17.6 * sqrt(B - 1.1) - 5.0 * log10(B - 1.1) - 8.0;
    else
        return 20.0 * log10(B + 0.1 * B * B * B);
}

/* ITU-R P.526-16, section 3.1.1.2, master equation (13):
 *
 *   20*log10(E/E0) = F(X) + G(Y1) + G(Y2)
 *
 * Returns Ah = -(F(X) + G(Y1) + G(Y2)) in dB, i.e. positive means
 * diffraction loss, negative means the formula indicates gain (the caller
 * is responsible for treating negative Ah as zero loss).
 *
 * ae_km is passed explicitly so this function can be called with either
 * the standard effective Earth radius EFFECTIVE_EARTH_RADIUS_KM (for
 * over-horizon paths, step 1 of section 3.2) or the modified radius aem
 * (for the interpolation path, step 5 of section 3.2).
 */
static double apply_section_3_1_1(double d_km, double h1_m, double h2_m,
                                  double f_MHz, double ae_km)
{
    double X = normalise_distance(d_km, ae_km, f_MHz);
    double Y1 = normalise_height(h1_m, ae_km, f_MHz);
    double Y2 = normalise_height(h2_m, ae_km, f_MHz);

    return -(distance_function_F(X) + height_gain_function_G(Y1) +
             height_gain_function_G(Y2));
}

/* ITU-R P.526-16, section 3.2 -- six-step procedure.
 *
 * The geometry steps (1-4) work in SI units (metres) because equations
 * (21)-(24) require self-consistent units.  The section 3.1.1 call in
 * step 5 receives practical units (km / m / MHz) as required by eqs.
 * (14a) and (15a). */

double calc_spherical_earth_diffraction_loss(double d_km, double h_t_m,
                                             double h_r_m, double f_MHz)
{
    /* SI-unit quantities used throughout the geometry steps. */
    double ae_m = EFFECTIVE_EARTH_RADIUS_KM * 1000.0;
    double d_m = d_km * 1000.0;
    double lambda_m = SPEED_OF_LIGHT_M_PER_S / (f_MHz * 1.0e6);

    /* Step 1 (eq. 21): marginal line-of-sight distance.
     *
     *   d_los = 2 * sqrt(ae) * (sqrt(h1) + sqrt(h2))
     *
     * d_los is the path length at which the straight ray between the
     * two antennas is tangent to the spherical Earth surface.  If the
     * actual path length d equals or exceeds d_los the path is beyond
     * the radio horizon and section 3.1.1 is applied directly. */
    double sqrt_ht = sqrt(h_t_m);
    double sqrt_hr = sqrt(h_r_m);
    double d_los = 2.0 * sqrt(ae_m) * (sqrt_ht + sqrt_hr);

    if (d_m >= d_los)
        return apply_section_3_1_1(d_km, h_t_m, h_r_m, f_MHz,
                                   EFFECTIVE_EARTH_RADIUS_KM);

    /* Step 2 (eqs. 22-22e): smallest clearance height h.
     *
     * h is the height of the straight ray above the curved Earth
     * surface at the point of minimum separation between the ray and
     * the Earth.  All quantities in SI units (metres).
     *
     *   m = d^2 / (4 * ae * (h1 + h2))                        (22e)
     *   c = (h1 - h2) / (h1 + h2)                             (22d)
     *   q = 3*c*sqrt(3*m) / (2*(m+1)^(3/2))
     *   b = 2*sqrt((m+1)/(3*m)) * cos(pi/3 + arccos(q)/3)    (22c)
     *   d1 = (d/2) * (1 + b)                                  (22a)
     *   d2 = d - d1                                           (22b)
     *   h  = [(h1 - d1^2/(2*ae))*d2
     *          + (h2 - d2^2/(2*ae))*d1] / d                   (22) */
    double m = (d_m * d_m) / (4.0 * ae_m * (h_t_m + h_r_m));
    double c = (h_t_m - h_r_m) / (h_t_m + h_r_m);

    double b;
    if (m < 1.0e-10) {
        /* For very short paths m -> 0 the factor sqrt((m+1)/(3m))
         * diverges while cos(...) -> 0; their product converges to c
         * (verified by L'Hopital expansion of eq. 22c).  Apply the
         * limiting value directly to avoid a numerical 0 * Inf. */
        b = c;
    }
    else {
        /* Argument of arccos in eq. (22c).  Clamped to [-1, 1] to
         * guard against floating-point rounding pushing it marginally
         * outside the domain of acos(). */
        double q = 3.0 * c * sqrt(3.0 * m) / (2.0 * pow(m + 1.0, 1.5));
        if (q > 1.0)
            q = 1.0;
        if (q < -1.0)
            q = -1.0;

        b = 2.0 * sqrt((m + 1.0) / (3.0 * m)) * cos(M_PI / 3.0 + acos(q) / 3.0);
    }

    double d1 = 0.5 * d_m * (1.0 + b);
    double d2 = d_m - d1;

    double h = ((h_t_m - d1 * d1 / (2.0 * ae_m)) * d2 +
                (h_r_m - d2 * d2 / (2.0 * ae_m)) * d1) /
               d_m;

    /* Step 3 (eq. 23): required clearance for zero diffraction loss.
     *
     *   h_req = 0.552 * sqrt(d1 * d2 * lambda / d)
     *
     * h_req is the minimum clearance (in metres) needed for the path
     * to be treated as unobstructed.  If the actual clearance h
     * exceeds h_req the diffraction loss is zero. */
    double h_req = 0.552 * sqrt(d1 * d2 * lambda_m / d_m);

    if (h > h_req)
        return 0.0;

    /* Step 4 (eq. 24): modified effective Earth radius aem.
     *
     *   aem = 0.5 * (d / (sqrt(h1) + sqrt(h2)))^2
     *
     * aem is the effective Earth radius that would place the path
     * exactly at marginal LoS for the actual distance d and antenna
     * heights.  Section 3.1.1 is evaluated with aem in step 5. */
    double aem_m =
        0.5 * (d_m / (sqrt_ht + sqrt_hr)) * (d_m / (sqrt_ht + sqrt_hr));

    /* Step 5: diffraction loss Ah using the modified Earth radius.
     *
     * apply_section_3_1_1 expects ae in km; convert aem from metres.
     *
     * If Ah < 0 the formula yields gain, which is non-physical for a
     * diffraction mechanism.  The loss is treated as zero. */
    double Ah = apply_section_3_1_1(d_km, h_t_m, h_r_m, f_MHz, aem_m / 1000.0);

    if (Ah < 0.0)
        return 0.0;

    /* Step 6 (eq. 25): interpolated diffraction loss.
     *
     *   A = (1 - h / h_req) * Ah
     *
     * The factor (1 - h/h_req) interpolates between full loss (h = 0,
     * grazing incidence on the modified Earth) and zero loss (h = h_req,
     * just enough Fresnel clearance). */
    return (1.0 - h / h_req) * Ah;
}
