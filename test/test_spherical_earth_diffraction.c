/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek (modifications and corrections), Jozef Stefan Institute
 *
 * PURPOSE:      Unit tests for calc_spherical_earth_diffraction_loss().
 *
 *               Fifteen tests are executed:
 *
 *               1. Short LoS path with sufficient Fresnel clearance returns
 *                  exactly 0.0 dB (step 3 early exit in section 3.2).
 *
 *               2. Over-horizon path returns a value close to the manually
 *                  computed reference of 49.5 dB (step 1 in section 3.2,
 *                  direct application of section 3.1.1).
 *
 *               3. Diffraction loss increases monotonically with distance
 *                  for two over-horizon paths of different lengths.
 *
 *               4. The result is symmetric in h_t and h_r (swapping
 *                  transmitter and receiver heights gives the same loss).
 *
 *               5. The return value is non-negative for all test inputs.
 *
 *               Tests 6-9 are derived from the ITU-R P.526 nomogram
 *               (figure P.0526-03) for the distance function F(X).
 *
 *               Tests 10-14 are derived from the ITU-R P.526 nomogram
 *               (figure P.0526-04) "Diffraction by a spherical Earth -
 *               height-gain", horizontal polarisation, k = 4/3.  That
 *               nomogram displays G(Y) = H(h) in dB as a function of
 *               frequency (left scale, k=4/3) and antenna height above
 *               ground (right scale).  Since G(Y) and F(X) are additive
 *               (eq. 13: 20*log10(E/E0) = F(X) + G(Y1) + G(Y2)), the
 *               total Ah for an over-horizon path can be cross-checked
 *               against both nomograms independently.
 *
 *               All tests 10-14 use f=900 MHz, d=100 km.  All three
 *               height choices are over-horizon at that distance:
 *                 h=1.5 m  ->  d_los ~= 14.3 km  (<100 km)
 *                 h=10 m   ->  d_los ~= 36.9 km  (<100 km)
 *                 h=30 m   ->  d_los ~= 63.9 km  (<100 km)
 *
 *               10. Symmetric reference, h_t=h_r=1.5 m:
 *                   G(Y) ~= -23.7 dB (deep negative regime, B << 1).
 *                   Ah ~= 118.7 dB.
 *
 *               11. Symmetric reference, h_t=h_r=30 m:
 *                   G(Y) ~= +3.7 dB (positive regime, B > B_zero).
 *                   Ah ~= 63.9 dB.
 *
 *               12. Asymmetric reference, h_t=30 m, h_r=1.5 m:
 *                   Combines one positive and one deeply negative G(Y).
 *                   Ah ~= 91.3 dB.
 *
 *               13. Averaging identity (exact for over-horizon paths):
 *                   Ah(A, B) = [Ah(A,A) + Ah(B,B)] / 2, because X is
 *                   independent of antenna height and G is additive.
 *                   Verified with A=30 m, B=1.5 m; tolerance 1e-9.
 *
 *               14. Three-way height ordering at f=900 MHz, d=100 km:
 *                   loss(h=30 m) < loss(h=10 m) < loss(h=1.5 m),
 *                   consistent with the height-gain nomogram's axis.
 *
 *               15. Combined nomogram reading per ITU-R P.526-16 section
 *                   3.1.2, equation (20) and Note 1, over land (k=4/3):
 *                     F(d)  read from P.0526-03 at 900 MHz, 100 km
 *                     H(h1) read from P.0526-04 at 900 MHz, 30 m
 *                     H(h2) read from P.0526-04 at 900 MHz, 10 m
 *                   Eq. (20): 20*log10(E/E0) = F(d) + H(h1) + H(h2)
 *                             ~= -71.4 + 3.75 + (-7.01) = -74.66 dB
 *                   Note 1:  attenuation = -(-74.66) ~= 74.7 dB.
 *
 * REFERENCE VALUES (manually derived):
 *
 *   All reference values use ae = 6371 * 4/3 = 8494.667 km
 *   (EFFECTIVE_EARTH_RADIUS_KM = EARTH_RADIUS_KM * K_EARTH_RADIUS_FACTOR).
 *
 *   Case A - d=0.5 km, h_t=30 m, h_r=1.5 m, f=900 MHz
 *     ae  = 6371 * 4/3 km = 8494.667 km  (EFFECTIVE_EARTH_RADIUS_KM)
 *     d_los = 2*sqrt(ae_m)*(sqrt(30)+sqrt(1.5)) ~= 39 065 m  -> within LoS
 *     c = (30-1.5)/31.5 = 0.905,  b ~= c (m->0 limit),
 *     d1 ~= 476 m,  d2 ~= 24 m
 *     h     ~= 2.85 m   (clearance above curved Earth)
 *     h_req = 0.552*sqrt(d1*d2*lambda/d) ~= 1.51 m
 *     h > h_req  ->  loss = 0.0 dB  (exact, step 3 exit)
 *
 *   Case B - d=50 km, h_t=30 m, h_r=1.5 m, f=900 MHz
 *     d > d_los  ->  section 3.1.1 applied directly (step 1)
 *     X  = 2.188 * 900^(1/3) * ae^(-2/3) * 50 ~= 2.54
 *     F(X) = 11 + 10*log10(2.54) - 17.6*2.54          ~= -29.6 dB
 *     Y1   = 9.575e-3 * 900^(2/3) * ae^(-1/3) * 30    ~=  1.313
 *     G(Y1)= 20*log10(Y1 + 0.1*Y1^3)                  ~=  +3.75 dB
 *     Y2   = 9.575e-3 * 900^(2/3) * ae^(-1/3) * 1.5   ~=  0.0656
 *     G(Y2)= 20*log10(Y2 + 0.1*Y2^3)                  ~= -23.65 dB
 *     Ah   = -(F + G(Y1) + G(Y2))                      ~= +49.5 dB
 *     Test uses tolerance +-1.5 dB to accommodate rounding in the
 *     manual calculation.
 *
 *   All reference values below also use
 *   ae = 6371 * 4/3 = 8494.667 km.
 *   F(X) at f=900 MHz, d=100 km:
 *     X   = 2.188 * 900^(1/3) * ae^(-2/3) * 100   ~=  5.08
 *     F(X)= 11 + 10*log10(5.08) - 17.6*5.08        ~= -71.4 dB
 *   G(Y) at f=900 MHz for selected heights (ae used via eq. 15a):
 *     h=1.5 m:  Y ~= 0.0657,  G ~= -23.65 dB  (eq. 18a, B << 1)
 *     h=10 m:   Y ~= 0.438,   G ~=  -7.01 dB  (eq. 18a)
 *     h=30 m:   Y ~= 1.313,   G ~=  +3.75 dB  (eq. 18a, above zero)
 *   Total Ah = -(F(X) + G(Y1) + G(Y2)):
 *     h_t=h_r=1.5 m:       ~= 118.7 dB
 *     h_t=h_r=10 m:        ~=  85.4 dB
 *     h_t=h_r=30 m:        ~=  63.9 dB
 *     h_t=30 m, h_r=1.5 m: ~=  91.3 dB
 *       (= arithmetic mean of the two symmetric cases above, exact)
 *
 *   Case E2 -- d=100 km, h_t=30 m, h_r=10 m, f=900 MHz  (nomogram test 15)
 *     d_los = 2*sqrt(ae_m)*(sqrt(30)+sqrt(10)) ~= 50 359 m = 50.4 km
 *     d > d_los  ->  section 3.1.1 applied directly (step 1)
 *
 *     From P.0526-03 (distance nomogram, k=4/3):
 *       X   = 2.188 * 900^(1/3) * ae^(-2/3) * 100  ~=  5.08
 *       F(X)= 11 + 10*log10(5.08) - 17.6*5.08       ~= -71.4 dB
 *
 *     From P.0526-04 (height-gain nomogram, k=4/3):
 *       Y1  = 9.575e-3 * 900^(2/3) * ae^(-1/3) * 30 ~=  1.313
 *       H(h1) = G(Y1) = 20*log10(1.313 + 0.1*1.313^3) ~= +3.75 dB
 *       Y2  = 9.575e-3 * 900^(2/3) * ae^(-1/3) * 10 ~=  0.438
 *       H(h2) = G(Y2) = 20*log10(0.438 + 0.1*0.438^3) ~= -7.01 dB
 *
 *     Equation (20): 20*log10(E/E0) = F(d) + H(h1) + H(h2)
 *                                   = -71.4 + 3.75 + (-7.01) = -74.66 dB
 *     Note 1: attenuation = -(eq. 20) = +74.66 dB ~= 74.7 dB
 *     Test uses tolerance +-2.0 dB.
 *
 *   Case C - d=100 km, h_t=h_r=10 m, f=100 MHz  (nomogram test 6)
 *     d_los = 2*sqrt(ae_m)*(sqrt(10)+sqrt(10)) ~= 36 868 m = 36.9 km
 *     d > d_los  ->  section 3.1.1 applied directly (step 1)
 *     X   = 2.188 * 100^(1/3) * ae^(-2/3) * 100  ~=  2.44
 *     F(X)= -20*log10(2.44) - 5.6488*2.44^1.425   ~= -28.0 dB
 *     Y   = 9.575e-3 * 100^(2/3) * ae^(-1/3) * 10 ~=  0.101
 *     G(Y)= 20*log10(0.101 + 0.1*0.101^3)          ~= -19.9 dB
 *     Ah  = -(F + 2*G)                              ~= +67.8 dB
 *     Test uses tolerance +-2.0 dB.
 *
 *   Case D - d=100 km, h_t=h_r=10 m, f=1 GHz  (nomogram test 7)
 *     d_los ~= 36.9 km  ->  d > d_los, step 1 applies
 *     X   = 2.188 * 1000^(1/3) * ae^(-2/3) * 100  ~=  5.25
 *     F(X)= 11 + 10*log10(5.25) - 17.6*5.25        ~= -74.3 dB
 *     Y   = 9.575e-3 * 1000^(2/3) * ae^(-1/3) * 10 ~=  0.469
 *     G(Y)= 20*log10(0.469 + 0.1*0.469^3)           ~=  -6.4 dB
 *     Ah  = -(F + 2*G)                               ~= +87.1 dB
 *     Test uses tolerance +-2.0 dB.
 *
 *   Case E - d=100 km, h_t=h_r=30 m, f=1 GHz  (nomogram test 9)
 *     d_los = 2*sqrt(ae_m)*(sqrt(30)+sqrt(30)) ~= 63 853 m = 63.9 km
 *     d > d_los  ->  step 1 applies
 *     Y   = 9.575e-3 * 1000^(2/3) * ae^(-1/3) * 30 ~=  1.408
 *     G(Y)= 20*log10(1.408 + 0.1*1.408^3)           ~=  +4.5 dB
 *     Ah  = -(F(5.25) + 2*G(1.408))                 ~= +65.2 dB
 *
 * COMPILATION AND RUN (from the module source directory):
 *
 *   make test   # run from the module root (raster/r.hataDEM2/)
 *   Exit code 0 = all tests passed, 1 = one or more tests failed.
 *
 * COPYRIGHT:    (C) 2009-2018 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#include <math.h>
#include <stdio.h>

#include "../local_proto.h"

static int g_tests_run = 0;
static int g_tests_passed = 0;

/* check_true: pass when condition is non-zero. */
static void check_true(int condition, const char *label)
{
    g_tests_run++;
    if (condition) {
        g_tests_passed++;
        printf("PASS : %s\n", label);
    }
    else {
        printf("FAIL : %s\n", label);
    }
}

/* check_near: pass when |actual - expected| <= tolerance. */
static void check_near(double actual, double expected, double tolerance,
                       const char *label)
{
    g_tests_run++;
    double diff = fabs(actual - expected);
    if (diff <= tolerance) {
        g_tests_passed++;
        printf("PASS : %s  (%.4f, expected %.4f, diff %.4f)\n", label, actual,
               expected, diff);
    }
    else {
        printf("FAIL : %s  (%.4f, expected %.4f, diff %.4f > tol %.4f)\n",
               label, actual, expected, diff, tolerance);
    }
}

int main(void)
{
    printf("=== calc_spherical_earth_diffraction_loss - unit tests ===\n");
    printf("    ae = %.3f km  (EARTH_RADIUS_KM * K_EARTH_RADIUS_FACTOR)\n\n",
           EFFECTIVE_EARTH_RADIUS_KM);

    /* Test 1: short LoS path with sufficient Fresnel clearance.
     *
     * d=0.5 km, h_t=30 m, h_r=1.5 m, f=900 MHz.
     * Manually verified: h ~= 2.85 m > h_req ~= 1.51 m.
     * Section 3.2, step 3 exits early and returns exactly 0.0. */
    {
        double loss =
            calc_spherical_earth_diffraction_loss(0.5, 30.0, 1.5, 900.0);
        check_true(loss == 0.0, "short LoS path (d=0.5 km, h_t=30, h_r=1.5): "
                                "h > h_req => exactly 0.0 dB");
    }

    /* Test 2: over-horizon path -- numerical reference value.
     *
     * d=50 km, h_t=30 m, h_r=1.5 m, f=900 MHz.
     * d > d_los ~= 39 km, so section 3.1.1 is applied directly (step 1).
     * Manually derived Ah ~= 49.5 dB; tolerance is +-1.5 dB. */
    {
        double loss =
            calc_spherical_earth_diffraction_loss(50.0, 30.0, 1.5, 900.0);
        check_near(loss, 49.5, 1.5,
                   "over-horizon (d=50 km, h_t=30, h_r=1.5, f=900): "
                   "~49.5 dB");
    }

    /* Test 3: monotonicity -- loss must increase with path length.
     *
     * Both 50 km and 60 km are over the horizon (d_los ~= 39 km), so
     * both follow the step 1 path.  Longer path => larger X => F(X)
     * more negative => greater Ah. */
    {
        double loss_50 =
            calc_spherical_earth_diffraction_loss(50.0, 30.0, 1.5, 900.0);
        double loss_60 =
            calc_spherical_earth_diffraction_loss(60.0, 30.0, 1.5, 900.0);
        check_true(loss_60 > loss_50, "monotonicity: loss(60 km) > loss(50 km) "
                                      "(both over-horizon)");
    }

    /* Test 4: symmetry -- swapping h_t and h_r must give the same result.
     *
     * The formula is symmetric in h1 and h2: d_los, h, h_req, aem, and
     * G(Y1)+G(Y2) are all unchanged when h_t and h_r are swapped.
     * Tolerance is set to the expected floating-point round-trip error
     * (a few ULPs), expressed as an absolute value. */
    {
        double loss_fwd =
            calc_spherical_earth_diffraction_loss(50.0, 30.0, 1.5, 900.0);
        double loss_rev =
            calc_spherical_earth_diffraction_loss(50.0, 1.5, 30.0, 900.0);
        check_near(loss_fwd, loss_rev, 1e-9,
                   "symmetry: loss(h_t=30, h_r=1.5) == loss(h_t=1.5, h_r=30)");
    }

    /* Test 5: non-negativity -- diffraction loss must never be negative.
     *
     * Checked for both a LoS path (zero loss) and an over-horizon path
     * (positive loss).  The section 3.2 procedure clamps negative Ah
     * values to zero (step 5), so all code paths should return >= 0. */
    {
        double loss_los =
            calc_spherical_earth_diffraction_loss(0.5, 30.0, 1.5, 900.0);
        double loss_oth =
            calc_spherical_earth_diffraction_loss(50.0, 30.0, 1.5, 900.0);
        check_true(loss_los >= 0.0,
                   "non-negativity: short LoS path returns >= 0");
        check_true(loss_oth >= 0.0,
                   "non-negativity: over-horizon path returns >= 0");
    }

    /* Test 6: nomogram reference -- f=100 MHz, d=100 km, h_t=h_r=10 m.
     *
     * d_los ~= 36.9 km < 100 km, so the path is over the horizon and
     * section 3.1.1 is applied directly (step 1).
     * Manually derived: X ~= 2.44, F(X) ~= -28.0 dB,
     *                   Y ~= 0.101, G(Y) ~= -19.9 dB per antenna,
     *                   Ah = -(F + 2G) ~= 67.8 dB.
     * Tolerance +-2.0 dB accounts for nomogram reading precision. */
    {
        double loss =
            calc_spherical_earth_diffraction_loss(100.0, 10.0, 10.0, 100.0);
        check_near(loss, 67.8, 2.0,
                   "nomogram ref (f=100 MHz, d=100 km, h_t=h_r=10 m): "
                   "~67.8 dB");
    }

    /* Test 7: nomogram reference -- f=1 GHz, d=100 km, h_t=h_r=10 m.
     *
     * Same geometry as test 6 but at 1 GHz.  Both d and X are larger,
     * driving F(X) further negative and raising Ah substantially.
     * Manually derived: X ~= 5.25, F(X) ~= -74.3 dB,
     *                   Y ~= 0.469, G(Y) ~= -6.4 dB per antenna,
     *                   Ah = -(F + 2G) ~= 87.1 dB.
     * Tolerance +-2.0 dB. */
    {
        double loss =
            calc_spherical_earth_diffraction_loss(100.0, 10.0, 10.0, 1000.0);
        check_near(loss, 87.1, 2.0,
                   "nomogram ref (f=1 GHz,   d=100 km, h_t=h_r=10 m): "
                   "~87.1 dB");
    }

    /* Test 8: frequency monotonicity (nomogram frequency axis).
     *
     * At fixed distance (100 km, over-horizon) and fixed antenna heights
     * (10 m), increasing frequency increases X and shifts F(X) more
     * negative, raising Ah.  G(Y) also increases with frequency but less
     * rapidly at these heights, so the net effect is more loss.
     * Tests 6 and 7 already provide the exact values; this test makes
     * the ordering constraint explicit. */
    {
        double loss_100mhz =
            calc_spherical_earth_diffraction_loss(100.0, 10.0, 10.0, 100.0);
        double loss_1ghz =
            calc_spherical_earth_diffraction_loss(100.0, 10.0, 10.0, 1000.0);
        check_true(loss_1ghz > loss_100mhz,
                   "frequency monotonicity: loss(1 GHz) > loss(100 MHz) "
                   "at d=100 km, h=10 m");
    }

    /* Test 9: height monotonicity (nomogram height axis).
     *
     * At fixed frequency (1 GHz) and distance (100 km), increasing
     * antenna height raises G(Y) (the height-gain term), reducing the
     * total loss.  h_t=h_r=30 m is used because its d_los ~= 63.9 km
     * is still below 100 km, keeping the path over-horizon.
     * Manually derived Ah(30 m) ~= 65.2 dB vs Ah(10 m) ~= 87.1 dB. */
    {
        double loss_10m =
            calc_spherical_earth_diffraction_loss(100.0, 10.0, 10.0, 1000.0);
        double loss_30m =
            calc_spherical_earth_diffraction_loss(100.0, 30.0, 30.0, 1000.0);
        check_true(loss_30m < loss_10m,
                   "height monotonicity: loss(h=30 m) < loss(h=10 m) "
                   "at f=1 GHz, d=100 km");
    }

    /* Test 10: height-gain nomogram -- h_t=h_r=1.5 m, f=900 MHz, d=100 km.
     *
     * h=1.5 m puts Y well into the B << 1 regime of eq. (18a), giving a
     * strongly negative G(Y) ~= -23.65 dB.  This is the typical mobile
     * handset height and produces the largest diffraction loss of the
     * three symmetric cases tested here.
     * Manually derived: F(X) ~= -71.4 dB, G ~= -23.65 dB per antenna,
     *                   Ah = -(F + 2G) ~= 118.7 dB.
     * Tolerance +-2.0 dB. */
    {
        double loss =
            calc_spherical_earth_diffraction_loss(100.0, 1.5, 1.5, 900.0);
        check_near(loss, 118.7, 2.0,
                   "height-gain ref (f=900 MHz, d=100 km, h_t=h_r=1.5 m): "
                   "~118.7 dB");
    }

    /* Test 11: height-gain nomogram -- h_t=h_r=30 m, f=900 MHz, d=100 km.
     *
     * h=30 m places Y above the zero-crossing of G(Y) (which occurs at
     * h ~= 21 m for 900 MHz), giving a positive G(Y) ~= +3.75 dB.  The
     * positive height-gain reduces the total diffraction loss relative
     * to the h=10 m and h=1.5 m cases.
     * Manually derived: F(X) ~= -71.4 dB, G ~= +3.75 dB per antenna,
     *                   Ah = -(F + 2G) ~= 63.9 dB.
     * Tolerance +-2.0 dB. */
    {
        double loss =
            calc_spherical_earth_diffraction_loss(100.0, 30.0, 30.0, 900.0);
        check_near(loss, 63.9, 2.0,
                   "height-gain ref (f=900 MHz, d=100 km, h_t=h_r=30 m):  "
                   "~63.9 dB");
    }

    /* Test 12: height-gain nomogram -- h_t=30 m, h_r=1.5 m, f=900 MHz,
     *          d=100 km.
     *
     * The asymmetric case combines a positive G(Y_BS) ~= +3.75 dB and a
     * strongly negative G(Y_MS) ~= -23.65 dB.  This is the realistic
     * scenario for a base station (30 m mast) communicating with a
     * handset (1.5 m height).
     * Manually derived: Ah = -(F + G_BS + G_MS)
     *                      = -(-71.4 + 3.75 + (-23.65)) ~= 91.3 dB.
     * Tolerance +-2.0 dB. */
    {
        double loss =
            calc_spherical_earth_diffraction_loss(100.0, 30.0, 1.5, 900.0);
        check_near(loss, 91.3, 2.0,
                   "height-gain ref (f=900 MHz, d=100 km, h_t=30 m, "
                   "h_r=1.5 m): ~91.3 dB");
    }

    /* Test 13: averaging identity (exact for over-horizon paths).
     *
     * For any over-horizon path section 3.2 reduces to section 3.1.1:
     *   Ah = -(F(X) + G(Y1) + G(Y2))
     * X depends only on distance and frequency, not on antenna height.
     * Therefore:
     *   Ah(A, B) = -(F + G(A) + G(B))
     *            = [ -(F + 2G(A)) + -(F + 2G(B)) ] / 2
     *            = [ Ah(A,A) + Ah(B,B) ] / 2
     * This is an exact algebraic identity; the test uses tolerance 1e-9
     * to allow for floating-point rounding only. */
    {
        double loss_sym_bs =
            calc_spherical_earth_diffraction_loss(100.0, 30.0, 30.0, 900.0);
        double loss_sym_ms =
            calc_spherical_earth_diffraction_loss(100.0, 1.5, 1.5, 900.0);
        double loss_asym =
            calc_spherical_earth_diffraction_loss(100.0, 30.0, 1.5, 900.0);
        double expected_mean = (loss_sym_bs + loss_sym_ms) / 2.0;
        check_near(loss_asym, expected_mean, 1e-9,
                   "averaging identity: Ah(30m,1.5m) = "
                   "[Ah(30m,30m) + Ah(1.5m,1.5m)] / 2");
    }

    /* Test 14: three-way height ordering (height-gain nomogram axis).
     *
     * At fixed frequency (900 MHz) and distance (100 km), increasing
     * antenna height raises G(Y) monotonically (visible on the
     * height-gain nomogram's middle scale).  This reduces the total
     * diffraction loss.  The three heights span negative G (1.5 m),
     * slightly negative G (10 m), and positive G (30 m) regimes. */
    {
        double loss_1p5m =
            calc_spherical_earth_diffraction_loss(100.0, 1.5, 1.5, 900.0);
        double loss_10m =
            calc_spherical_earth_diffraction_loss(100.0, 10.0, 10.0, 900.0);
        double loss_30m =
            calc_spherical_earth_diffraction_loss(100.0, 30.0, 30.0, 900.0);
        check_true(loss_30m < loss_10m,
                   "height ordering: loss(h=30 m) < loss(h=10 m) "
                   "at f=900 MHz, d=100 km");
        check_true(loss_10m < loss_1p5m,
                   "height ordering: loss(h=10 m) < loss(h=1.5 m) "
                   "at f=900 MHz, d=100 km");
    }

    /* Test 15: combined reading from P.0526-03 and P.0526-04 per
     *          ITU-R P.526-16 section 3.1.2, equation (20), Note 1.
     *
     * Scenario: f=900 MHz, d=100 km, h_t=30 m (BS), h_r=10 m (MS),
     *           over land (k=4/3, EFFECTIVE_EARTH_RADIUS_KM).
     *
     * Step 1: read F(d) from the distance nomogram (P.0526-03):
     *           draw a line from 900 MHz (k=4/3 scale) through 100 km
     *           on the distance scale -> F(d) ~= -71.4 dB.
     *
     * Step 2: read H(h) from the height-gain nomogram (P.0526-04)
     *         for each antenna independently:
     *           900 MHz through 30 m -> H(h1) ~= +3.75 dB
     *           900 MHz through 10 m -> H(h2) ~= -7.01 dB
     *
     * Step 3: apply equation (20):
     *           20*log10(E/E0) = F(d) + H(h1) + H(h2)
     *                         = -71.4 + 3.75 + (-7.01) = -74.66 dB
     *
     * Step 4: apply Note 1 -- attenuation = -(equation 20):
     *           attenuation = +74.66 dB ~= 74.7 dB
     *
     * d_los = 2*sqrt(ae_m)*(sqrt(30)+sqrt(10)) ~= 50.4 km < 100 km,
     * so the path is over-horizon and section 3.1.1 applies directly
     * (step 1 of section 3.2).  The nomogram method is therefore valid
     * (Note 1: method is invalid only when eq. (20) > 0 dB, i.e., when
     * it predicts gain; here it predicts -74.66 dB, a clear loss). */
    {
        double loss =
            calc_spherical_earth_diffraction_loss(100.0, 30.0, 10.0, 900.0);
        check_near(loss, 74.7, 2.0,
                   "combined nomogram (P.0526-03 + P.0526-04, sec. 3.1.2): "
                   "f=900 MHz, d=100 km, h_t=30 m, h_r=10 m: ~74.7 dB");
    }

    printf("\n=== %d / %d tests passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}