/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek, Jozef Stefan Institute (modifications and corrections)
 *
 * PURPOSE:      Terrain profile extraction and dominant obstacle detection.
 *
 *               extract_terrain_profile() walks the DEM from BS to MS using
 *               Bresenham's line algorithm, collecting ground elevation and
 *               Euclidean distance for every pixel along the path.
 *
 *               find_dominant_obstacle() scans that profile to identify the
 *               sample with the greatest height above the line-of-sight chord,
 *               which is the input to both diffraction calculations.
 *
 * COPYRIGHT:    (C) 2009-2025 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#include "local_proto.h"

int extract_terrain_profile(double **raster, int bs_row, int bs_col, int ms_row,
                            int ms_col, double scale_m, double *out_heights_m,
                            double *out_distances_m, int max_points)
{
    /* Absolute pixel offsets between the two endpoints */
    int dcol = abs(ms_col - bs_col);
    int drow = abs(ms_row - bs_row);

    /* Per-step direction: +1 toward increasing index, -1 toward decreasing */
    int scol = (ms_col >= bs_col) ? 1 : -1;
    int srow = (ms_row >= bs_row) ? 1 : -1;

    /* Bresenham's algorithm visits exactly max(dcol, drow) + 1 pixels.
     * Verify that the caller has allocated enough space before we begin. */
    int n_needed = (dcol > drow ? dcol : drow) + 1;
    if (n_needed > max_points)
        return -1;

    /* Bresenham error term: positive values favour column steps,
     * negative values favour row steps. */
    int err = dcol - drow;

    int row = bs_row;
    int col = bs_col;
    int n = 0;

    for (;;) {
        /* Record ground elevation at the current pixel.
         *
         * Distance is the Euclidean distance from the BS pixel centre to
         * the current pixel centre, not the accumulated Bresenham step
         * distance.  Euclidean distances are used so that the LOS chord
         * in find_dominant_obstacle() is parametrised correctly for all
         * path azimuths (accumulated step distances differ from Euclidean
         * distance for non-cardinal azimuths). */
        double dr = (double)(row - bs_row);
        double dc = (double)(col - bs_col);
        out_distances_m[n] = sqrt(dr * dr + dc * dc) * scale_m;
        out_heights_m[n] = raster[row][col];
        n++;

        /* Stop once the MS pixel has been recorded. */
        if (row == ms_row && col == ms_col)
            break;

        /* Advance to the next pixel along the Bresenham line.
         *
         * e2 is computed once and used in both conditions so that when
         * e2 satisfies both simultaneously, the algorithm takes a diagonal
         * step (both col and row advance in the same iteration). */
        int e2 = 2 * err;
        if (e2 > -drow) {
            err -= drow;
            col += scol;
        }
        if (e2 < dcol) {
            err += dcol;
            row += srow;
        }
    }

    return n;
}

void find_dominant_obstacle(const double *heights_m, const double *distances_m,
                            int n_points, double z_bs_trans_m,
                            double z_ms_trans_m, double d_total_m,
                            double *out_h_obs_m, double *out_d_bs_obs_m)
{
    /* For a degenerate path (BS and MS at the same pixel), d_total_m is zero
     * and the LOS chord is undefined.  Return the first profile sample height
     * relative to the BS antenna tip so that downstream code receives a
     * consistently very negative value (ground is always below the antenna). */
    if (n_points <= 0 || d_total_m <= 0.0) {
        *out_h_obs_m = (n_points > 0) ? heights_m[0] - z_bs_trans_m : 0.0;
        *out_d_bs_obs_m = 0.0;
        return;
    }

    /* Initialise from the first profile sample so no sentinel value is needed.
     * The BS pixel itself (index 0, distance 0) has a height above LOS equal
     * to ground_BS - z_bs_trans_m = -ant_height_bs, which is always negative,
     * so it will be superseded by any interior point that obstructs the path.
     * The same logic applies to the MS pixel at the other end. */
    double t0 = distances_m[0] / d_total_m;
    double los_h0 = z_bs_trans_m + (z_ms_trans_m - z_bs_trans_m) * t0;
    *out_h_obs_m = heights_m[0] - los_h0;
    *out_d_bs_obs_m = distances_m[0];

    for (int i = 1; i < n_points; i++) {
        /* Normalised position along the path: 0 at BS, 1 at MS. */
        double t = distances_m[i] / d_total_m;

        /* Height of the straight LOS chord at this sample.
         * The chord connects the BS antenna tip (z_bs_trans_m) to the
         * MS antenna tip (z_ms_trans_m). */
        double los_height_m = z_bs_trans_m + (z_ms_trans_m - z_bs_trans_m) * t;

        /* Positive: terrain penetrates the LOS (NLOS conditions).
         * Negative: terrain is below the LOS (clearance). */
        double h_above_los = heights_m[i] - los_height_m;

        if (h_above_los > *out_h_obs_m) {
            *out_h_obs_m = h_above_los;
            *out_d_bs_obs_m = distances_m[i];
        }
    }
}
