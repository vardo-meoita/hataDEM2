/****************************************************************************
 *
 * MODULE:       r.hataDEM2
 * AUTHOR(S):    tifil
 *               Andrej Vilhar, Jozef Stefan Institute
 *               Tomaz Javornik, Jozef Stefan Institute
 *               Andrej Hrovat, Jozef Stefan Institute
 *               Igor Ozimek, Jozef Stefan Institute (modifications and corrections)
 *
 * PURPOSE:      Calculates radio coverage from a single base station using
 *               the Ericsson model 9999 path loss formula:
 *
 *                 Lb = H_OA + m_k + sqrt( (ALPHA*K_DFR)^2 + J_DFR^2 )
 *
 *               H_OA  - Okamura-Hata open-area path loss
 *               m_k   - land-use clutter correction (raster input, dB)
 *               K_DFR - knife-edge diffraction (ITU-R P.526-16 Sec.4.1)
 *               J_DFR - spherical Earth diffraction (ITU-R P.526-16 Sec.3.2)
 *
 * COPYRIGHT:    (C) 2009-2025 Jozef Stefan Institute
 *               (C) 2026 tifil
 *               This program is free software under the GNU General Public
 *               License (>=v2). Read the file COPYING that comes with RaPlaT
 *               for details.
 *
 *****************************************************************************/

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <math.h>
#include <stdlib.h>

#include <grass/gis.h>
#include <grass/raster.h>
#include <grass/glocale.h>

#include "local_proto.h"

/* Opens the named raster map, reads every row into a newly allocated
 * nrows x ncols matrix of doubles, and closes the map.  GRASS null cells
 * are stored as NaN so that downstream code can test for them with isnan().
 * The caller must free each mat[row] and then mat itself with G_free(). */
static double **load_raster_to_double_matrix(const char *name, int nrows,
                                             int ncols)
{
    double **mat;
    FCELL *row_buf;
    int fd, row, col;

    fd = Rast_open_old(name, "");
    mat = (double **)G_malloc(nrows * sizeof(double *));
    row_buf = Rast_allocate_f_buf();

    for (row = 0; row < nrows; row++) {
        mat[row] = (double *)G_malloc(ncols * sizeof(double));
        Rast_get_f_row(fd, row_buf, row);
        for (col = 0; col < ncols; col++) {
            mat[row][col] = Rast_is_f_null_value(&row_buf[col])
                                ? (double)NAN
                                : (double)row_buf[col];
        }
    }

    G_free(row_buf);
    Rast_close(fd);
    return mat;
}

static void free_double_matrix(double **mat, int nrows)
{
    int row;

    for (row = 0; row < nrows; row++)
        G_free(mat[row]);
    G_free(mat);
}

int main(int argc, char *argv[])
{
    struct GModule *module;

    /* Options are grouped in a single struct following the r.sun convention. */
    struct {
        struct Option *dem, *mk, *output;
        struct Option *coordinate;
        struct Option *ant_height, *rx_ant_height, *frequency;
        struct Option *a0, *a1, *a2, *a3;
        struct Option *nprocs;
    } parm;

    struct Cell_head window;
    struct History history;

    int nrows, ncols;
    int bs_row, bs_col;
    int max_profile_pts;
    int fd_out;
    int nprocs;
    int row;

    double east, north;
    double z_bs_asl;

    double **dem;
    double **mk;
    FCELL **output;
    FCELL *out_row_buf;
    FCELL fnull;

    struct Model9999Params params;

    /* Initialise GRASS and define the module */
    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("raster"));
    G_add_keyword(_("radio propagation"));
    G_add_keyword(_("model 9999"));
    G_add_keyword(_("Ericsson"));
    G_add_keyword(_("parallel"));
    module->label = _("Ericsson model 9999 radio propagation.");
    module->description =
        _("Calculates radio coverage from a single base station using the "
          "Ericsson model 9999 path loss formula.  Combines the Okamura-Hata "
          "open-area model with knife-edge (ITU-R P.526-16 Sec.4.1) and "
          "spherical Earth (ITU-R P.526-16 Sec.3.2) diffraction corrections. "
          "The land-use clutter correction m_k is supplied as a raster map.");

    /* Input / output raster maps */
    parm.dem = G_define_standard_option(G_OPT_R_INPUT);
    parm.dem->key = "input_dem";
    parm.dem->label = _("Input elevation raster map");
    parm.dem->description = _("Digital elevation model [metres above MSL]");
    parm.dem->guisection = _("Input");

    parm.mk = G_define_standard_option(G_OPT_R_INPUT);
    parm.mk->key = "m_k";
    parm.mk->label = _("Land-use clutter correction raster map");
    parm.mk->description = _("Per-pixel clutter correction m_k [dB]");
    parm.mk->guisection = _("Input");

    parm.output = G_define_standard_option(G_OPT_R_OUTPUT);
    parm.output->description = _("Output path loss raster map [dB]");
    parm.output->guisection = _("Output");

    /* Base station location and antenna */
    parm.coordinate = G_define_standard_option(G_OPT_M_COORDS);
    parm.coordinate->required = YES;
    parm.coordinate->label = _("Base station location");
    parm.coordinate->description = _("Easting and northing of the base station "
                                     "antenna [map units]");
    parm.coordinate->guisection = _("Base station");

    parm.ant_height = G_define_option();
    parm.ant_height->key = "ant_height";
    parm.ant_height->type = TYPE_DOUBLE;
    parm.ant_height->required = NO;
    parm.ant_height->answer = "10";
    parm.ant_height->label = _("Base station antenna height [m]");
    parm.ant_height->description =
        _("Transmitter antenna height above local ground level");
    parm.ant_height->guisection = _("Base station");

    /* Mobile station / link parameters */
    parm.rx_ant_height = G_define_option();
    parm.rx_ant_height->key = "rx_ant_height";
    parm.rx_ant_height->type = TYPE_DOUBLE;
    parm.rx_ant_height->required = NO;
    parm.rx_ant_height->answer = "1.5";
    parm.rx_ant_height->label = _("Receiver antenna height h_m [m]");
    parm.rx_ant_height->description =
        _("Mobile antenna height above local ground level");
    parm.rx_ant_height->guisection = _("Link");

    parm.frequency = G_define_option();
    parm.frequency->key = "frequency";
    parm.frequency->type = TYPE_DOUBLE;
    parm.frequency->required = NO;
    parm.frequency->answer = "900";
    parm.frequency->label = _("Carrier frequency [MHz]");
    parm.frequency->description =
        _("Must be >= 10 MHz (lower bound of ITU-R P.526-16 Sec.3.2)");
    parm.frequency->guisection = _("Link");

    /* Okamura-Hata tuning parameters */
    parm.a0 = G_define_option();
    parm.a0->key = "a0";
    parm.a0->type = TYPE_DOUBLE;
    parm.a0->required = NO;
    parm.a0->answer = "36.2";
    parm.a0->label = _("A0: absolute level adjustment [dB]");
    parm.a0->description = _("Shifts the entire path-loss curve up or down");
    parm.a0->guisection = _("Tuning");

    parm.a1 = G_define_option();
    parm.a1->key = "a1";
    parm.a1->type = TYPE_DOUBLE;
    parm.a1->required = NO;
    parm.a1->answer = "30.7";
    parm.a1->label = _("A1: distance slope coefficient");
    parm.a1->description = _("Controls the slope of the loss curve vs log(d)");
    parm.a1->guisection = _("Tuning");

    parm.a2 = G_define_option();
    parm.a2->key = "a2";
    parm.a2->type = TYPE_DOUBLE;
    parm.a2->required = NO;
    parm.a2->answer = "-12.0";
    parm.a2->label = _("A2: effective antenna height adjustment [dB]");
    parm.a2->description = _("Separation between loss curves for different "
                             "effective antenna heights");
    parm.a2->guisection = _("Tuning");

    parm.a3 = G_define_option();
    parm.a3->key = "a3";
    parm.a3->type = TYPE_DOUBLE;
    parm.a3->required = NO;
    parm.a3->answer = "0.1";
    parm.a3->label = _("A3: slope/height interaction coefficient");
    parm.a3->description = _("Modifies the distance slope as a function of "
                             "effective antenna height");
    parm.a3->guisection = _("Tuning");

    /* Parallel processing */
    parm.nprocs = G_define_option();
    parm.nprocs->key = "nprocs";
    parm.nprocs->type = TYPE_INTEGER;
    parm.nprocs->required = NO;
    parm.nprocs->answer = "1";
    parm.nprocs->options = "1-1000";
    parm.nprocs->label = _("Number of parallel threads");
    parm.nprocs->description =
        _("Number of OpenMP threads for the per-pixel path loss computation");
    parm.nprocs->guisection = _("Parallel");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    /* Parse and validate all numeric options */
    params.ant_height_bs_m = atof(parm.ant_height->answer);
    params.ant_height_ms_m = atof(parm.rx_ant_height->answer);
    params.freq_mhz = atof(parm.frequency->answer);
    params.A0 = atof(parm.a0->answer);
    params.A1 = atof(parm.a1->answer);
    params.A2 = atof(parm.a2->answer);
    params.A3 = atof(parm.a3->answer);
    nprocs = atoi(parm.nprocs->answer);

    if (params.ant_height_bs_m <= 0.0)
        G_fatal_error(
            _("Base station antenna height must be positive (got %g)"),
            params.ant_height_bs_m);
    if (params.ant_height_ms_m <= 0.0)
        G_fatal_error(_("Receiver antenna height must be positive (got %g)"),
                      params.ant_height_ms_m);
    if (params.freq_mhz < 10.0)
        G_fatal_error(_("Carrier frequency must be >= 10 MHz (got %g MHz); "
                        "ITU-R P.526-16 Sec.3.2 requires f >= 10 MHz"),
                      params.freq_mhz);

    if (nprocs < 1) {
        G_warning(_("nprocs must be >= 1; using 1 (got %d)"), nprocs);
        nprocs = 1;
    }
#if defined(_OPENMP)
    omp_set_num_threads(nprocs);
    G_message(_("Using %d thread(s)"), nprocs);
#else
    if (nprocs > 1)
        G_warning(
            _("This build was compiled without OpenMP support; "
              "nprocs is ignored and computation will be single-threaded"));
    nprocs = 1;
#endif

    /* Parse and validate the base station coordinates */
    if (!G_scan_easting(parm.coordinate->answers[0], &east, G_projection()))
        G_fatal_error(_("Illegal easting value: <%s>"),
                      parm.coordinate->answers[0]);
    if (!G_scan_northing(parm.coordinate->answers[1], &north, G_projection()))
        G_fatal_error(_("Illegal northing value: <%s>"),
                      parm.coordinate->answers[1]);

    G_get_window(&window);

    if (east < window.west || east > window.east || north < window.south ||
        north > window.north)
        G_fatal_error(_("Base station coordinate (%.8g E, %.8g N) is outside "
                        "the current computational region"),
                      east, north);

    /* Region metrics and base-station pixel indices */
    nrows = Rast_window_rows();
    ncols = Rast_window_cols();

    params.scale_m = window.ew_res;

    /* Row increases southward; col increases eastward. */
    bs_row = (int)((window.north - north) / window.ns_res);
    bs_col = (int)((east - window.west) / window.ew_res);

    /* Guard against floating-point edge cases at region boundaries. */
    if (bs_row < 0)
        bs_row = 0;
    if (bs_row >= nrows)
        bs_row = nrows - 1;
    if (bs_col < 0)
        bs_col = 0;
    if (bs_col >= ncols)
        bs_col = ncols - 1;

    /* Precompute the Okamura-Hata link correction term.
     *
     * This combines the frequency-dependent term and the mobile antenna height
     * correction.  Both are constant across the entire raster computation, so
     * computing them once here avoids repeating log() and pow() calls inside
     * the per-pixel loop.  The value is also needed by the OpenCL kernel,
     * where it is passed as a scalar kernel argument. */
    params.link_correction =
        calc_hata_link_correction(params.freq_mhz, params.ant_height_ms_m);

    G_debug(1, "link_correction = %.4f dB  (freq=%.1f MHz, h_m=%.2f m)",
            params.link_correction, params.freq_mhz, params.ant_height_ms_m);

#ifdef USE_OPENCL
    /* OpenCL path
     *
     * Both input rasters are streamed from disk into GPU VRAM in small
     * batches; the result is streamed back from VRAM directly to the output
     * raster on disk.  The full rasters are never materialised in host RAM.
     * Peak host usage: two ~4 MiB staging slabs (upload + readback). */
    {
        int infd_dem = Rast_open_old(parm.dem->answer, "");
        int infd_mk = Rast_open_old(parm.mk->answer, "");

        /* Validate that the DEM has a real value at the BS pixel before
         * committing to the (potentially long) GPU upload. */
        {
            FCELL *bs_buf = Rast_allocate_f_buf();
            Rast_get_f_row(infd_dem, bs_buf, bs_row);
            FCELL bs_elev = bs_buf[bs_col];
            G_free(bs_buf);
            if (Rast_is_f_null_value(&bs_elev))
                G_fatal_error(
                    _("DEM has a null value at the base station pixel "
                      "(row %d, col %d). Cannot proceed."),
                    bs_row, bs_col);
        }

        fd_out = Rast_open_new(parm.output->answer, FCELL_TYPE);

        G_message(_("Computing path loss (OpenCL)..."));

        if (calc_model9999_path_loss_ocl(infd_dem, infd_mk, fd_out, nrows,
                                         ncols, bs_row, bs_col, &params) != 0)
            G_fatal_error(_("OpenCL path loss computation failed."));

        Rast_close(infd_dem);
        Rast_close(infd_mk);
        Rast_close(fd_out);
    }
#else /* USE_OPENCL not defined: CPU / OpenMP path */
    /* Load input raster maps into memory */
    G_message(_("Reading input raster maps..."));

    dem = load_raster_to_double_matrix(parm.dem->answer, nrows, ncols);
    mk = load_raster_to_double_matrix(parm.mk->answer, nrows, ncols);

    /* The DEM must be valid at the base station pixel. */
    z_bs_asl = dem[bs_row][bs_col];
    if (isnan(z_bs_asl))
        G_fatal_error(_("DEM has a null value at the base station pixel "
                        "(row %d, col %d). Cannot proceed."),
                      bs_row, bs_col);

    /* Allocate output matrix; initialise all cells to GRASS null.
     *
     * Storing results here before writing lets the parallel computation
     * proceed without any I/O serialisation.  Each thread writes to a
     * disjoint subset of cells, so no locking is needed. */
    Rast_set_f_null_value(&fnull, 1);

    output = (FCELL **)G_malloc(nrows * sizeof(FCELL *));
    for (row = 0; row < nrows; row++) {
        output[row] = (FCELL *)G_malloc(ncols * sizeof(FCELL));
        for (int col = 0; col < ncols; col++)
            output[row][col] = fnull;
    }

    /* Conservative upper bound on the Bresenham profile length: the
     * longest possible path is the diagonal of the raster. */
    max_profile_pts =
        (int)ceil(sqrt((double)nrows * nrows + (double)ncols * ncols)) + 2;

    /* Per-pixel path loss computation, parallelised over rows with OpenMP.
     *
     * The outer #pragma omp parallel block is opened once.  Each thread
     * allocates its own terrain profile buffers (thr_heights, thr_dists)
     * a single time and reuses them across all pixels it processes.  The
     * inner #pragma omp for distributes rows dynamically so that threads
     * with faster pixels can pick up more work.
     *
     * Additional notes:
     * - dem[][], mk[][], params: read-only; safe for concurrent access.
     * - output[row][col]: each (row, col) pair is written by exactly one
     *   thread; no synchronisation is needed.
     * - G_percent(): called only from thread 0, using an atomically captured
     *   row counter, to avoid interleaved output. */
    G_message(_("Computing path loss..."));

    {
        int rows_done = 0; /* shared counter, updated atomically */

#pragma omp parallel num_threads(nprocs)
        {
            /* Thread-local profile arrays.  Allocated once per thread and
             * reused for every pixel, avoiding per-pixel malloc overhead. */
            double *thr_heights =
                (double *)G_malloc(max_profile_pts * sizeof(double));
            double *thr_dists =
                (double *)G_malloc(max_profile_pts * sizeof(double));

#pragma omp for schedule(dynamic)
            for (row = 0; row < nrows; row++) {
                int done; /* local copy of the progress counter */

                for (int col = 0; col < ncols; col++) {
                    double z_ms_asl, m_k_val;
                    double d_col, d_row, d_km;
                    double z_bs_trans, z_ms_trans;
                    double H_EBK;
                    double h_obs, d_obs;
                    int n_pts;
                    double loss;

                    z_ms_asl = dem[row][col];
                    m_k_val = mk[row][col];

                    /* Skip cells where either input is null. */
                    if (isnan(z_ms_asl) || isnan(m_k_val))
                        continue;

                    /* Euclidean BS-to-MS distance in km.
                     * The multiplication by scale_m converts pixel offsets
                     * to metres; 1.0e-3 converts to km. */
                    d_col = (double)(col - bs_col);
                    d_row = (double)(row - bs_row);
                    d_km = sqrt(d_col * d_col + d_row * d_row) *
                           params.scale_m * 1.0e-3;

                    /* Clamp to the minimum computable distance (log(0) guard). */
                    if (d_km < MIN_DISTANCE_KM)
                        d_km = MIN_DISTANCE_KM;

                    /* Antenna tip elevations above mean sea level. */
                    z_bs_trans = z_bs_asl + params.ant_height_bs_m;
                    z_ms_trans = z_ms_asl + params.ant_height_ms_m;

                    /* Model 9999 effective BS antenna height H_EBK. */
                    H_EBK = calc_effective_antenna_height(
                        z_bs_asl, params.ant_height_bs_m, z_ms_asl,
                        params.ant_height_ms_m);

                    /* Terrain profile (Bresenham) and dominant obstacle. */
                    n_pts = extract_terrain_profile(
                        dem, bs_row, bs_col, row, col, params.scale_m,
                        thr_heights, thr_dists, max_profile_pts);

                    if (n_pts < 0) {
                        /* max_profile_pts is conservatively large, so this
                         * branch should never be reached in practice. */
                        G_warning(_("Profile buffer overflow at row %d col %d; "
                                    "skipping pixel"),
                                  row, col);
                        continue;
                    }

                    find_dominant_obstacle(
                        thr_heights, thr_dists, n_pts, z_bs_trans, z_ms_trans,
                        d_km * 1.0e3, /* total path length in metres */
                        &h_obs, &d_obs);

                    /* Assemble the model 9999 path loss. */
                    loss = calc_model9999_path_loss(d_km, H_EBK, m_k_val, h_obs,
                                                    d_obs, &params);

                    output[row][col] = (FCELL)loss;
                } /* col */

                /* Atomically capture the incremented counter so that each
                 * thread reports its own completed-row count without a
                 * critical section around the G_percent call. */
#pragma omp atomic capture
                done = ++rows_done;

                /* Confine progress output to thread 0 to avoid interleaving
                 * from multiple threads writing to stderr simultaneously. */
#if defined(_OPENMP)
                if (omp_get_thread_num() == 0)
#endif
                    G_percent(done, nrows, 2);

            } /* row */

            G_free(thr_heights);
            G_free(thr_dists);

        } /* omp parallel */

        G_percent(nrows, nrows, 2); /* ensure the meter reaches 100 % */
    }

    /* Write the output raster map row by row */
    fd_out = Rast_open_new(parm.output->answer, FCELL_TYPE);

    out_row_buf = Rast_allocate_f_buf();

    for (row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++)
            out_row_buf[col] = output[row][col];
        Rast_put_f_row(fd_out, out_row_buf);
    }

    G_free(out_row_buf);
    Rast_close(fd_out);
#endif /* USE_OPENCL */

    /* Record computation parameters in the output map's history */
    Rast_short_history(parm.output->answer, "raster", &history);

    Rast_append_format_history(
        &history,
        " ------------------------------------------------------------------");
    Rast_append_format_history(&history,
                               " r.hataDEM2  -  Ericsson model 9999 path loss");
    Rast_append_format_history(
        &history,
        " ------------------------------------------------------------------");
    Rast_append_format_history(
        &history, " Base station:             %.8g E  %.8g N", east, north);
    Rast_append_format_history(&history, " BS antenna height (m):    %.2f",
                               params.ant_height_bs_m);
    Rast_append_format_history(&history, " MS antenna height (m):    %.2f",
                               params.ant_height_ms_m);
    Rast_append_format_history(&history, " Carrier frequency (MHz):  %.1f",
                               params.freq_mhz);
    Rast_append_format_history(
        &history, " A0 / A1 / A2 / A3:        %.2f / %.2f / %.2f / %.2f",
        params.A0, params.A1, params.A2, params.A3);
    Rast_append_format_history(&history, " Input DEM:                %s",
                               parm.dem->answer);
    Rast_append_format_history(&history, " Clutter correction (m_k): %s",
                               parm.mk->answer);
    Rast_append_format_history(
        &history,
        " ------------------------------------------------------------------");

    Rast_command_history(&history);
    Rast_write_history(parm.output->answer, &history);

#ifndef USE_OPENCL
    /* Free all heap memory (CPU path only; OpenCL path uses no host matrices) */
    free_double_matrix(dem, nrows);
    free_double_matrix(mk, nrows);

    for (row = 0; row < nrows; row++)
        G_free(output[row]);
    G_free(output);
#endif /* USE_OPENCL */

    exit(EXIT_SUCCESS);
}
