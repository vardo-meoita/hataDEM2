# MODULE_TOPDIR must point to the GRASS installation prefix ($(GISBASE)),
# not to a relative path within the GRASS source tree.  Using $(GISBASE)
# ensures that Module.make — and all targets it defines, including install —
# are found correctly whether the module is built from inside the GRASS
# source tree or installed via g.extension from an external directory.
MODULE_TOPDIR = $(GISBASE)

PGM = r.hataDEM2

# $(MATHLIB)       — math library (-lm), needed for sqrt, log10, pow, hypot, etc.
# $(OPENMP_LIBPATH) — linker search path for the OpenMP runtime library,
#                    defined by the GRASS build system when OpenMP is available.
# $(OPENMP_LIB)   — OpenMP runtime library (e.g. -lgomp), empty when absent.
LIBES        = $(GISLIB) $(RASTERLIB) $(MATHLIB) $(OPENMP_LIBPATH) $(OPENMP_LIB)
DEPENDENCIES = $(GISDEP) $(RASTERDEP)

# $(OPENMP_INCPATH) — compiler include path for OpenMP headers, empty when absent.
EXTRA_INC    = $(OPENMP_INCPATH)

# $(OPENMP_CFLAGS) — compiler flag enabling OpenMP (e.g. -fopenmp), empty when absent.
EXTRA_CFLAGS = $(OPENMP_CFLAGS)

# OpenCL support is opt-in.  Build with:
#   make WITH_OPENCL=1
# or export the variable before calling make:
#   export WITH_OPENCL=1 && make
ifdef WITH_OPENCL
EXTRA_INC    += $(OCLINCPATH)
EXTRA_CFLAGS += -DUSE_OPENCL
LIBES        += -lOpenCL
endif

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# -------------------------------------------------------------------------
# Unit tests
#
# $(CC), $(CFLAGS), and the OPENMP_* variables are all provided by
# Module.make after the include above, so no separate grass-config call
# is needed.
#
# Usage:
#   make test        — build and run the test suite
#   make clean_test  — remove the test binary
# -------------------------------------------------------------------------

TEST_BIN = test/test_spherical_earth_diffraction

.PHONY: test clean_test

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): test/test_spherical_earth_diffraction.c spherical_earth_diffraction.c
	$(CC) $(CFLAGS) $(OPENMP_CFLAGS) -o $@ \
		test/test_spherical_earth_diffraction.c \
		spherical_earth_diffraction.c \
		$(MATHLIB) $(OPENMP_LIBPATH) $(OPENMP_LIB)

clean_test:
	$(RM) $(TEST_BIN)
