ifeq ($(OS),Windows_NT)
  UNAME	       := Windows
else
  UNAME	       := $(shell uname -s)
endif

TOP 	    	= .
# FIXME: Hack.  But we don't have realpath on OSX, do we?
ABSTOP	    	= $(shell pwd)
EXTDIR	    	= $(TOP)/external
SHRDIR	    	= $(TOP)/share

#-------------------------------------------------------------------------------
# C++ configuration

# Directories
CXX_DIR	    	= $(TOP)/c++
CXX_INCDIR 	= $(CXX_DIR)/include
CXX_SRCDIR 	= $(CXX_DIR)/src
CXX_TSTDIR    	= $(CXX_DIR)/test

# Gtest configuration
GTEST_DIR       = $(EXTDIR)/gtest
GTEST_INCDIR    = $(GTEST_DIR)/include
GTEST_LIB       = $(GTEST_DIR)/gtest_main.a

# Compiler and linker
CXX            += -std=c++14
CPPFLAGS        = -I$(CXX_INCDIR)
CXXFLAGS        = -fPIC -g -Wall
LDFLAGS	    	= 
LDLIBS          = 

#-------------------------------------------------------------------------------
# C++ building and linking

# Sources and outputs
CXX_SRCS        = $(wildcard $(CXX_SRCDIR)/*.cc) 
CXX_DEPS        = $(CXX_SRCS:%.cc=%.d)
CXX_OBJS        = $(CXX_SRCS:%.cc=%.o)
CXX_LIB	    	= $(CXX_SRCDIR)/libcron.a
CXX_BIN_SRCS	= $(wildcard $(CXX_SRCDIR)/bin/*.cc)
CXX_BINS        = $(CXX_BIN_SRCS:%.cc=%)

$(CXX_DEPS): \
%.d: 		%.cc
	@echo "generating $@"; set -e; \
	$(CXX) -MM $(CPPFLAGS) $< | sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' > $@

# How to link an executable. 
%:  	    	    	%.o
%:  	    	    	%.o 
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# How to build a static library.  
%.a:
%.a:
	@mkdir -p $(shell dirname $@)
	ar -r $@ $^

ifeq ($(UNAME),Darwin)
  SO_LDFLAGS = -bundle -undefined dynamic_lookup
else ifeq ($(UNAME),Linux)
  SO_LDFLAGS = -shared
endif

# How to build a shared library.
%.so:
%.so:
	$(CXX) $(LDFLAGS) $(SO_LDFLAGS) $^ $(LDLIBS) -o $@

# The static library includes these objects.
$(CXX_LIB): $(CXX_OBJS)

# Binaries link the static library.
$(CXX_BINS): $(CXX_LIB)
$(CXX_BINS): LDLIBS += $(CXX_LIB)

#-------------------------------------------------------------------------------
# C++ unit tests

# Unit tests sources and outputs
CXX_TST_SRCS    = $(wildcard $(CXX_TSTDIR)/*.cc)
CXX_TST_DEPS    = $(CXX_TST_SRCS:%.cc=%.d)
CXX_TST_OBJS    = $(CXX_TST_SRCS:%.cc=%.o)
CXX_TST_BINS    = $(CXX_TST_SRCS:%.cc=%)
CXX_TST_OKS     = $(CXX_TST_SRCS:%.cc=%.ok)

# Use gtest and cron to build tests.
$(CXX_TST_OBJS): CPPFLAGS += -I$(GTEST_INCDIR) -DGTEST_HAS_TR1_TUPLE=0
$(CXX_TST_BINS): $(CXX_LIB)
$(CXX_TST_BINS): LDLIBS += $(GTEST_LIB) $(CXX_LIB)

$(CXX_TST_DEPS): \
%.d: 			%.cc
	@echo "generating $@"; \
	set -e; $(CXX) $(CPPFLAGS) -MM $(CXX_TST_CPPFLAGS) $< | sed -E 's#([^ ]+:)#c++/test/\1#g' > $@

# Running tests.
%.ok:    	    	%
	@rm -f $@
	@echo testing $(shell basename $<) \
	&& (cd $(CXX_TSTDIR) && ./$(shell basename $<)) \
	&& touch $@

#-------------------------------------------------------------------------------
# Python configuration

# Python and tools
PYTHON	    	= python3
PYTEST	    	= py.test
PYTHON_CONFIG	= python3-config

# Directories
PY_DIR	    	= $(TOP)/python
PY_PKGDIR   	= $(PY_DIR)/cron
PY_PFXDIR    	= $(shell $(PYTHON_CONFIG) --prefix)

# Numpy
NPY_INCDIRS 	= $(shell $(PYTHON) -c 'from numpy.distutils.misc_util import get_numpy_include_dirs as g; print(" ".join(g()));')

#-------------------------------------------------------------------------------
# Python building extension code

# Sources and outputs
PY_SRCS   	= $(wildcard $(PY_PKGDIR)/*.cc)
PY_DEPS	    	= $(PY_SRCS:%.cc=%.d)
PY_OBJS	    	= $(PY_SRCS:%.cc=%.o)
PY_EXTMOD_SFX   = $(shell $(PYTHON) -c 'from importlib.machinery import EXTENSION_SUFFIXES as E; print(E[0]); ')
PY_EXTMOD	= $(PY_PKGDIR)/ext$(PY_EXTMOD_SFX)

# Compiling Python extension code.
$(PY_OBJS): CPPFLAGS += $(shell $(PYTHON_CONFIG) --includes)
$(PY_OBJS): CXXFLAGS += -fno-strict-aliasing -fwrapv
# FIXME: Remove this.
$(PY_OBJS): CXXFLAGS += -DNDEBUG
$(PY_OBJS): CPPFLAGS += $(NPY_INCDIRS:%=-I%)

# Linking Python exension modules.
$(PY_EXTMOD): $(PY_OBJS) $(CXX_LIB)
$(PY_EXTMOD): LDFLAGS += -L$(PY_PFXDIR)/lib

$(PY_DEPS): \
%.d: 		    	%.cc
	@echo "generating $@"; \
	set -e; $(CXX) -MM $(PY_CPPFLAGS) $< | sed 's,^\(.*\)\.o:,python/cron/\1.o:,g' > $@

# For compatibility and testing.
.PHONY: python-setuptools
python-setuptools:	$(CXX_LIB)
	cd python; $(PYTHON) setup.py build_ext --inplace

#-------------------------------------------------------------------------------
# zoneinfo

.PHONY: zoneinfo
zoneinfo:
	mkdir -p $(SHRDIR)
	tar jxf $(EXTDIR)/zoneinfo/zoneinfo-2016a.tar.bz2 -C $(SHRDIR)

# Use our zoneinfo directory for running tests.
export ZONEINFO = $(ABSTOP)/share/zoneinfo

#-------------------------------------------------------------------------------
# Phony targets

.PHONY: all
all:			cxx python 

.PHONY: test
test:			test-cxx test-python

.PHONY: clean
clean:			clean-cxx clean-python 

.PHONY: cxx
cxx:	    	    	$(CXX_LIB)

.PHONY: clean-cxx
clean-cxx:
	rm -f $(CXX_OBJS) $(CXX_LIB) $(CXX_BINS) $(OKS)

.PHONY: test-cxx-bins
test-cxx-bins:	    	$(CXX_TST_BINS)

.PHONY: test-cxx
test-cxx:   	    	$(CXX_TST_OKS)

.PHONY: python
python:			$(PY_DEPS) $(PY_EXTMOD)

.PHONY: clean-python
clean-python:
	rm -rf $(PY_OBJS) $(PY_EXTMOD)

.PHONY: test-python
test-python: 		$(PY_EXTMOD)
	$(PYTEST) python

.PHONY: testclean-python
testclean-python:

#-------------------------------------------------------------------------------

include $(CXX_DEPS) 
include $(CXX_TST_DEPS) 
include $(PY_DEPS)

