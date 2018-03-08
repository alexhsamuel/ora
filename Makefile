ifeq ($(OS),Windows_NT)
  UNAME	       := Windows
else
  UNAME	       := $(shell uname -s)
endif

ifeq ($(UNAME),Darwin)
  export MACOSX_DEPLOYMENT_TARGET = 10.9
endif

# Default target.
all:

#-------------------------------------------------------------------------------
# Locations

TOP 	    	= .
# FIXME: Hack.  But we don't have realpath on OSX, do we?
ABSTOP	    	= $(shell pwd)
EXTDIR	    	= $(TOP)/external

#-------------------------------------------------------------------------------
# C++ configuration

# Directories
CXX_DIR	    	= $(TOP)/cxx
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
CXXFLAGS    	= -g -Wall -fdiagnostics-color=always -O3
override CXXFLAGS += -fpic
CXX_DEPFLAGS	= -MT $@ -MMD -MP -MF $<.d
LDFLAGS	    	= 
LDLIBS          = 

ifeq ($(UNAME),Linux)
  CXXFLAGS     += -pthread
  LDLIBS       += -lpthread
endif

#-------------------------------------------------------------------------------
# Python configuration

# Python and tools
PYTHON	    	= python3
PYTEST	    	= py.test
PYTHON_CONFIG	= python3-config

ORA_NUMPY      ?= no

# Directories
PY_DIR	    	= $(TOP)/python
PY_PKGDIR   	= $(PY_DIR)/ora
PY_PFXDIR      := $(shell $(PYTHON_CONFIG) --prefix)

# Script to wrap docstrings as C++ string literals.
WRAP_DOCSTRINGS	= $(PY_DIR)/wrap_docstrings

#-------------------------------------------------------------------------------
# gtest

$(GTEST_LIB):	    $(GTEST_DIR)
	$(MAKE) -C $< $(notdir $@)

#-------------------------------------------------------------------------------
# C++ building and linking

# Sources and outputs
CXX_SRCS        = $(wildcard $(CXX_SRCDIR)/*.cc) 
DEPS	       += $(CXX_SRCS:%.cc=%.cc.d)
CXX_OBJS        = $(CXX_SRCS:%.cc=%.o)
CXX_LIB	    	= $(CXX_SRCDIR)/libora.a
CXX_BIN_SRCS	= $(wildcard $(CXX_SRCDIR)/bin/*.cc)
CXX_BINS        = $(CXX_BIN_SRCS:%.cc=%)

# How to compile a C++ file, and generate automatic dependencies.
%.o:	    	    	%.cc
%.o:	    	    	%.cc %.cc.d
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CXX_DEPFLAGS) $< -c -o $@

# How to generate assember for C++ files.
%.s:	    	    	%.cc force
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -S -o $@

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
DEPS           += $(CXX_TST_SRCS:%.cc=%.cc.d)
CXX_TST_OBJS    = $(CXX_TST_SRCS:%.cc=%.o)
CXX_TST_BINS    = $(CXX_TST_SRCS:%.cc=%)
CXX_TST_OKS     = $(CXX_TST_SRCS:%.cc=%.ok)

# Use gtest and ora to build tests.
$(CXX_TST_OBJS): CPPFLAGS += -I$(GTEST_INCDIR) -DGTEST_HAS_TR1_TUPLE=0
$(CXX_TST_BINS): $(GTEST_LIB) $(CXX_LIB)
$(CXX_TST_BINS): LDLIBS += $(GTEST_LIB) $(CXX_LIB)

# Use our zoneinfo directory for running tests.
ZONEINFO_DIR 	= $(PY_DIR)/ora/zoneinfo
$(CXX_TST_OKS): export ZONEINFO = $(ABSTOP)/$(ZONEINFO_DIR)

# Running tests.
%.ok:    	    	%
	@rm -f $@
	@echo testing $(shell basename $<) \
	&& (cd $(CXX_TSTDIR) && ./$(shell basename $<)) \
	&& touch $@

#-------------------------------------------------------------------------------
# Python building extension code

# Sources
PY_INCDIRS      = $(PY_PKGDIR)
PY_SRCS         = $(wildcard $(PY_PKGDIR)/*.cc) 
PY_CPPFLAGS    += $(PY_INCDIRS:%=-I%)
PY_CPPFLAGS    += $(shell $(PYTHON_CONFIG) --includes)
PY_CXXFLAGS    += -fno-strict-aliasing -fwrapv
PY_CXXFLAGS    += -DNDEBUG  # FIXME: Remove.
PY_DOCSTR       = $(wildcard $(PY_PKGDIR)/*.docstrings)

# Building with NumPy support
ifeq ($(ORA_NUMPY),yes)
  PY_INCDIRS   += $(shell $(PYTHON) -c 'from numpy.distutils.misc_util import get_numpy_include_dirs as g; print(" ".join(g()));')
  PY_SRCS      += $(wildcard $(PY_PKGDIR)/np/*.cc)
  PY_CPPFLAGS  += -DORA_NUMPY
endif

# Outputs
DEPS           += $(PY_SRCS:%.cc=%.cc.d)
PY_OBJS         = $(PY_SRCS:%.cc=%.o)
PY_EXTMOD_SFX   = $(shell $(PYTHON) -c 'from importlib.machinery import EXTENSION_SUFFIXES as E; print(E[0]); ')
PY_EXTMOD       = $(PY_PKGDIR)/ext$(PY_EXTMOD_SFX)
PY_DOCSTR_CC    = $(PY_DOCSTR:%.docstrings=%.docstrings.cc.inc)
PY_DOCSTR_HH    = $(PY_DOCSTR:%.docstrings=%.docstrings.hh.inc)

# Compiling Python extension code.
$(PY_OBJS): CPPFLAGS += $(PY_CPPFLAGS)
$(PY_OBJS): CXXFLAGS += $(PY_CXXFLAGS)

# Linking Python exension modules.
$(PY_EXTMOD): 	    	$(PY_OBJS) $(CXX_LIB)
$(PY_EXTMOD): LDFLAGS += -L$(PY_PFXDIR)/lib

# Wrapping Python extension docstrings as C++ string literals.
$(PY_DOCSTR_CC): %.cc.inc: % $(WRAP_DOCSTRINGS)
	$(WRAP_DOCSTRINGS) $<
$(PY_DOCSTR_HH): %.hh.inc: % $(WRAP_DOCSTRINGS)
	$(WRAP_DOCSTRINGS) $<
# Require the processed docstring sources to build objects.
$(PY_OBJS):    	    	$(PY_DOCSTR_CC) $(PY_DOCSTR_HH)

# For compatibility and testing.
.PHONY: python-setuptools
python-setuptools:	$(CXX_LIB)
	cd python; $(PYTHON) setup.py build_ext --inplace

#-------------------------------------------------------------------------------
# Phony targets

.PHONY: all
all:			cxx python

.PHONY: test
test:			test-cxx test-python

.PHONY: clean
clean:			clean-cxx clean-python

.PHONY: depclean
depclean:   	    	
	rm -f $(DEPS)

.PHONY: cxx
cxx:	    	    	$(CXX_LIB)

.PHONY: clean-cxx
clean-cxx:
	rm -f $(CXX_OBJS) $(CXX_LIB) $(CXX_BINS) $(CXX_TST_OBJS) \
	      $(CXX_TST_BINS) $(CXX_TST_OKS)
	$(MAKE) -C $(GTEST_DIR) clean

.PHONY: test-cxx-bins
test-cxx-bins:	    	$(CXX_TST_BINS)

.PHONY: test-cxx
test-cxx:   	    	$(CXX_TST_OKS)

.PHONY: python
python:			$(PY_EXTMOD)

.PHONY: clean-python
clean-python:
	rm -rf $(PY_OBJS) $(PY_EXTMOD) $(PY_DOCSTR_CC) $(PY_DOCSTR_HH)

.PHONY: docstrings
docstrings: 	    	$(PY_DOCSTR_CC) $(PY_DOCSTR_HH)

.PHONY: test-python
test-python: 		$(PY_EXTMOD)
	$(PYTEST) python

# Use this target as a dependency to force another target to be rebuilt.
.PHONY: force
force: ;

.PHONY: fixmes
fixmes:
	@find $(CXX_DIR) $(PY_DIR) -name \*.cc -o -name \*.hh -o -name \*.py \
	| xargs grep FIXME

#-------------------------------------------------------------------------------

# Include autodependency makefles.
%.d: ;
.PRECIOUS: $(DEPS)
-include $(DEPS) 

