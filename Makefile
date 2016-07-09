ifeq ($(OS),Windows_NT)
  UNAME	       := Windows
else
  UNAME	       := $(shell uname -s)
endif

# Default target.
all:

#-------------------------------------------------------------------------------
# Locations

TOP 	    	= .
# FIXME: Hack.  But we don't have realpath on OSX, do we?
ABSTOP	    	= $(shell pwd)
EXTDIR	    	= $(TOP)/external
SHRDIR	    	= $(TOP)/share

ZONEINFO_DIR 	= $(SHRDIR)/zoneinfo

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
CXXFLAGS    	= -g -Wall -fdiagnostics-color=always
override CXXFLAGS += -fPIC
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

# Directories
PY_DIR	    	= $(TOP)/python
PY_PKGDIR   	= $(PY_DIR)/cron
PY_PFXDIR      := $(shell $(PYTHON_CONFIG) --prefix)
NPY_INCDIRS    := $(shell $(PYTHON) -c 'from numpy.distutils.misc_util import get_numpy_include_dirs as g; print(" ".join(g()));')

#-------------------------------------------------------------------------------
# gtest

$(GTEST_LIB):	    $(GTEST_DIR)
	$(MAKE) -C $< $(notdir $@)

#-------------------------------------------------------------------------------
# zoneinfo

$(ZONEINFO_DIR):
	rm -rf $@
	mkdir -p $(dir $@)
	tar jxf $(EXTDIR)/zoneinfo/zoneinfo-2016a.tar.bz2 -C $(dir $@)

#-------------------------------------------------------------------------------
# C++ building and linking

# Sources and outputs
CXX_SRCS        = $(wildcard $(CXX_SRCDIR)/*.cc) 
DEPS	       += $(CXX_SRCS:%.cc=%.cc.d)
CXX_OBJS        = $(CXX_SRCS:%.cc=%.o)
CXX_LIB	    	= $(CXX_SRCDIR)/libcron.a
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

# Use gtest and cron to build tests.
$(CXX_TST_OBJS): CPPFLAGS += -I$(GTEST_INCDIR) -DGTEST_HAS_TR1_TUPLE=0
$(CXX_TST_BINS): $(GTEST_LIB) $(CXX_LIB)
$(CXX_TST_BINS): LDLIBS += $(GTEST_LIB) $(CXX_LIB)

# Use our zoneinfo directory for running tests.
$(CXX_TST_OKS):	    	$(ZONEINFO_DIR)
$(CXX_TST_OKS): export ZONEINFO = $(ABSTOP)/$(ZONEINFO_DIR)

# Running tests.
%.ok:    	    	%
	@rm -f $@
	@echo testing $(shell basename $<) \
	&& (cd $(CXX_TSTDIR) && ./$(shell basename $<)) \
	&& touch $@

#-------------------------------------------------------------------------------
# Python building extension code

# Sources and outputs
PY_SRCS   	= $(wildcard $(PY_PKGDIR)/*.cc)
DEPS           += $(PY_SRCS:%.cc=%.cc.d)
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
$(PY_EXTMOD): 	    	$(PY_OBJS) $(CXX_LIB)
$(PY_EXTMOD): LDFLAGS += -L$(PY_PFXDIR)/lib

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

.PHONY: test-cxx-bins
test-cxx-bins:	    	$(CXX_TST_BINS)

.PHONY: test-cxx
test-cxx:   	    	$(CXX_TST_OKS)

.PHONY: python
python:			$(PY_EXTMOD)

.PHONY: clean-python
clean-python:
	rm -rf $(PY_OBJS) $(PY_EXTMOD)

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

