ifeq ($(OS),Windows_NT)
  UNAME	       := Windows
else
  UNAME	       := $(shell uname -s)
endif

TOP 	    	= $(shell pwd)
EXTDIR	    	= $(TOP)/external
SHRDIR	    	= $(TOP)/share

#-------------------------------------------------------------------------------
# C++ configuration

# Directories
CXX_DIR	    	= $(TOP)/c++
CXX_INCDIR 	= $(CXX_DIR)/include
CXX_SRCDIR 	= $(CXX_DIR)/src
CXX_LIBDIR 	= $(CXX_DIR)/lib
CXX_BINDIR 	= $(CXX_DIR)/bin
CXX_TSTDIR    	= $(CXX_DIR)/test

# Gtest configuration
GTEST_DIR       = $(EXTDIR)/gtest
GTEST_INCDIR    = $(GTEST_DIR)/include
GTEST_LIB       = $(GTEST_DIR)/gtest_main.a

# Compiler and linker
CXX            := $(CXX) -std=c++14
CPPFLAGS        = -I$(CXX_INCDIR)
CXXFLAGS        = -fPIC -g -Wall
LDLIBS          = -lpthread

CXX_TST_CPPFLAGS= $(CPPFLAGS) -I$(GTEST_INCDIR) -DGTEST_HAS_TR1_TUPLE=0
CXX_TST_LIBS    = $(GTEST_LIB) $(CXX_LIB)

# Sources and outputs
CXX_SRCS        = $(wildcard $(CXX_SRCDIR)/*.cc) 
CXX_DEPS        = $(CXX_SRCS:%.cc=%.dd)
CXX_OBJS        = $(CXX_SRCS:%.cc=%.o)
CXX_LIB	    	= $(CXX_LIBDIR)/libcron.a
CXX_BIN_SRCS	= $(wildcard $(CXX_SRCDIR)/bin/*.cc)
CXX_BINS        = $(CXX_BIN_SRCS:%.cc=%)

# Unit tests sources and outputs
CXX_TST_SRCS    = $(wildcard $(CXX_TSTDIR)/*.cc)
CXX_TST_DEPS    = $(CXX_TST_SRCS:%.cc=%.dd)
CXX_TST_OBJS    = $(CXX_TST_SRCS:%.cc=%.o)
CXX_TST_BINS    = $(CXX_TST_SRCS:%.cc=%.exe)
CXX_TST_OKS     = $(CXX_TST_SRCS:%.cc=%.ok)

#-------------------------------------------------------------------------------
# Python configuration

PYTHON	    	= python3
PYTEST	    	= py.test
PYTHON_CONFIG	= python3-config
PY_PREFIX    	= $(shell $(PYTHON_CONFIG) --prefix)
PY_CPPFLAGS  	= $(CPPFLAGS) $(shell $(PYTHON_CONFIG) --includes)
PY_CXXFLAGS  	= $(CXXFLAGS) -DNDEBUG -fno-strict-aliasing -fwrapv
PY_LDFLAGS   	= -L$(PY_PREFIX)/lib
ifeq ($(UNAME),Darwin)
  PY_LDFLAGS   += -bundle -undefined dynamic_lookup
else ifeq ($(UNAME),Linux)
  PY_LDFLAGS   += -shared
endif
PY_LDLIBS	= 
PY_SRCS   	= $(wildcard python/fixfmt/*.cc)
PY_DEPS	    	= $(PY_SRCS:%.cc=%.dd)
PY_OBJS	    	= $(PY_SRCS:%.cc=%.o)
PY_EXTMOD	= python/fixfmt/_ext.so

TZCODE_DIST 	= $(TOP)/dist/tzcode2013c.tar.gz
TZDATA_DIST	= $(TOP)/dist/tzdata2016a.tar.gz
SOLAR_DIST  	= $(TOP)/dist/solar.tar.gz

#-------------------------------------------------------------------------------

.PHONY: all
all:			$(CXX_LIB) test

.PHONY: test
test:			test-cxx 

.PHONY: clean
clean:			clean-cxx testclean

.PHONY: testclean
testclean:		testclean-cxx 

#-------------------------------------------------------------------------------
# C++

$(CXX_DEPS): \
%.dd: 		%.cc
	@echo "generating $@"; set -e; \
	$(CXX) -MM $(CPPFLAGS) $< | sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' > $@

.PHONY: clean-cxx
clean-cxx:
	rm -f $(CXX_OBJS) $(CXX_LIB) $(CXX_BINS) $(CXX_DEPS) $(OKS)

.PHONY: testclean
testclean-cxx:
	rm -f $(CXX_TST_DEPS) $(CXX_TST_OBJS) $(CXX_TST_BINS) $(CXX_TST_OKS)

.PHONY: test-cxx
test-cxx: $(CXX_TST_OKS)

$(CXX_LIB):		$(CXX_OBJS)
	mkdir -p $(shell dirname $@)
	ar -r $@ $^

$(CXX_BINS): \
src/bin/%:	   	src/bin/%.cc $(CXX_LIB)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDLIBS) -o $@

$(CXX_TST_DEPS): \
%.dd: 			%.cc
	@echo "generating $@"; \
	set -e; $(CXX) $(CPPFLAGS) -MM $(CXX_TST_CPPFLAGS) $< | sed -E 's#([^ ]+:)#test/\1#g' > $@

$(CXX_TST_OBJS): \
%.o: 	    	    	%.cc
	$(CXX) $(CXX_TST_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(CXX_TST_BINS): \
%.exe: 	    	    	%.o $(CXX_TST_LIBS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(CXX_TST_LIBS) $(LDLIBS) -o $@

$(CXX_TST_OKS): \
%.ok:    	    	%.exe
	@rm -f $@
	@echo testing $(shell basename $<) && \
	  cd $(CXX_TSTDIR) && $< && touch $@

# Use our zoneinfo directory for running tests.
$(CXX_TST_OKS): export ZONEINFO = $(TOP)/share/zoneinfo

#-------------------------------------------------------------------------------
# Python

.PHONY: python
python:			$(PY_EXTMOD)

.PHONY: clean-python
clean-python:
	rm -rf $(PY_DEPS) $(PY_OBJS) $(PY_EXTMOD)

$(PY_DEPS): \
%.dd: 		    	%.cc
	$(CXX) -MM $(PY_CPPFLAGS) $< | sed 's,^\(.*\)\.o:,python/fixfmt/\1.o:,g' > $@

$(PY_OBJS): \
%.o:			%.cc
	$(CXX) $(PY_CPPFLAGS) $(PY_CXXFLAGS) -c $< -o $@

$(PY_EXTMOD):		$(PY_OBJS) $(CXX_LIB)
	$(CXX) $(PY_LDFLAGS) $^ $(PY_LDLIBS) -o $@

.PHONY: test-python
test-python: 		$(PY_EXTMOD)
	$(PYTEST) python

.PHONY: testclean-python
testclean-python:
	:

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

#-------------------------------------------------------------------------------

include $(CXX_DEPS) 
include $(CXX_TST_DEPS) 
include $(PY_DEPS)

