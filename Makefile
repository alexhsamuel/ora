ifeq ($(OS),Windows_NT)
  UNAME	       := Windows
else
  UNAME	       := $(shell uname -s)
endif

# GTEST_DIR       = ./googletest/googletest
# GTEST_INCDIR    = $(GTEST_DIR)/include
# GTEST_LIB       = $(GTEST_DIR)/make/gtest_main.a

CXX            := $(CXX) -std=c++14
CPPFLAGS        = -I./include
CXXFLAGS        = -fPIC -g -Wall
LDLIBS          = -lpthread

SOURCES         = $(wildcard src/*.cc) 
DEPS            = $(SOURCES:%.cc=%.dd)
OBJS            = $(SOURCES:%.cc=%.o)
LIB	    	= lib/libcron.a
BINS            = $(SOURCES:%.cc=%)

TEST_SOURCES    = $(wildcard test/*.cc)
TEST_DEPS       = $(TEST_SOURCES:%.cc=%.dd)
TEST_OBJS       = $(TEST_SOURCES:%.cc=%.o)
TEST_BINS       = $(TEST_SOURCES:%.cc=%.exe)
TEST_OKS        = $(TEST_SOURCES:%.cc=%.ok)

# TEST_CPPFLAGS   = $(CPPFLAGS) -I$(GTEST_INCDIR)
# TEST_LIBS       = $(GTEST_LIB) $(LIB)

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
PY_SOURCES   	= $(wildcard python/fixfmt/*.cc)
PY_DEPS	    	= $(PY_SOURCES:%.cc=%.dd)
PY_OBJS	    	= $(PY_SOURCES:%.cc=%.o)
PY_EXTMOD	= python/fixfmt/_ext.so

#-------------------------------------------------------------------------------

.PHONY: all
all:			$(LIB) test

.PHONY: test
test:			test-cxx test-python

.PHONY: clean
clean:			clean-cxx clean-python testclean

.PHONY: testclean
testclean:		testclean-cxx testclean-python

#-------------------------------------------------------------------------------
# C++

$(DEPS): \
%.dd: 		%.cc
	@echo "generating $@"; set -e; \
	$(CXX) -MM $(CPPFLAGS) $< | sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' > $@

.PHONY: clean-cxx
clean-cxx:
	rm -f $(OBJS) $(LIB) $(BINS) $(DEPS) $(OKS)

.PHONY: testclean
testclean-cxx:
	rm -f $(TEST_DEPS) $(TEST_OBJS) $(TEST_BINS) $(TEST_OKS)

.PHONY: test-cxx
test-cxx: $(TEST_OKS)

$(LIB):			$(OBJS)
	ar -r $@ $^

$(TEST_DEPS): \
%.dd: 			%.cc
	@echo "generating $@"; \
	set -e; $(CXX) $(CPPFLAGS) -MM $(TEST_CPPFLAGS) $< | sed -E 's#([^ ]+:)#test/\1#g' > $@

$(TEST_OBJS): \
%.o: 	    	    	%.cc
	$(CXX) $(TEST_CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(TEST_BINS): \
%.exe: 	    	    	%.o $(TEST_LIBS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(TEST_LIBS) $(LDLIBS) -o $@

$(TEST_OKS): \
%.ok:    	    	%.exe
	@rm -f $@
	$< && touch $@

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

$(PY_EXTMOD):		$(PY_OBJS) $(LIB)
	$(CXX) $(PY_LDFLAGS) $^ $(PY_LDLIBS) -o $@

.PHONY: test-python
test-python: 		$(PY_EXTMOD)
	$(PYTEST) python

.PHONY: testclean-python
testclean-python:

# For compatibility and testing.
.PHONY: python-setuptools
python-setuptools:	$(LIB)
	cd python; $(PYTHON) setup.py build_ext --inplace

#-------------------------------------------------------------------------------

include $(DEPS) 
include $(TEST_DEPS) 
include $(PY_DEPS)

