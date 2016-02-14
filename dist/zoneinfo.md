# zoneinfo data

The `zoneinfo` directory contains time zone information.  It is compiled from
information in the "Olsen"
[tz database](http://web.cs.ucla.edu/~eggert/tz/tz-link.htm) and in the public
domain.  The data are updated several times a year to reflect political changes
in time zone policy.

On Linux-like systems, a version is generally installed in
`/usr/share/zoneinfo`.  We include another copy here as the system copy might
not be up to date, and may also employ older formats for the data files (as is
the case with Darwin/OSX).


## Build instructions

To get and build the tz database and produce a new zoneinfo tarball, follow
these steps.

1. Download the most recent code and data tarballs:

  ```sh
wget --retr-symlinks 'ftp://ftp.iana.org/tz/tz*-latest.tar.gz'
```

1. Also download these missing "solar" files:

  ```sh
wget https://www.ietf.org/timezones/data/solar8{7,8,9}
```

1. Unpack the sources and data.

  ```sh
tar zxf tzcode-latest.tar.gz
tar zxf tzdata-latest.tar.gz
```

1. Build and install the code and data to a temporary location.

  ```sh
make LOCALTIME=UTC TOPDIR=$(pwd)/install INSTALL
```

1. Collect the zoneinfo outputs into a tarball.

  ```sh
tar jcf zoneinfo.tar.bz2 -C install/etc zoneinfo
```

