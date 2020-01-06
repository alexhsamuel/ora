#!/bin/bash
#
# Usage: update-zoninfo.sh VERSION
#
# Note: Recent distributions are lzipped.  Requires tar with lunzip support.

set -ex

root="$(cd "$(dirname $0)"/..; pwd)"
zoneinfo="$root"/python/ora/zoneinfo

tmpdir=$(mktemp -d)
cd $tmpdir

version=$1
if [[ -z "$version" ]]; then
    echo "usage: $0 VERSION" >&2
    exit 1
fi

curl https://data.iana.org/time-zones/releases/tzdb-$version.tar.lz -O

tar xf tzdb-$version.tar.lz
cd tzdb-$version

make LOCALTIME=UTC TOPDIR=$tmpdir/install INSTALL

rm -rf "$zoneinfo"
cp -r $tmpdir/install/usr/share/zoneinfo "$zoneinfo"
echo $version > "$zoneinfo"/+VERSION

rm -rf $tmpdir

