#!/bin/bash
set -ex

root="$(cd "$(dirname $0)"/..; pwd)"

tmpdir=$(mktemp -d)
tmpdir=$HOME/tmp/zoneinfo
cd $tmpdir

version=$1
if [[ -z "$version" ]]; then
    echo "usage: $0 VERSION" >&2
    exit 1
fi

wget https://data.iana.org/time-zones/releases/tzdb-$version.tar.lz

lzcat tzdb-$version.tar.lz | tar xf -

make LOCALTIME=UTC TOPDIR=$tmpdir/install INSTALL

rm -rf $root/share/zoneinfo
cp -r $tmpdir/install/ext/zoneinfo $root/share/

# rm -rf $tmpdir
