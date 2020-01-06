path PY ++ $(dirname ${BASH_SOURCE[0]})/python

function c++() { /usr/bin/c++ -std=c++14 -Wall "$@"; }
function supdoc() { clear; py3 -m supdoc.cli "$@" | less -eF; }

