path PY ++ python

function c++() { /usr/bin/c++ -std=c++14 -Wall "$@"; }
function supdoc() { clear; py3 -m supdoc.cli "$@" | less -eF; }

function ora-build() { 
    local src="$1"
    local prog="${src%.cc}"
    c++ -I $HOME/dev/ora/cxx/include \
        "$src" \
        $CRON_HOME/cxx/src/libora.a \
        -o "$prog"
}

