. $(dirname $BASH_ARGV)/cfg.sh
export PATH=$PYTHON_HOME/bin:$PATH
export PYTHONPATH=$CRON_HOME/python:$SUPDOC_HOME/src/python
export PYTHONSTARTUP=$CRON_HOME/pythonstartup
export ZONEINFO=$CRON_HOME/share/zoneinfo
function py3() { $PYTHON_HOME/bin/python3 -q "$@"; }
function supdoc() { clear; py3 -m supdoc.cli "$@" | less -eF; }

function cron-build() { 
    local src="$1"
    local prog="${src%.cc}"
    c++ -std=c++14 -Wall \
         -I $CRON_HOME/c++/include \
         "$src" $CRON_HOME/c++/src/libcron.a \
         -o "$prog"
}

