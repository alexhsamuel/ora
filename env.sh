. $(dirname $BASH_ARGV)/cfg.sh
export PATH=$PYTHON_HOME/bin:$PATH
export PYTHONPATH=$PLYNTH_HOME:$CRON_HOME/python
export PYTHONSTARTUP=$CRON_HOME/pythonstartup
export ZONEINFO=$CRON_HOME/share/zoneinfo
function py3() { $PYTHON_HOME/bin/python3 -q "$@"; }
