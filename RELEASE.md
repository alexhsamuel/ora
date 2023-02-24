# v0.8

- Ora now chooses a zoneinfo directory similarly to the standard `zoneinfo`
  module.  By default, it uses zoneinfo directory specified by the Python build,
  if available; else, the data installed with the `tzdata` module, if available;
  else the zoneinfo directory packaged with Ora itself.

