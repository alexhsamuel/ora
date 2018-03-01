Ora, a time and date library
============================

.. toctree::
   :maxdepth: 1
   :caption: Contents

   motivation
   times
   dates
   time zones
   localization
   limitations


Tour
----

::

    >>> from ora import *
    >>> time = now()
    >>> print(time)
    2018-02-27T03:07:08.29307700+00:00

    >>> z = TimeZone("America/New_York")
    >>> date, daytime = time @ z
    >>> print(date)
    2018-01-21



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

