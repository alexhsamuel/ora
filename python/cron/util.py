def format_call(__fn, *args, **kw_args):
    """
    Formats a function call, with arguments, as a string.

      >>> format_call(open, "data.csv", mode="r")
      "open('data.csv', mode='r')"

    @param __fn
      The function to call, or its name.
    @rtype
       `str`
    """
    try:
        name = __fn.__name__
    except AttributeError:
        name = str(__fn)
    args = [ repr(a) for a in args ]
    args.extend( n + "=" + repr(v) for n, v in kw_args.items() )
    return "{}({})".format(name, ", ".join(args))


def format_ctor(obj, *args, **kw_args):
    return format_call(obj.__class__, *args, **kw_args)


