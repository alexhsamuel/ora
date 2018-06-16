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


#-------------------------------------------------------------------------------

class Range:
    """
    Analogous to build-in `range`, but works for types other than `int`.
    """

    def __init__(self, start, stop, step=1):
        if step == 0:
            raise ValueError("step may not be zero")
        self.start = start
        self.stop = stop
        self.step = step


    def __repr__(self):
        return (
            format_ctor(self, self.start, self.stop) if self.step == 1
            else format_ctor(self, self.start, self.stop, self.step)
        )


    def __len__(self):
        return (self.stop - self.start) // self.step


    def __getitem__(self, idx):
        length = len(self)
        if -length <= idx < 0:
            idx += length
        if idx < length:
            return self.start + self.step * idx
        else:
            raise IndexError(f"index {idx} out of range")


    def __iter__(self):
        i = self.start
        if self.step > 0:
            while i < self.stop:
                yield i
                i += self.step
        else:
            while i > self.stop:
                yield i
                i += self.step


    def __reversed__(self):
        return type(self)(self[-1], self.start - 1, -self.step)



