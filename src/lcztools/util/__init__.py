try:
    from tqdm import tqdm
except:
    def tqdm(iterator):
        print("Please install the real tqdm")
        yield from progress(iterator)

def progress(iterator):
    for idx, stuff in enumerate(iterator):
        if idx%50==0:
            if idx>0:
                print()
            print('{:6}: '.format(idx), end='')
        yield stuff
        print('.', end='')
    if idx%50!=0:
        print()
        print('{:6}: '.format(idx+1), end='')
    print("DONE")
            

def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    From: https://stevenloria.com/lazy-properties/
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property

