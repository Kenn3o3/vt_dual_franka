class _DisabledRun:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def finish(self):
        return None


def login(*args, **kwargs):
    del args, kwargs
    return None


def init(*args, **kwargs):
    del args, kwargs
    return _DisabledRun()


def log(*args, **kwargs):
    del args, kwargs
    return None


def get_run():
    return None


def finish(*args, **kwargs):
    del args, kwargs
    return None


def define_metric(*args, **kwargs):
    del args, kwargs
    return None


class Video(_DisabledRun):
    pass


class Image(_DisabledRun):
    pass
