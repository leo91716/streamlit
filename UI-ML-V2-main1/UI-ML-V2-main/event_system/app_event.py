

class AppEvent:
    def __init__(self) -> None:
        self.__funcs = []

    def register(self, func):
        self.__funcs.append(func)

    def trigger(self, *args, **kwargs):
        for f in self.__funcs:
            f(*args, **kwargs)
