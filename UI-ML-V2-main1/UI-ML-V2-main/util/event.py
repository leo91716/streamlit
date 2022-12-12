

class Event:
    def __init__(self) -> None:
        self.__func_list = []

    def invoke(self, *args, **kwargs):
        for f in self.__func_list:
            f(*args, **kwargs)

    def addListener(self, func):
        self.__func_list.append(func)
        return func
