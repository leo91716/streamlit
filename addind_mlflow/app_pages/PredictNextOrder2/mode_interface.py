

class RunMode:
    def __init__(self, display_name) -> None:
        self.__display_name = display_name

    def show_options(self):
        pass

    def run(self, do_update):
        pass

    def get_display_name(self):
        return self.__display_name
