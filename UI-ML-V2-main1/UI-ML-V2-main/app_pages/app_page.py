
class AppPage:
    @classmethod
    def run(cls):
        cls._run_page()

    @staticmethod
    def _run_page():
        raise NotImplementedError()

    @staticmethod
    def get_name():
        return 'App Page'
