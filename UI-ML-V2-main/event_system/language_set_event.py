
import json

from .app_event import AppEvent


class LanguageSetEvent(AppEvent):
    def trigger(self, filename):
        with open(filename) as f:
            j = json.load(f)
            return super().trigger(j)
