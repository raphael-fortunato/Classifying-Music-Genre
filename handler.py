import logging
from ts.torch_handler.base_handler import BaseHandler


class SongBaseHandler(BaseHandler):
    def preprocess(self, requests):
        return requests
