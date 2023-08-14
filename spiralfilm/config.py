import time, random, os


# configは、繰り返し使える設定変数を指定します。
class FilmConfig:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        temperature=0.0,
        top_p=1.0,
        n=1,
        # stream=False,
        stop=None,
        max_tokens=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        max_retries=100000,
        timeout=10,
        max_queues=10,
        use_cache=False,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        # self.stream = stream
        self.stop = stop
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_retries = max_retries
        self.timeout = timeout
        self.wait_time = 1.0
        self.max_wait_time = 60.0
        self.use_cache = use_cache
        self.max_queues = max_queues
        self.cache_path = ".cache.pickle"
        self.apikeys = []

    def _wait_time(self, retry_count):
        if retry_count == 0:
            return 0.0
        else:
            return min(self.wait_time * (2 ** (retry_count - 1)), self.max_wait_time)

    def get_apikey(self):
        if self.apikeys == []:
            return os.environ["OPENAI_API_KEY"], 0.0
        else:
            # Find the apikey with the earliest available time
            # Sort the apikeys by available time
            self.apikeys.sort(key=lambda x: x["available_time"])
            # Get the first apikey
            apikey = self.apikeys[0]
            time_to_wait = apikey["available_time"] - time.mktime(time.gmtime())
            return apikey["apikey"], time_to_wait

    def update_apikey(self, apikey, status):
        for item in self.apikeys:
            if item["apikey"] == apikey:
                if status == "success":
                    item["last_called"] = time.mktime(time.gmtime())
                    item["retry_count"] = 0
                    item["available_time"] = item["last_called"] + self._wait_time(
                        item["retry_count"]
                    )
                elif status == "failure":
                    item["retry_count"] += 1
                    item["available_time"] = item["last_called"] + self._wait_time(
                        item["retry_count"]
                    )
                else:
                    raise ValueError("status must be either 'success' or 'failure'")
                break
        else:
            # may be default api key
            pass

    def add_key(self, apikey):
        item = {}
        item["type"] = "openai"
        item["apikey"] = apikey
        item["last_called"] = time.mktime(time.gmtime())
        item["retry_count"] = 0
        item["available_time"] = item["last_called"] + self._wait_time(
            item["retry_count"]
        )
        self.apikeys.append(item)

    def to_dict(self, mode="APICalling"):
        """Converts the config object to a dictionary representation for API call."""
        assert mode in ["APICalling", "Caching"]

        config_dict = vars(self).copy()

        keys_APICalling = [
            "model",
            "temperature",
            "top_p",
            "n",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "timeout",
        ]

        keys_Caching = [
            "model",
            "temperature",
            "top_p",
            "n",
            "stop",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
        ]

        keys = []
        if mode == "APICalling":
            keys = keys_APICalling
        elif mode == "Caching":
            keys = keys_Caching

        return {key: value for key, value in config_dict.items() if key in keys}


class FilmEmbedConfig(FilmConfig):
    def __init__(
        self,
        model="text-embedding-ada-002",
        use_cache=False,
    ):
        super().__init__(
            model=model,
            use_cache=use_cache,
        )

    def to_dict(self, mode="APICalling"):
        """Converts the config object to a dictionary representation for API call."""
        assert mode in ["APICalling", "Caching"]

        config_dict = vars(self).copy()

        keys = []
        if mode == "APICalling":
            keys = ["model"]
        elif mode == "Caching":
            keys = ["model"]

        return {key: value for key, value in config_dict.items() if key in keys}
