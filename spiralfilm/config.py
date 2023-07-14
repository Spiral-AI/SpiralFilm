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
        max_retries=100,
        timeout=10,
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
        self.use_cache = use_cache
        self.cache_path = ".cache.pickle"

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
