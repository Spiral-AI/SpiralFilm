# configは、繰り返す使える設定変数を指定します。
class FilmConfig:
    def __init__(
        self,
        model,
        name=None,
        temperature=1.0,
        top_p=1.0,
        n=1,
        stream=False,
        stop=None,
        max_tokens=None,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        user=None,
    ):
        self.model = model
        self.name = name
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.stop = stop
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.user = user

    def to_dict(self):
        """Converts the config object to a dictionary representation for API call."""
        config_dict = vars(self).copy()
        return {key: value for key, value in config_dict.items() if value is not None}
