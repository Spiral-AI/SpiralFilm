import time
import os
import logging
from copy import copy

logger = logging.getLogger(__name__)


# configは、繰り返し使える設定変数を指定します。
class FilmConfig:
    def __init__(
        self,
        model="gpt-3.5-turbo",
        api_type="openai",
        azure_deployment_id=None,
        azure_api_version=None,
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
        assert api_type in ["openai", "azure"]
        if api_type == "azure":
            assert (
                azure_deployment_id is not None
            ), "azure_deployment_id must be specified when api_type is azure. This is an ID provided in Azure portal."
            assert (
                azure_api_version is not None
            ), "azure_api_version must be specified when api_type is azure. Select from https://learn.microsoft.com/ja-jp/azure/ai-services/openai/reference"

        # https://learn.microsoft.com/ja-jp/azure/ai-services/openai/how-to/switching-endpoints

        self.model = model
        self.api_type = api_type
        self.azure_deployment_id = azure_deployment_id
        self.azure_api_version = azure_api_version
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        # self.stream = stream
        self.stop = [stop] if isinstance(stop, str) else stop
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
        self.selected_apikey = None  # required to track if this is openai or azure

    @staticmethod
    def create_from(config, **kwargs):
        new_config = copy(config)
        for key, value in kwargs.items():
            if hasattr(new_config, key):
                setattr(new_config, key, value)
            else:
                raise ValueError(f"Unknown attribute: {key}")

        return new_config

    def _wait_time(self, retry_count):
        if retry_count == 0:
            return 0.0
        else:
            return min(self.wait_time * (2 ** (retry_count - 1)), self.max_wait_time)

    def get_apikey(self):
        if self.apikeys == []:
            if self.api_type == "openai":
                # check if os.environ["OPENAI_API_KEY"] is set.
                if not "OPENAI_API_KEY" in os.environ:
                    raise ValueError(
                        "OPENAI_API_KEY is not set in environment variables."
                    )

                self.add_key(os.environ["OPENAI_API_KEY"])
                return (
                    self.get_apikey()
                )  # Doing some trick. This call envokes the else statements from the next lines.
            elif self.api_type == "azure":
                # check if the keys are set
                if ("AZURE_API_KEY" not in os.environ) or (
                    "AZURE_API_BASE" not in os.environ
                ):
                    raise ValueError(
                        "Set AZURE_API_KEY and AZURE_API_BASE in environment variables."
                    )

                self.add_key(
                    api_key=os.environ["AZURE_API_KEY"],
                    api_base=os.environ["AZURE_API_BASE"],
                )
                return (
                    self.get_apikey()
                )  # Doing some trick. This call envokes the else statements from the next lines.

        else:
            # Sort the apikeys by available time
            self.apikeys.sort(key=lambda x: x["available_time"])
            # Get the first apikey
            apikey = self.apikeys[0]
            time_to_wait = apikey["available_time"] - time.mktime(time.gmtime())

            self.selected_apikey = apikey
            return apikey, time_to_wait

    def update_apikey(self, apikey, status):
        for item in self.apikeys:
            if item["api_key"] == apikey["api_key"]:
                if status == "success":
                    item["last_called"] = time.mktime(time.gmtime())
                    item["retry_count"] = 0
                    item["available_time"] = item["last_called"] + self._wait_time(
                        item["retry_count"]
                    )
                elif status == "failure":
                    item["last_called"] = time.mktime(time.gmtime())
                    item["retry_count"] += 1
                    item["available_time"] = item["last_called"] + self._wait_time(
                        item["retry_count"]
                    )
                else:
                    raise ValueError("status must be either 'success' or 'failure'")
                break

    def add_key(
        self,
        api_key,
        api_base=None,
    ):
        api_key_and_base = []
        if self.api_type == "azure":
            assert (
                api_base is not None
            ), "api_base must be specified when api_type is azure. This is an endpoint URL provided in Azure portal."

            # If multiple api_key is specified, api_base should be the comma separated list with the same number of items.
            assert len(api_key.split(",")) == len(
                api_base.split(",")
            ), "The number of api_key and api_base must be the same."

            api_key_and_base = list(zip(api_key.split(","), api_base.split(",")))
        else:
            api_keys = api_key.split(",")
            api_key_and_base = [(api_key, None) for api_key in api_keys]

        # for api_key_item, api_base_item in zip(api_key.split(","), api_base.split(",")):
        for api_key_item, api_base_item in api_key_and_base:
            item = {}
            item["api_key"] = api_key_item.strip()
            item["api_base"] = (
                api_base_item.strip() if api_base_item is not None else None
            )
            item["last_called"] = time.mktime(time.gmtime())
            item["retry_count"] = 0
            item["available_time"] = item["last_called"] + self._wait_time(
                item["retry_count"]
            )
            self.apikeys.append(item)

    def to_dict(self, mode="APICalling", override_params={}):
        """
        Converts the config object to a dictionary representation for API call.

        Args:
            mode (str): Either "APICalling" or "Caching". If "APICalling", the dictionary will contain all the parameters required for API call. If "Caching", the dictionary will contain only the parameters required for caching.
            override_params (dict): A dictionary of parameters to override the config object. This is useful when you want to override the parameters for a specific API call.
        """
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

        result = {key: value for key, value in config_dict.items() if key in keys}

        # take care of azure
        if self.api_type == "azure":
            del result["model"]
            result["deployment_id"] = self.azure_deployment_id

        # override parameters
        for key, value in override_params.items():
            assert key in keys, f"override_params contain invalid parameter: {key}"
            result[key] = value

        return result


class FilmEmbedConfig(FilmConfig):
    def __init__(
        self,
        model="text-embedding-ada-002",
        api_type="openai",
        azure_deployment_id=None,
        azure_api_version=None,
        use_cache=False,
    ):
        super().__init__(
            model=model,
            use_cache=use_cache,
            api_type=api_type,
            azure_deployment_id=azure_deployment_id,
            azure_api_version=azure_api_version,
        )

    def to_dict(self, mode="APICalling", override_params={}):
        """
        Converts the config object to a dictionary representation for API call.
        """
        assert mode in ["APICalling", "Caching"]

        config_dict = vars(self).copy()

        keys = []
        if mode == "APICalling":
            keys = ["model"]
        elif mode == "Caching":
            keys = ["model"]

        result = {key: value for key, value in config_dict.items() if key in keys}
        # take care of azure
        if self.api_type == "azure":
            del result["model"]
            result["deployment_id"] = self.azure_deployment_id

        # override parameters
        for key, value in override_params.items():
            assert key in keys, f"override_params contain invalid parameter: {key}"
            result[key] = value

        return result
