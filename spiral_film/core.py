# このクラスは、以下の仕様に従って作られます
# 目的: OpenAI APIの薄いラッパーとして機能する
# 機能: 以下の機能を提供する
# - Automatic retry
# - Placeholder functionality
# - Confirmation of sent prompts, time measurement features, and logging
# - Generation of appropriate exceptions
# 呼び出し方: 以下のように呼び出す
# from spiral_film import FilmCore, FilmConfig
# ...
# prompt = """...{{variable1}} ... {{variable2}}..."""
# myconfig = FilmConfig(model="gpt-4", temperature=0.9, max_tokens=100)
# fc = FilmCore(prompt, config=myconfig)
# result = fc({"variable1":"summer", "variable2":"hot"})


import time
import logging
import openai
import tiktoken
import re
import logging
import os
import pickle
import hashlib
from .config import FilmConfig


class FilmCore:
    def __init__(
        self,
        prompt,
        history=[],
        system_prompt=None,
        config=FilmConfig(),
    ):
        """
        Args:
            prompt (str): The prompt to be sent to the API.
            history (list): A list of messages to be sent to the API before the prompt.
                            It is expected that the user's input and the system's response will alternate.
            system_prompt (str): The system prompt to be sent to the API. If None, default prompt is used.
            config (FilmConfig): A FilmConfig object.

        """
        assert (
            len(history) % 2 == 0
        ), "History must contain an even number of messages. The user's input and the system's response will alternate."

        assert config.model.startswith("gpt-4") or config.model.startswith(
            "gpt-3.5-turbo"
        ), "Only GPT-4 and GPT-3.5-turbo are supported."

        self.history = history
        self.prompt = prompt  # user prompt
        if system_prompt is None:
            self.system_prompt = "You're a helpful assistant."
        else:
            self.system_prompt = system_prompt
        self.config = config
        self.wait_time = [1, 3, 5, 10]  # time to wait before retrying

        self.result = None
        self.result_message = None
        self.finished_reason = None
        self.token_usages = None
        self.is_cache_hit = None

        if self.config.use_cache:
            start_time = time.time()
            if os.path.exists(self.config.cache_path):
                with open(self.config.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            else:
                self.cache = {}

            end_time = time.time()

            if end_time - start_time > 1.0:
                logger.warning(
                    f"cache loading time = {end_time - start_time} sec is too long."
                    + f"Consider deleting cache file ({self.config.cache_path})"
                )

    def run(self, placeholders={}):
        """Run the API with the given placeholders and config.
        Args:
            placeholders (dict): A dictionary of placeholders and their values.
        Returns:
            The result of the API call.

        Usage:
            fc = FilmCore(prompt, config=myconfig).run(
                    {"variable1":"summer", "variable2":"hot"}
                )

        Reference for the returnd results from OpenAI ChatCompletion, which will be stored in self.result
        {
            "choices": [
                {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
                    "role": "assistant"
                }
                }
            ],
            "created": 1677664795,
            "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
            "model": "gpt-3.5-turbo-0613",
            "object": "chat.completion",
            "usage": {
                "completion_tokens": 17,
                "prompt_tokens": 57,
                "total_tokens": 74
            }
        }
        """
        prompt = self._placeholder(self.prompt, placeholders)
        messages = self._messages(prompt, self.history, self.system_prompt)
        self.result_prompt = prompt
        self.result_messages = messages

        start_time = time.time()

        # check Cache
        self.is_cache_hit = False
        if self.config.use_cache:
            message_hash = hashlib.md5(
                (str(messages) + str(self.config.to_dict(mode="Caching"))).encode()
            ).hexdigest()
            if message_hash in self.cache:
                logging.info("Cache hit.")
                self.result = self.cache[message_hash]
                self.is_cache_hit = True

        # call API if cache is not hit
        if self.is_cache_hit == False:
            self.result = self._call_with_retry(messages, config=self.config)

            if self.config.use_cache:
                # message_hash is already calcuated
                self.cache[message_hash] = self.result
                with open(self.config.cache_path, "wb") as f:
                    pickle.dump(self.cache, f)

        end_time = time.time()

        # Parse the result
        self.result_content = self.result["choices"][0]["message"]["content"]
        self.finished_reason = self.result["choices"][0]["finish_reason"]
        self.token_usages = self.result["usage"]

        logging.info(
            f"Prompt: {self.prompt}\n\n"
            + f"Result: {self.result}\n\n"
            + f"Time taken: {end_time - start_time} sec."
        )
        return self.result_message

    def _call_with_retry(self, messages, config):
        """
        Error handling and automatic retry.
        See details:
        https://help.openai.com/en/articles/6897213-openai-library-error-types-guidance
        https://github.com/openai/openai-python/blob/main/openai/api_resources/chat_completion.py

        Args:
            messages (list): A list of messages to be sent to the API.
            config (FilmConfig): A FilmConfig object.
        Returns:
            The result of the API call.
        """

        for i in range(self.config.max_retries):
            try:
                return openai.ChatCompletion.create(
                    messages=messages, **config.to_dict()
                )
            except (
                openai.error.RateLimitError,
                openai.error.Timeout,
                openai.error.APIError,
                openai.error.APIConnectionError,
            ) as err:
                wait_time = self.wait_time[min(i, len(self.wait_time) - 1)]
                logging.warning(
                    f"API error: {err},"
                    + f"wait for {wait_time}s and retry ({i + 1}/{self.config.max_retries})"
                )
                time.sleep(wait_time)
            except Exception as err:
                logging.error(f"Error: {err}")
                raise
        raise Exception("Max retries exceeded.")

    def _placeholder(self, prompt, placeholders):
        """
        Replace placeholders in the prompt with the given values.
        Generate errors if the placeholders are not found.

        Args:
            prompt (str): The prompt to be sent to the API.
            placeholders (dict): A dictionary of placeholders and their values.
        Returns:
            The prompt with the placeholders replaced with their values.
        """
        for key, value in placeholders.items():
            # re.escape is used to escape special characters in 'key'
            pattern = f"{{{{{re.escape(key)}}}}}"

            if not re.search(pattern, prompt):
                raise ValueError(f"Placeholder '{key}' not found in the prompt.")

            prompt = re.sub(pattern, value, prompt)
        return prompt

    def placeholders(self):
        """
        Return a list of placeholders in the prompt.
        Placeholders are enclosed in double curly brackets, e.g. {{placeholder}}.

        Return:
            A list of placeholders in the prompt.
        """
        return re.findall(r"\{\{(.+?)\}\}", self.prompt)

    def _messages(self, prompt, history, system_prompt):
        """
        Generate messages to be sent to the API.
        Args:
            prompt (str): The prompt to be sent to the API.
            history (list): A list of messages to be sent to the API before the prompt.
                            It is expected that the user's input and the system's response will alternate.
            system_prompt (str): The system prompt to be sent to the API. If None, default prompt is used.
        Returns:
            A list of messages to be sent to the API.

        Example:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        """
        messages = []
        # add system message
        messages.append({"role": "system", "content": system_prompt})

        # add history
        for idx, item in enumerate(history):
            if idx % 2 == 0:
                messages.append({"role": "user", "content": item})
            else:
                messages.append({"role": "assistant", "content": item})

        # add prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def num_tokens(self, placeholders={}):
        """Returns the number of tokens used by a list of messages.
        Args:
            placeholders (dict): A dictionary of placeholders and their values.
        Returns:
            The number of tokens used by the messages.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        prompt = self._placeholder(self.prompt, placeholders)
        messages = self._messages(prompt, self.history, self.system_prompt)

        num_tokens = 0
        for message in messages:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def max_tokens(self):
        """
        Returns the maximum number of tokens allowed by the API.
        See details:
        https://platform.openai.com/docs/models/gpt-4
        """
        if self.config.model.startswith("gpt-4-32k"):
            return 32768
        elif self.config.model.startswith("gpt-4"):
            return 8192
        elif self.config.model.startswith("gpt-3.5-turbo-16k"):
            return 16384
        elif self.config.model.startswith("gpt-3.5-turbo"):
            return 4096
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

    def get_history(self):
        """
        combine the history, prompts, and results into a single list so that it can be used for the input for the next call.
        Only callable after run() has been called.

        Returns:
            A list of messages to be sent to the API.
        """

        if self.result_message is None:
            raise Exception("Please call run() first.")

        history = self.history[:]
        history.append(self.prompt)
        history.append(self.result_message)
        return history

    def summary(self, save_path=None):
        """
        Generate a summary of the conversation.

        Returns:
            str : A summary of the conversation.
        """

        return_string = ""

        if self.result_messages:
            return_string += "\n".join(
                [f"{x['role']}: {x['content']}" for x in self.result_messages]
            )
            return_string += "\n----------\n"

        if self.result_content:
            return_string += "assistant: " + self.result_content

        if save_path is not None:
            # if save_path contains a directory, create it if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir != "" and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # create file
            with open(save_path, "w") as f:
                f.write(return_string)
        return return_string
