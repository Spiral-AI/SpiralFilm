# このクラスは、embeddingを行うための薄いラッパーを提供します

import time
import logging
import openai
import tiktoken
import re
import logging
import os
import pickle
import hashlib
import asyncio
from threading import Lock
from .config import FilmEmbedConfig


class FilmEmbed:
    def __init__(
        self,
        config=FilmEmbedConfig(),
    ):
        """
        Args:
            config (FilmConfig): A FilmConfig object. Only the following attributes are used:
                - model
                - use_cache
        """

        assert (
            config.model == "text-embedding-ada-002"
        ), "Only text-embedding-ada-002 is supported."
        self.config = config
        self.wait_time = [1, 3, 5, 10]  # time to wait before retrying

        self.results = None
        self.cache_lock = Lock()  # Create a lock for cache

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

    def run(self, texts):
        """Call OpenAI Embedding API to embed the text.
        Args:
            texts (str or list(str)): The text to embed. Accetps a string or a list of strings.
        Returns:
            list(float) or list(list(float))): The embedded text. Returns a list of floats or a list of list of floats depending on the input type.

        Usage:
            vec = FilmEmbed().run(["Today is a good day."])
            print(vec[0])

        """

        start_time = time.time()

        single_item = False
        if isinstance(texts, str):
            messages = [texts]
            single_item = True
        else:
            messages = texts  # rename

        # check if the message is stored in cache
        cached_results = []  # Store None if cache is not hit
        hash_dict = {}
        for message in messages:
            # check Cache
            result = None
            if self.config.use_cache:
                with self.cache_lock:  # Acquire lock when accessing cache
                    message_hash = hashlib.md5(
                        (
                            str(messages) + str(self.config.to_dict(mode="Caching"))
                        ).encode()
                    ).hexdigest()
                    hash_dict[message] = message_hash
                    if message_hash in self.cache:
                        logging.info(f"Cache hit.: {message}")
                        result = self.cache[message_hash]
            cached_results.append(result)

        # extract messages that are not stored in cache
        messages_to_call = [
            message
            for message, result in zip(messages, cached_results)
            if result is None
        ]
        if len(messages_to_call) == 0:
            api_results = {"data": []}
        else:
            api_results = self._call_with_retry(messages_to_call, config=self.config)

        # Parse the result
        self.results = []

        for message, cached_vec in zip(messages, cached_results):
            if cached_vec is not None:
                self.results.append(cached_vec)
            else:
                api_vec = api_results["data"].pop(0)["embedding"]
                self.results.append(api_vec)
                if self.config.use_cache:
                    message_hash = hash_dict[message]
                    self.cache[message_hash] = api_vec

        # update cache
        if self.config.use_cache:
            with self.cache_lock:  # Acquire lock when updating cache
                with open(self.config.cache_path, "wb") as f:
                    pickle.dump(self.cache, f)

        end_time = time.time()

        logging.info(
            f"Messages: {messages}\n\n"
            + f"Results: {self.results}\n\n"
            + f"Time taken: {end_time - start_time} sec."
        )

        if single_item:
            return self.results[0]
        else:
            return self.results

    async def run_async(self, texts):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.run, texts)
        return result

    def _call_with_retry(self, messages, config):
        """
        Error handling and automatic retry.
        See details:
        https://platform.openai.com/docs/api-reference/embeddings/create

        Args:
            messages (list): A list of messages to be sent to the API.
            config (FilmConfig): A FilmConfig object.
        Returns:
            The result of the API call.
        """

        for i in range(self.config.max_retries):
            try:
                return openai.Embedding.create(input=messages, **config.to_dict())
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

    def num_tokens(self, texts):
        """Returns the number of tokens.
        Args:
            texts (str or list(str)): The text to embed. Accetps a string or a list of strings.
        Returns:
            The number of tokens. If texts is a list, returns a list of integers. If texts is a string, returns an integer.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        single_item = False
        if isinstance(texts, str):
            texts = [texts]
            single_item = True

        num_tokens = [len(encoding.encode(text)) for text in texts]

        if single_item:
            return num_tokens[0]
        else:
            return num_tokens

    def max_tokens(self):
        """
        Returns the maximum number of tokens allowed by the API.
        See details:
        https://platform.openai.com/docs/models/gpt-4
        """
        if self.config.model == "text-embedding-ada-002":
            return 8192
        else:
            raise ValueError(f"Unknown model: {self.config.model}")
