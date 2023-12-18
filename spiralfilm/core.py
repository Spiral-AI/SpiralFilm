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
import tqdm
import re
import os
import pickle
import hashlib
import asyncio
from threading import Lock
import threading
import queue
from .config import FilmConfig
from .errors import *

import logging

logger = logging.getLogger(__name__)


class FilmCore:
    def __init__(
        self,
        prompt,
        history=[],
        system_prompt=None,
        config=FilmConfig(),
        override_params={},
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

        self.result = None
        self.result_messages = None
        self.result_content = ""
        self.finish_reason = None
        self.token_usages = None
        self.cache_lock = Lock()  # Create a lock for cache
        self.override_params = override_params

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

    @staticmethod
    def create_from(
        existing_instance,
        prompt,
        system_prompt=None,
        config=None,
    ):
        """
        既存のFilmCoreインスタンスの履歴を使用して、新しいFilmCoreインスタンスを作成します。

        Parameters:
        - existing_instance: 既存のFilmCoreインスタンス
        - new_prompt: 新しいインスタンスのプロンプト

        Returns:
        新しいFilmCoreインスタンス
        """

        # 新しいインスタンスを作成し、履歴を引き継ぎます
        new_instance = FilmCore(
            prompt=prompt,
            history=existing_instance.get_history(),
            system_prompt=system_prompt
            if system_prompt is not None
            else existing_instance.system_prompt,
            config=existing_instance.config if config is None else config,
        )

        return new_instance

    async def stream_async(self, placeholders={}):
        """
        asyncかつ、streaming形式でAPIを呼び出す。
        これが最も中核となる関数であり、run()などはこれをラップして実装されている。
        Args:
            messages (list): A list of messages to be sent to the API.
            config (FilmConfig): A FilmConfig object.
        Returns:
            The result of the API call.
        """
        # 必要なデータの準備
        prompt = self._placeholder(self.prompt, placeholders)
        messages = self._messages(prompt, self.history, self.system_prompt)

        # 結果を格納する変数の初期化
        self.result_prompt = prompt
        self.result_messages = messages

        # キャッシュの確認
        t0 = time.time()
        if self.config.use_cache:
            cache_result, message_hash = await self.read_cache(
                messages, config=self.config, override_params=self.override_params
            )
        else:
            cache_result, message_hash = None, None
        t1 = time.time()
        if (t1 - t0) > 1.0:
            logger.warning(
                f"cache checking time = {t1-t0} sec is too long."
                + f"Consider deleting cache file ({self.config.cache_path})"
            )

        # tokenだけ、仮にretryされた場合には累積で計算する (そうでないと、コストが間違う)
        self.token_usages = {
            "prompt_tokens": self.num_tokens(messages),
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # OpenAIの呼び出し (リトライ付き)
        for _ in range(self.config.max_retries):
            self.result = []  # すべてのAPI呼び出しの結果を保存する
            self.result_content = ""
            self.finish_reason = None
            try:
                # キャッシュなしの呼び出しパターン
                if cache_result is None:
                    apikey, time_to_wait = self.config.get_apikey()
                    # TODO: Azureとの連携機能を追加
                    # TODO: Retryについて設定を追加
                    client = openai.AsyncOpenAI(api_key=apikey["api_key"])

                    if time_to_wait > 0:
                        logger.warning(f"Waiting for {time_to_wait}s...")
                        time.sleep(time_to_wait)

                    chunk_gen = await client.chat.completions.create(
                        messages=messages,
                        **self.config.to_dict(override_params=self.override_params),
                        stream=True,
                    )
                # キャッシュありの呼び出しパターン
                else:
                    chunk_gen = cache_result

                # generatorをループで回す
                async for chunk in chunk_gen:
                    self.token_usages["completion_tokens"] += 1  # インクリメント
                    delta = chunk.choices[0].delta.content
                    # 最新の値を設定する
                    self.finish_reason = chunk.choices[0].finish_reason
                    # すべての出力を保存する
                    self.result.append(chunk)

                    # content filterの場合はStream途中でもすぐにエラーを投げる
                    if self.finish_reason == "content_filter":
                        self.config.update_apikey(apikey, status="success")
                        raise ContentFilterError(
                            f"Response has a finish reason of 'content_filter'\n{messages}"
                        )

                    self.result_content += delta if delta is not None else ""
                    # Stop wordsのチェック。存在したらループを抜ける
                    is_stop = False
                    if self.config.stop is not None:
                        for stopword in self.config.stop:
                            if self.result_content.startswith(stopword):
                                is_stop = True
                                break
                    if is_stop:
                        break
                    # 文字をyieldする。
                    if delta is not None:
                        yield delta
                else:
                    self.config.update_apikey(apikey, status="success")
                    break
            except (
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ) as err:
                self.config.update_apikey(apikey, status="failure")
                logger.warning(f"Retryable Error: {err}")
            except ContentFilterError as cfe:
                logger.error(f"Error due to content filter: {cfe}")
                raise
            except Exception as err:
                logger.error(f"Error: {err}")
                raise
        else:
            raise MaxRetriesExceededError("Max retries exceeded.")

        # キャッシュの保存
        if self.config.use_cache and cache_result is None and self.result:
            self.save_cache(message_hash, self.result)

        # トークン数の計算
        self.token_usages["total_tokens"] = (
            self.token_usages["prompt_tokens"] + self.token_usages["completion_tokens"]
        )

    async def run_async(self, placeholders={}):
        """
        stream_async()の非stream版。
        """
        result = ""
        async for t in self.stream_async(placeholders):
            result += t
        return result

    def stream(self, placeholders={}):
        """
        stream_async()の同期版。
        """
        q = queue.Queue()

        def producer():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def produce():
                async for chunk in self.stream_async(placeholders):
                    q.put(chunk)
                q.put(None)  # ストリームの終了を示す

            loop.run_until_complete(produce())

        threading.Thread(target=producer).start()

        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield chunk

    def run(self, placeholders={}):
        """
        run_async()の同期版。
        """
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.run_async(placeholders))
        return result

    def _set_api(self, apikey):
        if self.config.api_type == "openai":
            openai.api_type = "open_ai"
            openai.api_key = apikey["api_key"]
            openai.api_base = "https://api.openai.com/v1"
            openai.api_version = None
        elif self.config.api_type == "azure":
            openai.api_type = "azure"
            openai.api_key = apikey["api_key"]
            openai.api_base = apikey["api_base"]
            openai.api_version = self.config.azure_api_version
        return

    def read_cache(self, messages, config, override_params):
        with self.cache_lock:  # Acquire lock when accessing cache
            message_hash = hashlib.md5(
                (
                    str(messages)
                    + str(
                        self.config.to_dict(
                            mode="Caching",
                            override_params=self.override_params,
                        )
                    )
                ).encode()
            ).hexdigest()
            if message_hash in self.cache:
                logger.info("Cache hit.")
                raw_result = self.cache[message_hash]
                return raw_result, message_hash
            else:
                return None, None

    def save_cache(self, message_hash, api_result):
        with self.cache_lock:
            self.cache[message_hash] = api_result
            with open(self.config.cache_path, "wb") as f:
                pickle.dump(self.cache, f)

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
        # check if all placeholders are found
        expected_placeholders = self.placeholders()
        if not set(expected_placeholders) == set(placeholders.keys()):
            raise ValueError(
                f"Expected placeholders: {expected_placeholders}\n"
                + f"Given placeholders: {placeholders.keys()}"
            )
        # replace placeholders
        for key, value in placeholders.items():
            # re.escape is used to escape special characters in 'key'
            pattern = f"{{{{{re.escape(key)}}}}}"

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

    def num_tokens(self, messages):
        """Returns the number of tokens used by a list of messages.
        Args:
            messages (list): A list of messages to be sent to the API.
        Returns:
            The number of tokens used by the messages.
        """
        try:
            encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
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

        if self.result_messages is None:
            raise Exception("Please call run() first.")

        return [x["content"] for x in self.result_messages]

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
