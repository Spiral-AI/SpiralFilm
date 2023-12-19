import json
import logging

logger = logging.getLogger(__name__)


# parserは、出力をパースするためのクラスです。
class FilmParser:
    def __init__(
        self,
        format,
        prefix="",
        sanity_check=None,
        line_prefix=["- "],
    ):
        """
        format: lines, markdown, json
        prefix: prefix to be added at the beginning
        sanity_check: function to check if the output is valid. Use 'assert' statement to check.
        max_retry: max number of retry. If sanity_check fails, retry until this number is reached. Set to 0 to disable retry.
        line_prefix: list of prefix to be recognized as a line. Used only when format is 'lines'.
        """
        assert format in [
            "lines",
            "markdown",
            "json",
        ], "format must be one of ['lines','markdown','json']"
        self.format = format
        self.prefix = prefix
        self.sanity_check = sanity_check
        if isinstance(line_prefix, str):
            self.list_prefix = [line_prefix]
        else:
            self.list_prefix = line_prefix

        self.set_retry_prompt(
            """
Please process the input text to ensure it strictly adheres to the {{format}} format.
- Identify and extract the essential content specific to the {{format}} format from the input.
- Remove any additional or extraneous lines before and after this core content.
- Ensure that the cleaned-up content is correctly formatted and can be successfully parsed as {{format}}.
- Do not alter the original texts in the contents.
                              
# input text
{{input_text}}

# formatted text
"""
        )

    def set_retry_prompt(self, prompt):
        self.retry_prompt = prompt

    def _try_parse(self, text):
        if self.format == "lines":
            result = self.parse_lines(text)
        elif self.format == "markdown":
            result = self.parse_markdown(text)
        elif self.format == "json":
            result = self.parse_json(text)

        if self.sanity_check is not None:
            self.sanity_check(result)  # raise error when sanity check fails

        return result

    def _refine(self, text, config):
        from .core import FilmCore  # Cannot import at the top due to circuler import

        retry_film = FilmCore(prompt=self.retry_prompt, config=config)
        placeholder = {}
        placeholder["input_text"] = text
        placeholder["format"] = self.format
        return retry_film.run(placeholder)

    def parse(self, text, config):
        try:
            result = self._try_parse(self.prefix + text)
        except Exception as err1:
            logger.info(f"Parse error. Retry: {err1}")
            retry_text = self._refine(self.prefix + text, config)
            try:
                result = self._try_parse(retry_text)
            except Exception as err2:
                logger.error(f"Retry failed: {err2}")
                raise err2
        return result

    def parse_lines(self, text):
        """
        Parse text as list of lines.
        Args:
            text: str
        Returns:
            list of str
        """
        lines = text.split("\n")
        lines = [line.strip() for line in lines]  # remove spaces
        if len(self.list_prefix) == 0:
            return lines
        new_lines = []
        for line in lines:
            for prefix in self.list_prefix:
                if line.startswith(prefix):
                    new_lines.append(line[len(prefix) :])
                    break
        lines = [line.strip() for line in new_lines]  # remove spaces
        return lines

    def parse_json(self, text):
        """
        Parse text as json.
        Args:
            text: str
        Returns:
            structured dict
        """
        return json.loads(text)

    def parse_markdown(self, text):
        """
        Parse text as markdown.
        Args:
            text: str
        Returns:
            structured dict
        """

        result = {}
        lines = text.split("\n")
        title = ""
        for line in lines:
            line = line.strip()
            for prefix in self.remove_prefix:
                if line.startswith(prefix):
                    line = line[len(prefix) :]
                    break

            if line == "":
                continue

            if line.startswith("# "):
                title = line[2:].strip()
                result[title] = ""
            else:
                result[title] += line + "\n"
        return result
