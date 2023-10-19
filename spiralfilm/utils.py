def TextCutter(
    text, max_chars=None, strategy="chars:start", max_lines=None, max_chars_in_line=None
):
    """Limit the number of characters and lines in a string.

    Args:
        text (str): The text to limit.
        max_chars (int): The maximum number of characters in the text.
        strategy (str): Method to limit the text.
            - "chars:end": Cut out the text without considering the line breaks. Truncate from the start and keep the end.
            - "chars:start": Cut out the text without considering the line breaks. Truncate from the end and keep the start.
            - "lines:end": Cut out the whole lines to ensure the number of chars. Truncate from the start and keep the end.
            - "lines:start": Cut out the whole lines to ensure the number of chars. Truncate from the end and keep the start.
        max_lines (int): The maximum number of lines in the text.
        max_chars_in_line (int): The maximum number of characters in a line.

    Returns:
        str: The limited text.

    """
    assert strategy in ["chars:end", "chars:start", "lines:end", "lines:start"]

    if max_lines is not None:
        text = "\n".join(text.split("\n")[:max_lines])

    if max_chars_in_line is not None:
        lines = text.split("\n")
        lines = [line[:max_chars_in_line] for line in lines]
        text = "\n".join(lines)

    if max_chars is not None:
        if strategy == "chars:end":
            text = text[-max_chars:]
        elif strategy == "chars:start":
            text = text[:max_chars]
        elif strategy == "lines:end":
            lines = text.split("\n")
            while len("\n".join(lines)) > max_chars:
                lines = lines[1:]
            text = "\n".join(lines)
        elif strategy == "lines:start":
            lines = text.split("\n")
            while len("\n".join(lines)) > max_chars:
                lines = lines[:-1]
            text = "\n".join(lines)

    return text
