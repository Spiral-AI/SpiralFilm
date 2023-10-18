def TextCutter(text, max_chars=None, max_lines=None, max_chars_in_line=None):
    """Limit the number of characters and lines in a string.

    Args:
        text (str): The text to limit.
        max_chars (int): The maximum number of characters in the text.
        max_lines (int): The maximum number of lines in the text.
        max_chars_in_line (int): The maximum number of characters in a line.

    Returns:
        str: The limited text.

    """

    if max_lines is not None:
        text = "\n".join(text.split("\n")[:max_lines])

    if max_chars_in_line is not None:
        lines = text.split("\n")
        lines = [line[:max_chars_in_line] for line in lines]
        text = "\n".join(lines)

    if max_chars is not None:
        text = text[:max_chars]

    return text
