"""
A demo showing how to use FilmCore.stream_async() to stream the generated text.
"""

import asyncio
import curses
from spiralfilm import FilmCore

stdscr = curses.initscr()

current_col_positions = [0, 0]  # Maintain current column position for each line


# This looks like a lot of code, but it's just handling multi-line output, so you can basically ignore it.
async def print_stream(stream_gen, line):
    global current_col_positions
    async for result in stream_gen:
        if (
            current_col_positions[line] >= curses.COLS
        ):  # If current column position exceeds window width
            current_col_positions[line] = 0  # Reset to the start of the line
            stdscr.addstr(
                line, current_col_positions[line], "\n"
            )  # Move to the next line
        stdscr.addstr(line, current_col_positions[line], result)
        current_col_positions[line] += len(result)  # Update column position
        stdscr.refresh()


async def main():
    placeholders1 = {"name": "Alice", "topic": "movies"}
    placeholders2 = {"name": "Bob", "topic": "sports"}

    _prompt = "Hi {{name}}! Please talk whatever about {{topic}}."
    # Important: Since stream object holds class variables to store the outputs, you need to create a new stream object for each line.
    stream1 = FilmCore(_prompt)
    stream2 = FilmCore(_prompt)

    task1 = asyncio.create_task(
        print_stream(stream1.stream_async(placeholders1), 0)
    )  # line 0
    task2 = asyncio.create_task(
        print_stream(stream2.stream_async(placeholders2), 1)
    )  # line 1

    await asyncio.gather(task1, task2)

    print("task1 :", stream1.token_usages)
    print("task2 :", stream2.token_usages)


if __name__ == "__main__":
    try:
        asyncio.run(main())
        stdscr.getkey()  # Wait for user input
    finally:
        curses.endwin()
