import asyncio
from spiralfilm import FilmCore


async def do_something_else():
    for i in range(5):
        print(f"Do something else {i}", end="; ", flush=True)
        await asyncio.sleep(1)  # wait for 1 second
    print(end="\n", flush=True)


async def main():
    _prompt = "Talk freely about movie."
    stream = FilmCore(_prompt)

    task1 = asyncio.create_task(stream.run_async())
    task2 = asyncio.create_task(do_something_else())

    results = await asyncio.gather(task1, task2)

    # print the result from run_async
    print(f"Result from run_async: {results[0]}")


if __name__ == "__main__":
    asyncio.run(main())
