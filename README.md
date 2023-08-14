# SpiralFilm 🌀🎥
## Introduction 🚀
SpiralFilm is your friendly neighborhood wrapper for the OpenAI ChatGPT family of APIs. It's designed for those knowledge-hungry language model enthusiasts who can't wait to whip up various applications at lightning speed. 🌩️

Here's how we roll:

- Unlike LlamaIndex 🦙, we're not into the whole integration-with-VectorDB-and-the-likes thing. We believe in you, developers, and trust in your abilities to call upon various databases as you please. 💪
- Forget about high-level abstraction like you'd see in LangChain 🔗. With us, you can tweak prompts without needing to dig into the dark depths of the code. 😎
- We're not into overcomplicating stuff. So, unlike guidance, we won't boggle you down with complex processing of prompts. We're more of a keep-it-simple-stupid kind of wrapper, especially when it comes to APIs like gpt-3.5-turbo 🚀 or gpt-4. 🤖

What we do bring to the table includes:

- Automatic retry (because sometimes, at first, you don't succeed) ↩️
- Placeholder functionality (like mad-libs, but for code) 🖍️
- Token count verification (keeping count, so you don't have to) 🔢
- Keeping you in the loop with confirmations of sent prompts, timing features, and logging 🕒
- Caching functionality (to speed up repeated requests and reduce server load) 🚀
- Async execution support, allowing you to run multiple tasks simultaneously, thereby making your application more efficient and responsive. 👾
- And more! 🎉

## Installation 🔧

For the vast majority of users, the easiest way to install SpiralFilm is simply by using pip:
```
pip install spiralfilm
```

However, for those adventurous spirits who want the very latest version, or perhaps the ones who love to walk on the bleeding edge, here's how you can get the freshest cut from our repository:

```
# For the pip wizards 🧙‍♀️
pip install git+https@github.com:Spiral-AI/SpiralFilm.git@main

# For the poetry aficionados 🖋️
poetry add git+https://github.com/Spiral-AI/SpiralFilm.git@main
```

For our dear developers, once you've cloned from git, jump into the folder and give this command a spin. Now you can see your modifications to SpiralFilm take effect in real-time in your other code! 

```
git clone git@github.com:Spiral-AI/SpiralFilm.git
cd SpiralFilm
pip install -e .
```
Magic! 🎩✨

## Tutorial 📚

Now that you've got SpiralFilm installed, let's see it in action! Here are a couple of simple examples to get you started:

### Example 1: The Simple Scenario 🏄‍♀️
For this, we'll use the script in `examples/simple_example.py`

```python
from spiral_film import FilmCore

# First things first, let's set up the environment variable for your OpenAI API key
# Uncomment and insert your key as shown below
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

# Now, let's create a filmcore instance
f = FilmCore(
    prompt="""
Talk as you want.
You're {{user_name}}.
"""
).run(placeholders={"user_name": "Tom"})  # Let's pretend we're Tom for this one

# Print it out and see what Tom has to say!
print(f)
```

### Example 2: The Configured Convo 🤖
Next up, we'll use a configuration to fine-tune our instance. You can find this script in `examples/config_example.py`

```python
from spiral_film import FilmCore, FilmConfig

# Let's set up our config
config = FilmConfig(model="gpt-4", temperature=0.5, max_tokens=100)

# And our conversation template
_template = """
Talk as you want.
You're {{user_name}}.
"""

# Now we'll create a filmcore instance with our config and template
f = FilmCore(
    prompt=_template,
    config=config,
).run(placeholders={"user_name": "Tom"})  # Tom is back for another round!

# Let's see what Tom has to say under this new configuration
print(f)
```

### Example 3: Recollections and Context Memory 🧠
There's immense power in context, and with `FilmCore`, you can harness this power seamlessly. This example, which you can find in `examples/conversation_example.py`, showcases how you can retain context and query it in subsequent interactions:


By using the create_from method, we can ensure a smooth continuation of the conversation. So, whether it's a fact, a story detail, or a crucial piece of data, FilmCore helps keep the narrative threads intact. 🧵📖
```python
from spiralfilm import FilmCore

fc1 = FilmCore(
    prompt="""
Remember that x={{num}}.
Hello!
"""
)
print(fc1.run(placeholders={"num": "1234"}))

fc2 = FilmCore.create_from(
    fc1,
    prompt="""
Do you remember x?
                         """,
)

print(fc2.run())

```


### Example 4: Deep Dive with Embeddings 🌊
If you're keen on exploring semantic relationships between sentences, the `FilmEmbed` utility is your new best friend. Dive into the embedding space and uncover hidden dimensions of meaning. Let's see it in action in the `examples/embed_example.py` script:
```python
from spiralfilm import FilmEmbed

examples = []

examples.append("Today is a super good day.")
examples.append("Today is a good day.")
examples.append("Today is a bad day.")

vecs = FilmEmbed().run(texts=examples)


def calc_similarity(v1, v2):
    return sum([v1[i] * v2[i] for i in range(len(v1))])


print(
    f"Similarity between '{examples[0]}' and '{examples[1]}' : ",
    calc_similarity(vecs[0], vecs[1]),
)
print(
    f"Similarity between '{examples[0]}' and '{examples[2]}' : ",
    calc_similarity(vecs[0], vecs[2]),
)

```

With this, you're equipped to explore semantic spaces and better understand the relationship between different sentences. What story do your embeddings tell? 🧐📊

And that's it, folks! You're now ready to start making your own epic conversational masterpieces with SpiralFilm! 🎬🍿 Happy coding! 💻🚀

But wait, there's more! Be sure to check out the "examples" folder for more usage scenarios and ideas. We've packed it full of tips, tricks, and goodies to get you up and running in no time. 📚🔍

## Contribution 🤝

If you feel like giving back, we always welcome contributions. But remember, at SpiralFilm, we're all about keeping it simple and transparent. We love that you're excited to add features, but let's keep it in line with our "thin-wrapper" philosophy. That way, everyone can continue to enjoy the beauty of simplicity! 💖🌐