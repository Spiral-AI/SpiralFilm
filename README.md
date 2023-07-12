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
- And more! 🎉

## Installation 🔧

For the everyday users who simply can't wait to dive into the SpiralFilm action, here's how you can get the latest version all shiny and ready:

```
# For the pip wizards 🧙‍♀️
pip install git+https@github.com:Spiral-AI/SpiralFilm.git@main

# For the poetry aficionados 🖋️
poetry add git+https://github.com/Spiral-AI/SpiralFilm.git@main
```

For our dear developers, once you've cloned from git, jump into the folder and give this command a spin. Now you can see your modifications to SpiralFilm take effect in real-time in your other code! 

```
pip install -e .
```
Magic! 🎩✨

## Tutorial 📚

Now that you've got SpiralFilm installed, let's see it in action! Here are a couple of simple examples to get you started:

### Example 1: The Simple Scenario 🏄‍♀️
For this, we'll use the script in `./examples/simple_example.py`

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

# Don't forget to set your OpenAI API key as an environment variable!
# Uncomment and set your key
# import os
# os.environ["OPENAI_API_KEY"] = "your key here"

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

And that's it, folks! You're now ready to start making your own epic conversational masterpieces with SpiralFilm! 🎬🍿 Happy coding! 💻🚀
