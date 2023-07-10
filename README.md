# SpiralFilm
## Introduction
SpiralFilm is a thin wrapper for the OpenAI GPT family of APIs. It packages the necessary features for users with a certain level of understanding of language models to create various applications as quickly as possible. 

The main design philosophies are:

- Unlike LlamaIndex, it does not encompass integration with VectorDB and the likes. We expect developers to have a certain level of development capability to call various databases.
- It does not provide a high level of abstraction like LangChain. Developers are able to easily modify prompts without delving deep into the code.
- It does not perform overly complex processing of prompts like guidance. With conversational APIs like gpt-3.5-turbo or gpt-4 in mind, it keeps processing to the bare minimum.

As a result, the provided features include:

- Automatic retry
- Placeholder functionality
- Token count verification
- Confirmation of sent prompts, time measurement features, and logging
- Generation of appropriate exceptions

Additionally, we propose templates for prompt modules that are easy to version control as an advanced feature.

## Installation
XXX

## TBD
