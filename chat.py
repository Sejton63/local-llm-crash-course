from ctransformers import AutoModelForCausalLM
import chainlit as cl

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assistant that follows instruction extremely well. Help as much as you can. Give short answers."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(f"Prompt created: {prompt}")
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    response = f"Hello, you just sent: {message.content}!"
    await cl.Message(response).send()

"""
history = []
question = "The capital of India is"
prompt = get_prompt(question)
answer = ""
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
    answer += word
print()
history.append(answer)

question = "And of the USA"
prompt = get_prompt(question, history)
for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
"""
