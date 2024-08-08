import speech_recognition as sr
import os
from dotenv import load_dotenv
import pyttsx3
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain import LLMChain

load_dotenv()

mic_device_index = 0

# Initialise the Large Language Model
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=1,
    model_name='gpt-3.5-turbo'
)

# Create a prompt template
template = """
You are front desk chatbot at a science day event in the school.
Most of the students are from grade 6 to grade 10. You are there to engage with the students and answer their questions.
Don't give long responses and always feel free to ask interesting questions that keeps someone engaged.
You should also be a bit entertaining and not boring to talk to. Use informal language.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""

# Create a prompt template
prompt = PromptTemplate.from_template(template)

# Create some memory for the agent
memory = ConversationBufferMemory(memory_key="chat_history")

# Initialise the conversation chain
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

engine = pyttsx3.init()

# Configure voice (optional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Set properties (optional)
engine.setProperty('rate', 180)
engine.setProperty('volume', 0.9)

recognizer = sr.Recognizer()


def listen():
    with sr.Microphone(device_index=mic_device_index) as source:
        print("Say something...")
        audio = recognizer.listen(source, phrase_time_limit=5, timeout=5)
    print("Voice captured")
    try:
        text = recognizer.recognize_google(audio)  # speech to text
        print(f"Captured text: {text}")
        return text
    except:
        print("Could not understand audio")


def prompt_model(text):
    # Prompt the LLM chain
    response = conversation_chain.run({"question": text})
    return response


def respond(model_response):
    # Run the speech synthesis
    engine.say(model_response)
    engine.runAndWait()


def conversation():
    user_input = ""
    while True:
        user_input = listen()
        if user_input is None:
            user_input = listen()
        elif "bye" in user_input.lower():
            respond(
                conversation_chain.run({
                    "question": "Send a friendly goodbye question and give a nice short sweet compliment based on the conversation."
                })
            )
            return
        else:
            model_response = prompt_model(user_input)
            respond(model_response)


def get_microphone():
    mic_list = sr.Microphone.list_microphone_names()
    for i, microphone_name in enumerate(mic_list):
        print(f"{i} -> {microphone_name}")
    print("Choose the microphone index:")
    try:
        mic_index = int(input())
        device = mic_list[mic_index]
        print(f"Selected microphone: {device}")
        return mic_index
    except ValueError as e:
        print("Invalid input")
        raise e


if __name__ == "__main__":
    mic_device_index = get_microphone()
    # respond(conversation_chain.run({"question": "Greet me in a friendly way"}))
    conversation()
