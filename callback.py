#!/usr/bin/env python3
# Scripts/intent.py

import os, sys, json, logging
from pathlib import Path
from datetime import datetime
import pytz

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit import rtc
from livekit.plugins import openai

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, ValidationError
from typing import Literal, Optional
import csv
import re
from langchain_core.messages import BaseMessage
import logging
logging.basicConfig(level=logging.DEBUG)


# === Timezone & Paths ===
india = pytz.timezone('Asia/Kolkata')
THIS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(THIS_DIR))
dotenv_path = THIS_DIR / "AcessTokens" / "env.ragu"
print(f"🔑 Loading environment from {dotenv_path}")
load_dotenv(dotenv_path)

# === Logging ===
logger = logging.getLogger("testing-agent")
logger.setLevel(logging.INFO)

CALLBACK_CSV_PATH = THIS_DIR / "conversation_logs.csv"

# === Load Section Prompts from JSON ===
PROMPT_JSON = THIS_DIR / "data" / "RAG_Prompts.json"
print("✅ Logging to:", CALLBACK_CSV_PATH.resolve())

with open(PROMPT_JSON, "r", encoding="utf-8") as f:
    prompt_map = json.load(f)
section_titles = list(prompt_map.keys())

# === Chat History Stack ===
class ConversationHistory:
    def __init__(self, max_history=5):
        self.stack = []
        self.max_history = max_history

    def add(self, user_query, agent_output):
        self.stack.append((user_query, agent_output))
        if len(self.stack) > self.max_history:
            self.stack = self.stack[-self.max_history:]

    def get_formatted_history(self):
        messages = []
        for q, a in self.stack:
            messages.append(ChatMessage(role="user", content=q))
            messages.append(ChatMessage(role="assistant", content=a))
        return messages

history_stack = ConversationHistory(max_history=5)


# === Intent Classification Schema ===
class IntentOutput(BaseModel):
    intent: Literal[
        "Admission-Related Intent",
        "Courses and Program Intent",
        "Eligibility & Admission Criteria",
        "Campus and Facilities Intent",
        "Career and Placement Intent",
        "Department and Faculty Intent",
        "Contact and Support Intent",
        "Institution Overview & Miscellaneous Intent",
    ]

async def classify_intent_llm(query: str) -> Optional[str]:
    prompt = (
        "You are an intent classification assistant for queries about Vivekanandha group of institutions.\n"
        "Given a user's question, select only one of the following intent categories that best matches the user's input:\n\n"
        + "\n".join(f"    {k}" for k in section_titles) +
        "\n\nReturn your answer as only the intent name (exactly as listed).\n"
        "Do not explain or output anything else."
    )
    user_message = f"User input: {query}\n\nOutput:"
    context_messages = history_stack.get_formatted_history()
    response = await Settings.llm.acomplete(
        messages=[
            ChatMessage(role="system", content=prompt),
            *context_messages,
            ChatMessage(role="user", content=user_message),
        ]
    )
    raw_intent = response.message.content.strip()
    try:
        intent_obj = IntentOutput(intent=raw_intent)
        return intent_obj.intent
    except ValidationError:
        print(f"⚠️ Invalid intent detected: {raw_intent}")
        return None

# === LLM-Based Section Guessing ===
async def guess_section_from_query_llm(query: str, section_titles: list[str]) -> str:
    system_prompt = (
        "You are an assistant for routing student queries to the correct section of a college handbook. "
        "Given a user's question and a list of possible section titles, return ONLY the best matching section title. "
        "If none clearly match, return an empty string."
    )
    user_prompt = (
        f"User question: {query}\n\n"
        f"Available sections:\n" + "\n".join(f"- {title}" for title in section_titles) +
        "\n\nReturn the most relevant section title from the list above. Do not explain."
    )
    context_messages = history_stack.get_formatted_history()
    response = await Settings.llm.acomplete(
        messages=[
            ChatMessage(role="system", content=system_prompt),
            *context_messages,
            ChatMessage(role="user", content=user_prompt),
        ]
    )
    return response.message.content.strip()

# === Azure Embedding & LLM ===
Settings.embed_model = AzureOpenAIEmbedding(
    model=os.getenv("EMBEDDING_MODEL"),
    deployment_name=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
    api_key=os.getenv("EMBEDDING_API_KEY"),
    azure_endpoint=os.getenv("EMBEDDING_AZURE_ENDPOINT"),
    api_version=os.getenv("EMBEDDING_API_VERSION"),
)
Settings.llm = AzureOpenAI(
    model=os.getenv("LLM_MODEL"),
    deployment_name=os.getenv("LLM_DEPLOYMENT_NAME"),
    api_key=os.getenv("LLM_API_KEY"),
    azure_endpoint=os.getenv("LLM_AZURE_ENDPOINT"),
    api_version=os.getenv("LLM_API_VERSION"),
)

# === Vector DB ===
PERSIST_DIR = THIS_DIR / "query-engine-storage"
if not PERSIST_DIR.exists():
    print("📚 Starting embeddings...")
    documents = SimpleDirectoryReader(THIS_DIR / "data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("✅ Embeddings completed.")
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# === Function Tool: Store STT Number to CSV ===
@llm.function_tool
def store_callback_number(spoken_number_text: str, participant_id: str) -> str:
    """
    Store spoken phone number text (verbal or digit) into CSV after extraction.
    """
    def verbal_to_digits(text: str) -> str:
        verbal_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        words = text.lower().split()
        digits = [verbal_map.get(word, word) for word in words]
        return ''.join(digits)

    try:
        number = verbal_to_digits(spoken_number_text)
        number = re.sub(r'[^0-9]', '', number)

        if not re.fullmatch(r"\d{10}", number):
            return "Invalid phone number. Please say a 10-digit number."

        timestamp = datetime.now(india).strftime("%Y-%m-%d %H:%M:%S")
        with open(CALLBACK_CSV_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["participant_id", "phone_number", "timestamp"])
            writer.writerow([participant_id, number, timestamp])

        logger.info(f"✅ Stored callback number: {number} for {participant_id}")
        return f"Phone number {number} stored. Thank you!"

    except Exception as e:
        logger.exception("Error storing phone number")
        return "Error occurred while saving the phone number."


@llm.function_tool

def store_conversation(user_input: BaseMessage, ai_output: BaseMessage, participant_id: str) -> str:
    """
    Store full user and AI messages (ChatMessage or plain string) into CSV with timestamp.
    """
    def extract_content(msg):
        return msg.content if hasattr(msg, "content") else str(msg)

    try:
        user_text = extract_content(user_input)
        ai_text = extract_content(ai_output)
        timestamp = datetime.now(india).strftime("%Y-%m-%d %H:%M:%S")

        with open(CALLBACK_CSV_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["participant_id", "user_input", "ai_output", "timestamp"])
            writer.writerow([participant_id, user_text, ai_text, timestamp])

        logger.info(f"🗣️ Logged conversation for {participant_id}: 🧑 {user_text} | 🤖 {ai_text}")
        print(f"👤 {participant_id} | User: {user_text} | AI: {ai_text}")
        return "Conversation stored successfully."
    except Exception as e:
        logger.exception("Error storing conversation")
        return "Failed to log conversation."
    
# === RAG Query Tool ===
@llm.function_tool
async def query_docuement(query: str, section: str = "") -> str:
    if not section:
        section = await classify_intent_llm(query)
        if not section:
            section = await guess_section_from_query_llm(query, section_titles)
        if not section:
            return "Sorry, the system could not classify your query."
    system_prompt = prompt_map.get(section.strip(), "")
    query_engine = index.as_query_engine(use_async=True, system_prompt=system_prompt)
    res = await query_engine.aquery(query)
    print(f"📘 Query in section [{section or 'default'}]:", res)
    history_stack.add(query, str(res))
    return str(res)

# === Agent instructions as a single constant ===
PROMPT_INSTRUCTIONS = """
🪪 name: Priya
🎙️ Voice Style
Friendly, confident, respectful tone.


Speak in Tamil-English code-mix: Tamil in Tamil script, English in English.


Avoid robotic/formal tone.


Native Tamil slang only — never use Malayalam slang.


Common words: வணக்கம், நன்றி, சரி, பேசு, நண்பர், பெற்றோர், பயணம்.



🌐 Language Selection (At Start of Call)
After greeting, always ask:
"Shall we continue in English or Tamil? / தமிழா பேசலாமா, இல்லா English-ல பேசலாமா?"
If user says “Tamil” or speaks Tamil → speak 100% Tamil. No English.


If user says “English” → speak only English. No Tamil.


If user mixes → use Tamil-English code-mix style.


⚠️ Strict Tamil pronunciation — avoid Malayalam-influenced speech.



👋 Your Role
Greet warmly:


"வணக்கம்! நான் Priya, Vivekanandha College of Engineering for Women, Tiruchengode-ல இருந்து பேசுறேன்."
Guide helpfully on: admissions, eligibility, courses, fees, hostel, placement, transport, departments.


Never hallucinate. Use only document content via:


query_document(query, section)


If no section → classify or guess using LLM.


If question is irrelevant:


"Sorry, that information is not available."
 "மன்னிச்சுக்கோங்க, அந்த தகவல் நம்மிடம் கிடைக்கல."

✅ Help Topics
Admissions, Eligibility, Courses, Fees, Hostel, Placement, Transport, Departments


🔁 Same Question → Same Answer
Repeat correct answers if asked again.
🚫 If Course Unavailable
"அந்த course நம்ம கல்லூரில இல்ல, நீங்க. Engineering departments மட்டும் இருக்கு."

⏳ Delay Handling
If delay in response:
10s → "ஒரு நிமிஷம், சொல்றேன்"


20s → Repeat


30s → "தகவல் இப்போ கிடைக்கல, எங்க website-ல பாருங்க, இல்லைனா கொஞ்ச நேரம் கழிச்சு call பண்ணுங்க."



📞 Callback Number Handling

If user says:

    English: “that’s all”, “nothing else”, “no more questions”

    Tamil: “இல்லை பாஸ்”, “அவ்ளோதான்”, “வேற எதுவும் வேணாம்”, “முடிச்சாச்சு”

→ Ask:

    "Shall we call back on this number, or do you want us to use a different number?"
    "நீங்க பேசற இந்த number-க்கு தான் நாங்க திரும்ப call பண்ணலாமா? இல்லா வேற எந்த number use பண்ணணும்?"

If user gives another number:

    Say:

        "சரி, சொல்லுங்க." / "Sure, please tell me."

    Wait silently for 10 seconds.

    Convert spoken number (Tamil-English mix) to text using voice-to-text.

    Do not repeat it aloud.

    Confirm with user:

        "நீங்க சொன்ன number சரியா?"
        "Is this the correct number?"

If confirmed (user says "சரி", "ஆமாம்", "yes", "correct"):

✅ Store number and print it:

print(f"📞 Caller call back phone number: {number}")

If user says it's wrong ("இல்லை", "தவறு", "wrong"):

❌ Go back to step 1. Repeat until confirmed.

End Politely

    "நன்றி, உங்களுக்கு நல்ல நாளா இருக்கட்டும்!"

    "Thanks for calling. All the best!"




""".strip()
PROMPT_INSTRUCTIONS += """

📞 Callback Number Handling (English Version)
If user says:
    "that's all", "nothing else", "no more questions"
→ Ask:
    "Would you like us to call you back on this number or a different one?"

If user says: "different number" → say:
    "Okay, please tell me the number."
    Wait 10 seconds, listen to number.

Call this tool:
[TOOL: store_callback_number, {"spoken_number_text": "<spoken number>", "participant_id": "<id>"}]

Confirm:
    "Is this the correct number?"
If user confirms:
    ✅ Store.
Else:
    ❌ Ask again.

End politely:
    "Thank you! Have a nice day!"
"""
# Add tool to RagAgent
class RagAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=PROMPT_INSTRUCTIONS,
            llm=openai.realtime.RealtimeModel.with_azure(
                azure_deployment=os.getenv("VOICE_LLM_DEPLOYMENT"),
                azure_endpoint=os.getenv("VOICE_LLM_ENDPOINT"),
                api_key=os.getenv("VOICE_LLM_API_KEY"),
                api_version=os.getenv("VOICE_LLM_API_VERSION"),
            ),
            tools=[query_docuement, store_callback_number, store_conversation],
        )

    async def on_enter(self):
        self.session.generate_reply()


    
        
    

# === Entrypoint ===
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    

    #print(PROMPT_INSTRUCTIONS)

    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        phone_number = participant.attributes.get('sip.phoneNumber', 'unknown')
        print(f"📞 Caller phone number: {phone_number}")
        call_time = datetime.now(india).strftime("%Y-%m-%d %H:%M:%S")
    else:
        phone_number = "test-user"

    session = AgentSession()
    await session.start(agent=RagAgent(), room=ctx.room)

    print(f"👥 Session started for {phone_number} at {call_time}")    

# === Main Launcher ===
if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage, AIMessage

    mode = "test"  # change to "prod" to run the actual agent

    if mode == "prod":
        cli.run_app(
            WorkerOptions(
                agent_name="devserver",
                entrypoint_fnc=entrypoint
            )
        )
    elif mode == "test":
        async def test_logging():
            result = await store_conversation.arun(
                user_input=HumanMessage(content="What are your services?"),
                ai_output=AIMessage(content="We offer AI-based voice assistants."),
                participant_id="test_user"
            )
            print("✅ Logging result:", result)

        asyncio.run(test_logging())

