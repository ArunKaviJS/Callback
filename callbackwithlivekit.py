#!/usr/bin/env python3
# Scripts/intent.py

import os
import sys
import json
import logging
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

from pydantic import BaseModel, ValidationError
from typing import Literal, Optional, AsyncIterable, AsyncGenerator
from livekit.agents import UserInputTranscribedEvent, ConversationItemAddedEvent
from livekit.agents.llm import ImageContent, AudioContent

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

# === Load Section Prompts from JSON ===
PROMPT_JSON = THIS_DIR / "data" / "RAG_Prompts.json"
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

# === LLM and Embedding Configuration ===
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
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("LLM_API_VERSION"),
)

# === Index Creation or Loading ===
PERSIST_DIR = THIS_DIR / "query-engine-storage"
if not PERSIST_DIR.exists():
    documents = SimpleDirectoryReader(THIS_DIR / "data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

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
    history_stack.add(query, str(res))
    return str(res)

PROMPT_INSTRUCTIONS = """
You are a helpful AI assistant. Ask the user to speak their 10-digit phone number one digit at a time.
Repeat the number back clearly after all digits are received.
Use the STT text to extract each digit.
Make sure to verify it is 10 digits long before confirming.
""".strip()

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
            tools=[query_docuement],
        )

    async def on_enter(self):
        self.session.on("user_input_transcribed", self.handle_user_input_transcribed)
        self.session.on("conversation_item_added", self.handle_conversation_item_added)
        self.session.generate_reply()

    async def transcription_node(
        self, text: AsyncIterable[str]
    ) -> AsyncGenerator[str, None]:
        async for chunk in text:
            logger.info(f"[STT] Transcript: {chunk}")
            print(f"🎤 Transcript: {chunk}")
            yield chunk

    def handle_user_input_transcribed(self, event: UserInputTranscribedEvent):
        print(f"[Live STT] User input transcribed: {event.transcript}, Final: {event.is_final}, Speaker: {event.speaker_id}")

    def handle_conversation_item_added(self, event: ConversationItemAddedEvent):
        print(f"[Conversation Log] From {event.item.role}: {event.item.text_content}, Interrupted: {event.item.interrupted}")
        for content in event.item.content:
            if isinstance(content, str):
                print(f"  - Text: {content}")
            elif isinstance(content, ImageContent):
                print(f"  - Image: {content.image}")
            elif isinstance(content, AudioContent):
                print(f"  - Audio: {content.frame}, Transcript: {content.transcript}")

# === Entry Point ===
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    call_time = datetime.now(india).strftime("%Y-%m-%d %H:%M:%S")
    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        phone_number = participant.attributes.get('sip.phoneNumber', 'unknown')
    else:
        phone_number = "test-user"

    session = AgentSession()
    await session.start(agent=RagAgent(), room=ctx.room)
    print(f"👥 Session started for {phone_number} at {call_time}")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="devserver",
            entrypoint_fnc=entrypoint
        )
    )
