```python
#!/usr/bin/env python3
# Scripts/conversation.py

import os
import sys
import json
import csv
import asyncio
from pathlib import Path
from datetime import datetime
import pytz
from dotenv import load_dotenv
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
import livekit
from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli, function_tool
from livekit import rtc
from livekit.plugins import openai

# === Prompt Instructions ===
PROMPT_INSTRUCTIONS = """
🪪 name: Priya
🎙️ Voice Style
Friendly, confident, respectful tone.

Speak in Tamil-English code-mix: Tamil in Tamil script, English in English.

Avoid robotic/formal tone.

Native Tamil slang only — never use Malayalam slang.

Common words: வணakkம், நன்றி, சரி, பேசு, நண்பர், பெற்றோர், பயணம்.

🌐 Language Selection (At Start of Call)
After greeting, always ask:
"Shall we continue in English or Tamil? / தமிழா பேசலாமா, இல்லா English-ல பேசலாமா?"
If user says “Tamil” or speaks Tamil → speak 100% Tamil. No English.

If user says “English” → speak only English. No Tamil.

If user mixes → use Tamil-English code-mix style.

⚠️ Strict Tamil pronunciation — avoid Malayalam-influenced speech.

👋 Your Role
Greet warmly:
"வணakkம்! நான் Priya, Vivekanandha College of Engineering for Women, Tiruchengode-ல இருந்து பேசுறேன்."
Guide helpfully on: admissions, eligibility, courses, fees, hostel, placement, transport, departments.

Never hallucinate. Use only document content via query processing.

If no relevant content → 
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
30s → "தகவல் இப்போ கிடைக்கல, எங்க website-ல பாருங்க, இllைனா கொஞ்ச நேரம் கழிச்சு call பண்ணுங்க."

📞 Callback Number Handling
If user says:
    English: “that’s all”, “nothing else”, “no more questions”
    Tamil: “இllை பாஸ்”, “அவ்ளோதான்”, “வேற எதuvும் வேணாம்”, “முடிச்சாச்சு”
→ Ask:
    "Shall we call back on this number, or do you want us to use a different number?"
    "நீங்க பேசற இந்த number-க்கு தான் நாங்க திரும்ப call பண்ணலாமா? இllா வேற எந்த number use பண்ணணும்?"

If user gives another number:
    Say:
        "சரி, சொllுங்க." / "Sure, please tell me."
    Wait silently for 10 seconds.
    Convert spoken number (Tamil-English mix) to text using voice-to-text (simulated here with input).
    Do not repeat it aloud.
    Confirm with user:
        "நீங்க சொn்ன number சரியா?"
        "Is this the correct number?"

If confirmed (user says "சரி", "ஆமாம்", "yes", "correct"):
✅ Store number and print it:
print(f"📞 Caller call back phone number: {number}")

If user says it's wrong ("இllை", "தவறு", "wrong"):
❌ Go back to step 1. Repeat until confirmed.

End Politely
    "நன்றி, உங்களுக்கு நall நாளா இருக்கட்டும்!"
    "Thanks for calling. All the best!"
""".strip()

# === Timezone & Paths ===
india = pytz.timezone('Asia/Kolkata')
THIS_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(THIS_DIR))
dotenv_path = THIS_DIR / "AcessTokens" / "env.ragu"
print(f"🔑 Loading environment from {dotenv_path}")
load_dotenv(dotenv_path)

# === Memory Agent ===
class MemoryAgent:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.conversation_history = {}  # {session_id: [(role, content, timestamp), ...]}
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['session_id', 'timestamp', 'role', 'content'])

    def add_entry(self, session_id, role, content):
        timestamp = datetime.now(india).strftime("%Y-%m-%d %H:%M:%S")
        entry = (role, content, timestamp)
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        self.conversation_history[session_id].append(entry)
        self._save_to_csv(session_id, role, content)

    def _save_to_csv(self, session_id, role, content):
        timestamp = datetime.now(india).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([session_id, timestamp, role, content])
        except Exception as e:
            print(f"Error saving to CSV: {e}")

# === Azure Embedding & LLM for RAG ===
try:
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
except Exception as e:
    print(f"Failed to initialize Azure embeddings/LLM: {e}")
    raise

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

# === Query Processing Tool ===
@function_tool
async def process_query(query: str, session_id: str) -> str:
    """Process a user query using a vector search and return a response based on document content."""
    try:
        # Embed query and perform similarity search
        query_engine = index.as_query_engine(use_async=True, system_prompt=PROMPT_INSTRUCTIONS)
        res = await query_engine.aquery(query)
        print(f"Query processed with vector search: {res}")
        
        return str(res)
    except Exception as e:
        print(f"Query processing failed: {e}")
        return "மன்னிச்சுக்கோங்க, அந்த தகவல் நம்மிடம் கிடைக்கல / Sorry, that information is not available."

# === Agent: Azure Realtime Voice LLM + RAG Tool ===
class RagAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(
            instructions=PROMPT_INSTRUCTIONS,
            llm=openai.realtime.RealtimeModel.with_azure(
                azure_deployment=os.getenv("VOICE_LLM_DEPLOYMENT"),
                azure_endpoint=os.getenv("VOICE_LLM_ENDPOINT"),
                api_key=os.getenv("VOICE_LLM_API_KEY"),
                api_version=os.getenv("VOICE_LLM_API_VERSION"),
            ),
            tools=[process_query],  # Use the decorated function directly
        )
        self.ctx = ctx
        self.memory_agent = MemoryAgent(csv_path=THIS_DIR / "data" / "conversation_history.csv")

    async def on_audio_frame(self, frame):
        # Simulate STT: Convert voice frame to text
        try:
            stt_text = await self.llm.transcribe(frame)
            if stt_text:
                print(f"STT converted: {stt_text}")
                session_id = self.ctx.session_id
                # Store the user query
                self.memory_agent.add_entry(session_id, "user", stt_text)
                # Process the query and get response
                response = await process_query(stt_text, session_id)
                # Store the assistant response
                self.memory_agent.add_entry(session_id, "assistant", response)
                # Simulate TTS: Convert text to voice
                voice_response = await self.llm.synthesize(response)
                await self.ctx.room.publish_audio(voice_response)
                print(f"TTS response sent: {response}")
        except Exception as e:
            print(f"Error in on_audio_frame: {e}")
            self.memory_agent.add_entry(self.ctx.session_id, "assistant", f"Error processing request: {str(e)}")

    async def on_enter(self):
        await self.session.generate_reply()

# === Entrypoint ===
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        phone_number = participant.attributes.get('sip.phoneNumber', 'unknown')
        print(f"📞 Caller phone number: {phone_number}")
        call_time = datetime.now(india).strftime("%Y-%m-%d %H:%M:%S")
    else:
        phone_number = "test-user"

    session = AgentSession()
    await session.start(agent=RagAgent(ctx), room=ctx.room)

    print(f"👥 Session started for {phone_number} at {call_time}")    

# === CLI App Launcher ===
if __name__ == "__main__":
    try:
        print("Starting assistant")
        cli.run_app(
            WorkerOptions(
                agent_name="devserver",
                entrypoint_fnc=entrypoint
            )
        )
    except Exception as e:
        print(f"Assistant failed: {e}")
        raise
```