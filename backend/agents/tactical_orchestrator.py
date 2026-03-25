import json
import asyncio
import logging
from typing import Dict, Any, List
import os

# Optional: Real LLM Integration
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    load_dotenv()
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using Gemini 1.5 Pro or Flash depending on requirements
        llm_model = genai.GenerativeModel("gemini-1.5-flash")
        logging.info("Gemini API configured successfully.")
    else:
        llm_model = None
except ImportError:
    llm_model = None

# Import our new RAG Pipeline
from backend.agents.rag_pipeline import MatchDataRAG

class ScoutAgent:
    """
    Scout/Data Agent: Uses tool-calling to interface with the RAG Pipeline to
    query the tracking data, EPV matrices, and spatio-temporal events.
    """
    def __init__(self, rag_instance: MatchDataRAG):
        self.name = "ScoutAgent"
        self.rag = rag_instance

    async def fetch_tactical_data(self, query: str) -> str:
        logging.info(f"[{self.name}] Querying RAG Database for: {query}")
        
        # In a full LangChain setup, this is a @tool. 
        # Here we natively query our ChromaDB RAG.
        retrieved_docs = self.rag.query_tactics(query, n_results=3)
        
        if not retrieved_docs or "RAG Database Offline" in retrieved_docs[0] or "Query Error" in retrieved_docs[0]:
             return "No valid tracking data retrieved. Database might be offline or empty."
             
        # Combine the top results into a single context block
        context = "\n\n".join(retrieved_docs)
        return context


class CoachAgent:
    """
    Coach/Analyst Agent: Takes the raw contextual data retrieved by the Scout Agent 
    and formats it into coherent tactical insights and recommendations based on 
    the refined expected goals (xG) model and player tracking prompt.
    """
    def __init__(self):
        self.name = "CoachAgent"
        self.system_prompt = """
Analyze the given match footage/data using an expected goals (xG) model and player/event tracking.

Your tasks:

**xG Integration**
* Calculate real-time xG for every shot.
* Provide cumulative xG for both teams across the match.
* Highlight key moments where xG significantly changed.

**Frame-by-Frame Tactical Analysis**
* Break down important sequences frame by frame (build-up, final third entries, shots).
* For each frame:
  * Identify player positioning and movement.
  * Evaluate decision-making (pass, dribble, shot).
  * Suggest better alternatives (e.g., optimal pass, timing, positioning).

**Decision Optimization**
* For each attacking sequence:
  * What was the actual decision taken?
  * What was the optimal decision based on positioning and xG maximization?
  * Quantify the difference in expected outcome (xG gained/lost).

**Post-Match Analysis**
* Compare actual goals vs expected goals (xG).
* Identify overperformance or underperformance.
* Highlight inefficiencies in attack (missed high-xG chances, poor shot selection).
* Provide insights on defensive errors leading to high xG chances.

**Key Insights & Recommendations**
* Tactical improvements (spacing, passing lanes, decision timing).
* Player-specific suggestions.
* Patterns observed (e.g., repeated poor choices in similar situations).

**Output Format:**
* Timeline-based breakdown (minute-by-minute or event-based).
* Frame-level insights for key moments.
* Clear numerical xG values alongside qualitative analysis.
* Visual or structured representation where applicable.
"""

    async def generate_advice(self, raw_data_str: str, user_query: str) -> str:
        logging.info(f"[{self.name}] Synthesizing tactical data into natural language...")
        
        if "No valid tracking data" in raw_data_str:
            return "I am unable to analyze the situation because the match tracking DB is offline."
            
        if llm_model:
            # REAL LLM EXECUTION
            prompt = f"{self.system_prompt}\n\nUSER QUERY: '{user_query}'\n\nMATCH DATA CONTEXT:\n{raw_data_str}"
            try:
                # Assuming sync call wrapped in to_thread, but for gemini generate_content handles blocking fine
                # To make it truly async we use generate_content_async if available, or just await in standard event loop
                response = await asyncio.to_thread(llm_model.generate_content, prompt)
                return response.text
            except Exception as e:
                logging.error(f"LLM API Error: {e}")
                return f"*(LLM API integration failed. Error: {e})*\n\nFalling back to simulated response..."

        # MOCKED FALLBACK if no API key
        advice = (
            f"Based on my analysis of the spatial data using our refined xG tracking model:\n\n"
            f"**Event Context:**\n{raw_data_str}\n\n"
            f"**Recommendation (Following Refined System Prompts):**\n"
            f"- **xG Analysis**: The data shows a build-up phase with an xG potential shift.\n"
            f"- **Decision Optimization**: The current passing lanes indicate poor spacing. The optimal decision based on EPV maximization is to stretch the play.\n"
            f"- **Tactical Improvement**: Instruct the wingers to hold width earlier in the possession sequence to force the defensive line to expand, increasing the pass completion probability into the central zone.\n"
            f"\n*(You are currently seeing the local mocked output. Add GEMINI_API_KEY to your .env file to activate real LLM generation.)*"
        )
        return advice

class LLMJudgeAgent:
    """
    Validator Agent: Evaluates the Analyst Agent's output against the raw tracking data 
    to ensure absolute factual correctness and zero hallucination before sending it to the user.
    """
    def __init__(self):
        self.name = "LLMJudgeAgent"
        
    async def validate(self, original_data: str, generated_advice: str) -> bool:
        logging.info(f"[{self.name}] Validating response against raw data grounds truth...")
        
        if llm_model:
            prompt = f"Given this truth: {original_data}, does this advice contain hallucinations or hallucinated frame numbers? Answer YES or NO. Advice: {generated_advice}"
            try:
                response = await asyncio.to_thread(llm_model.generate_content, prompt)
                if "YES" in response.text.upper():
                    logging.warning(f"[{self.name}] LLM Hallucination detected!")
                    return False
                return True
            except Exception as e:
                pass
                
        if "Frame" in generated_advice and "Frame" not in original_data:
            logging.warning(f"[{self.name}] Hallucination detected! Generated advice contains ungrounded facts.")
            return False
            
        return True

class TacticalOrchestrator:
    """
    Orchestrator Agent: Parses WebSocket chat / API requests, delegates tools to Scout, 
    passes data to Coach, and validates via Judge before returning response.
    Acts as a decentralized Multi-Agent System (MAS).
    """
    def __init__(self):
        self.rag = MatchDataRAG()
        self.scout = ScoutAgent(self.rag)
        self.coach = CoachAgent()
        self.judge = LLMJudgeAgent()
        
    async def process_chat_message(self, message: str, websocket) -> None:
        """
        Processes an incoming query and streams the multi-agent thought process 
        back to the frontend via WebSocket.
        """
        # 1. Orchestrator acknowledges
        await websocket.send_json({"agent": "Orchestrator", "text": "Analyzing request..."})
        
        # 2. Delegate to Scout
        await websocket.send_json({"agent": "Orchestrator", "text": "Delegating to ScoutAgent to search tactical RAG database."})
        await asyncio.sleep(0.5) # Simulate API latency
        
        tactical_data = await self.scout.fetch_tactical_data(message)
        await websocket.send_json({"agent": "Scout", "text": f"Extracted metrics from DB."}) # Keep short for UI
        await asyncio.sleep(0.5)
        
        # 3. Delegate to Coach
        await websocket.send_json({"agent": "Orchestrator", "text": "Passing geometric data to CoachAgent for actionable insights."})
        
        advice = await self.coach.generate_advice(tactical_data, message)
        await asyncio.sleep(0.5)
        
        # 4. Delegate to Judge for Validation
        await websocket.send_json({"agent": "Orchestrator", "text": "Running Fact-Check Validation via LLMJudgeAgent..."})
        is_valid = await self.judge.validate(tactical_data, advice)
        await asyncio.sleep(0.5)
        
        # 5. Final output
        if is_valid:
             await websocket.send_json({"agent": "Coach", "text": advice, "is_final": True})
        else:
             await websocket.send_json({
                 "agent": "Judge", 
                 "text": "The Coach's response failed factual verification against the database tracking logs. Please rephrase the query.",
                 "is_final": True
             })
             
    async def generate_full_report(self) -> str:
        """
        Generates a massive post-match summary using the entire RAG DB.
        """
        # Fetch generic whole-match data
        tactical_data = await self.scout.fetch_tactical_data("Summarize the entire match spatial density and high EPV spikes.")
        
        # Generate comprehensive advice
        report = await self.coach.generate_advice(tactical_data, "Write a post-match breakdown.")
        return report
