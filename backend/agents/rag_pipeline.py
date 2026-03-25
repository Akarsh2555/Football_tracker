import os
import json
import logging
from typing import List, Dict, Any

try:
    import chromadb
except ImportError:
    chromadb = None
    logging.warning("chromadb not installed. RAG features will be simulated or disabled.")

class MatchDataRAG:
    """
    RAG Pipeline for Post-Match Tactical Analysis.
    Structures numerical tracking arrays (xT grids, EPV surfaces, coordinates) 
    into semantic text descriptions and embeds them into a Vector Database.
    """
    def __init__(self, persist_directory: str = "backend/chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "match_tactics_log"
        self.client = None
        self.collection = None
        
        if chromadb:
            # Initialize ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            try:
                # Use ephemeral client for prototype if persist fails, or PersistentClient
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                logging.info(f"RAG Pipeline initialized. ChromaDB collection '{self.collection_name}' ready.")
            except Exception as e:
                logging.error(f"Failed to initialize ChromaDB: {e}")
                self.client = None

    def serialize_frame_context(self, frame_idx: int, tracking_data: List[Dict], frame_intel: Dict) -> str:
        """
        Converts clinical numerical data for a specific frame into a semantic 
        paragraphs so the LLM can "read" the spatial layout.
        """
        if not tracking_data:
            return f"[Frame {frame_idx}] No players tracked."
            
        my_team_count = sum(1 for p in tracking_data if p.get('is_my_team'))
        opp_team_count = len(tracking_data) - my_team_count
        
        semantic_doc = f"--- [Frame {frame_idx}] Tactical Snapshot ---\n"
        semantic_doc += f"Spatial Density: {my_team_count} Attacking Players vs {opp_team_count} Defending Players.\n"
        
        # Analyze Intel
        if frame_intel:
            ball_carrier_id = frame_intel.get('ball_carrier')
            if ball_carrier_id is not None:
                semantic_doc += f"Possession Status: Player ID {ball_carrier_id} is in control of the ball.\n"
            
            # Contextual xG or Risk
            c_xg = frame_intel.get('contextual_xg')
            if c_xg:
                semantic_doc += f"Current Formational xG Build-Up: {c_xg:.3f}. "
                if c_xg > 0.15:
                    semantic_doc += "HIGH THREAT MOMENTUM.\n"
                else:
                    semantic_doc += "Building slowly from the back.\n"
                    
            # Pass Engine Output
            ranked_passes = frame_intel.get('ranked_passes')
            if ranked_passes and len(ranked_passes) > 0:
                best_pass = ranked_passes[0]
                semantic_doc += f"Highest Expected Value (EPV) Pass Option: Target at coordinates ({best_pass['pos'][0]:.1f}, {best_pass['pos'][1]:.1f}) "
                semantic_doc += f"yielding an EPV spike of +{best_pass['ev']: .3f} with {best_pass['pass_prob']*100:.0f}% completion probability.\n"

        return semantic_doc

    def ingest_match_data(self, json_tracking_path: str, csv_intel_path: str = None) -> int:
        """
        Ingests the entire exported match data log, serializes it into semantic 
        chunks, and embeds it into the vector database.
        
        Args:
            json_tracking_path: Path to the exported tracking_data.json
            csv_intel_path: Optional path to the exported intel csv
            
        Returns:
            Number of documents ingested.
        """
        if not self.collection:
            logging.error("RAG Collection not initialized. Cannot ingest.")
            return 0
            
        if not os.path.exists(json_tracking_path):
            logging.error(f"Tracking JSON not found at {json_tracking_path}.")
            return 0

        # Note: In a production scenario, we'd batch these or only embed 
        # significant moments (e.g. shots, turnovers, high xT spikes) to save DB overhead.
        # Here we embed keyframes (e.g. every 30th frame ~ 1 second of play).
        
        try:
            with open(json_tracking_path, 'r') as f:
                tracking_log = json.load(f)
                
            docs = []
            metadatas = []
            ids = []
            
            # Simple Keyframe extraction (1 per second assuming 30fps)
            keyframes = [frame for i, frame in enumerate(tracking_log) if i % 30 == 0]
            
            for kf in keyframes:
                frame_idx = kf.get('frame', 0)
                players = kf.get('players', [])
                
                # Mock intel merging since JSON currently doesn't hold the rich intel dict
                # In a full flow, we'd cross-reference the `csv_intel_path` here via pandas.
                # For the hackathon RAG, we'll embed the core spatial truth.
                
                my_team_speed = sum(p.get('speed', 0) for p in players if p.get('is_my_team', False))
                
                semantic_text = f"[Frame {frame_idx}] Tactical Setup: "
                semantic_text += f"{len(players)} players actively tracked. "
                if my_team_speed > 20.0 * 5: # Arbitrary high total threshold
                    semantic_text += "High urgency / Sprinting phase detected. "
                
                if len(players) > 0:
                   semantic_text += f"Average team depth is mapped."
                
                docs.append(semantic_text)
                metadatas.append({"frame_idx": frame_idx, "type": "spatial_snapshot"})
                ids.append(f"snapshot_{frame_idx}")
                
            if docs:
                self.collection.add(
                    documents=docs,
                    metadatas=metadatas,
                    ids=ids
                )
                logging.info(f"Ingested {len(docs)} keyframe moments into ChromaDB RAG.")
                return len(docs)
                
        except Exception as e:
            logging.error(f"Error during RAG ingestion: {e}")
            
        return 0

    def query_tactics(self, natural_language_query: str, n_results: int = 3) -> List[str]:
        """
        Takes a natural language question (e.g. "When was our highest threat?"), 
        embeds it, and retrieves the closest frame clusters.
        """
        if not self.collection:
            return ["RAG Database Offline. Semantic search disabled."]
            
        try:
            results = self.collection.query(
                query_texts=[natural_language_query],
                n_results=n_results
            )
            
            # results['documents'] is a list of lists: [['doc1', 'doc2']]
            if results and 'documents' in results and len(results['documents']) > 0:
                retrieved_docs = results['documents'][0]
                return retrieved_docs
            return ["No highly relevant tactical moments found for this query."]
            
        except Exception as e:
            logging.error(f"RAG Query failed: {e}")
            return [f"Query Error: {str(e)}"]

# Example usage/tester
if __name__ == "__main__":
    rag = MatchDataRAG(persist_directory=".chroma_mock")
    print(rag.query_tactics("What happened in the early frames?"))
