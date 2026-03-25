import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.agents.rag_pipeline import MatchDataRAG

def main():
    print("Loading Tactical Data into ChromaDB RAG...")
    
    # Path to your tracking output (adjust if your output dir is different)
    tracking_json = "output/tracking_data.json"
    
    if not os.path.exists(tracking_json):
        print(f"Error: {tracking_json} not found. Please run the tracking_system/main.py pipeline first.")
        return

    rag = MatchDataRAG(persist_directory="backend/chroma_db")
    
    # Ingest the JSON data into the vector database
    docs_ingested = rag.ingest_match_data(tracking_json)
    
    if docs_ingested > 0:
        print(f"Success! {docs_ingested} tactical frame clusters ingested into the Vector DB.")
        print("The Tactical Assistant is now ready to answer queries.")
    else:
        print("Ingestion failed or no data was processed.")

if __name__ == "__main__":
    main()
