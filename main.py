from pathlib import Path

from src.data_ingestion import ingest_air_quality_data
from src.settings import OPENAQ_API_KEY
from src.pipeline import run_full_pipeline


print("API key loaded successfully!")
print(OPENAQ_API_KEY[:8] + "...")

result = run_full_pipeline()
print("\nPipeline completed. Summary:")
print(result)

