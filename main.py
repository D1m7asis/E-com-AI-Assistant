import os
import polars as pl

from submission.sales_assistant import SalesAssistant

from dotenv import load_dotenv


load_dotenv()


if __name__ == "__main__":
    assistant = SalesAssistant(
        gigachat_credentials=os.environ.get("GIGACHAT_TOKEN"),
        gigachat_scope=os.environ.get("GIGACHAT_SCOPE"),
        categories=pl.read_parquet("submission/data/categories.parquet"),
        items=pl.read_parquet("submission/data/items.parquet"),
    )
    
    assistant_message = assistant.start()

    while True:
        print(f"assistant:\t{assistant_message}")
        user_message = input("user:\t")

        assistant_message = assistant.chat(user_message)
