import polars as pl
import chromadb
from chromadb.api import ClientAPI
from sentence_transformers import SentenceTransformer


def create_chroma_collection(
    client: ClientAPI,
    categories_dataframe: pl.DataFrame,
    collection_name: str = "categories",
) -> None:
    category_collection = client.create_collection(collection_name)
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

    category_names = categories_dataframe["name"].to_list()
    category_embeddings = embedding_model.encode(category_names, show_progress_bar=True)
    category_embeddings_list = [embedding.tolist() for embedding in category_embeddings]

    print("Adding categories to Chroma")
    for index, embedding in enumerate(category_embeddings_list):
        row = categories_dataframe.row(index)
        category_collection.add(
            ids=[str(row[0])],
            documents=[category_names[index]],
            embeddings=[embedding],
            metadatas=[{"id": row[0], "name": row[2]}],
        )


if __name__ == "__main__":
    chroma_client = chromadb.PersistentClient(path="../Nodes/chroma_data")
    categories_dataframe = pl.read_parquet("data/categories.parquet")
    create_chroma_collection(chroma_client, categories_dataframe, "categories")
