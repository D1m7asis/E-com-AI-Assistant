from sentence_transformers import SentenceTransformer

from chromadb.api import ClientAPI, Collection
from chromadb.types import Metadata

from typing import List
import re
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing import preprocess_string
from string import punctuation

ENCODING_MODEL_BIG = SentenceTransformer("intfloat/multilingual-e5-large")
ENCODING_MODEL_SMALL = SentenceTransformer("intfloat/multilingual-e5-small")


def normalize_text(user_prompt: str) -> str:
    text = user_prompt.lower()

    # Удаление пунктуации и цифр
    text = re.sub(r"[{}]".format(punctuation), "", text)
    text = re.sub(r"\b\d+\b", "", text)

    # Препроцессинг текста с помощью gensim
    words = preprocess_string(text)

    # Удаление стоп-слов
    words = [word for word in words if word not in STOPWORDS]

    normalized_text = " ".join(words).strip()
    return normalized_text


def search_categories(
    user_prompt: str, categories_collection: Collection, topn: int
) -> List[List[Metadata]]:
    query_embedding = ENCODING_MODEL_BIG.encode([normalize_text(user_prompt)])
    results = categories_collection.query(query_embeddings=query_embedding, n_results=topn)
    return results["metadatas"]


def search_items(user_prompt: str, items_vector_collection, topn: int, category_id: str) -> List[Metadata]:
    query_embedding = ENCODING_MODEL_SMALL.encode([user_prompt])

    try:
        search_results = items_vector_collection.query(
            query_embeddings=query_embedding,
            n_results=topn,
            where={"category_id": category_id}
        )

        result = search_results["metadatas"][0] if "metadatas" in search_results else []

    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        result = []

    return result


def get_chroma_collection(client: ClientAPI, collection_name: str) -> Collection:
    """
    Returns chroma db collection by name.
    """
    print("Chroma init")
    collection = client.get_collection(collection_name)

    print("Init finished")
    return collection
