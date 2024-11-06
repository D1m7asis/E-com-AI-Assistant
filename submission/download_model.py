from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    model_big = SentenceTransformer('intfloat/multilingual-e5-large')
    model_small = SentenceTransformer('intfloat/multilingual-e5-small')
