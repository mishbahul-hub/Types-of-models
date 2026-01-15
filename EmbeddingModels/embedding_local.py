from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model = 'sentence-transformers/all-MiniLM-L6-v2'
)

text = "Guwahati is the largest city in the Indian state of Assam."

embedding_vector = embeddings.embed_query(text)

print(str(embedding_vector))