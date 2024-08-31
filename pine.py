from pinecone import Pinecone,ServerlessSpec # Correct import for Pinecone client
import google.generativeai as genai

# Configure Google Generative AI API
genai.configure(api_key="AIzaSyDGJFFUTdcyFzaIcgS698-I7ZvZiWK0WuI")

# Initialize Pinecone
pinecone.init(api_key="4f3dd80b-1c86-4be6-90b4-7f0f21b6bc06", environment="us-west1-gcp")  # Specify the environment



def clean_vector_id(vector_id):
    vector_id = ''.join(char for char in vector_id if ord(char) < 128)
    return vector_id



def generate_embeddings(text):
    result = genai.embed_content(
    model="models/text-embedding-004", 
    content=text,
    task_type="retrieval_document")
    embeddings = result['embedding']
    return embeddings
        
        
def store(text,namespace,metadata):
    index = pc.Index("classroom")
    vectors = []
    for i in text:
        vectors.append({
            "id" : clean_vector_id(str(i)),
            "values" : generate_embeddings(text=clean_vector_id(str(i))),
            "metadata" : metadata
        })
    index.upsert(vectors=vectors,namespace=namespace)
    print('Embeding Generation Done')


def similarity(text,namespace):
    index = pc.Index("classroom")
    query_results1 = index.query(
    namespace=namespace,
    vector=generate_embeddings(text=text),
    top_k=10,
    include_values=True
    )
    for i in query_results1.matches:
        if i :
            text+=str(i.id)
    return text

