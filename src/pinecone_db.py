from pinecone import Pinecone, ServerlessSpec 
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
pinecone_client = Pinecone()  


class vectorDB:
    def __init__(self, index_name ):
        self.MODEL = "text-embedding-3-small"
        self.index_name = index_name

    def data_preprocess(self):
        try:
            file_path  = "src\knowledge_base.txt"

            with open(file_path, "r") as f:
                lines = f.readlines()

            texts = [p.strip() for p in lines]
            texts = [text for text in texts if text]

            content = '\n'.join(texts)
            chunks = []
            start = 0
            while start < len(content):
                end = min(start + 120, len(content))
                if end < len(content):
                    boundary = content.rfind(' ', start, end)
                    if boundary > start:
                        end = boundary
                chunk = content[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == len(content):
                    break
                start = max(end - 10, start + 1)
            return chunks
        
        except Exception as e:
            raise e
    
    def init_pinecone(self):
        try:  
            spec = ServerlessSpec(cloud='aws', region='us-east-1')    
 
            pinecone_client.create_index(  
                self.index_name,  
                dimension=1536,  # dimensionality of text-embedding-ada-002  
                metric='dotproduct',  
                spec=spec  
            )
            
        except Exception as e:
            raise e
        
    def upload_embeddings(self):
        try:
            self.init_pinecone() #index created
            token_split_texts = self.data_preprocess() #chunks array created

            index = pinecone_client.Index(self.index_name)

            count = 0  # we'll use the count to create unique IDs
            batch_size = 4  # process everything in batches of 4
            for i in range(0, len(token_split_texts), batch_size):
                # set end position of batch
                i_end = min(i+batch_size, len(token_split_texts))
                # get batch of lines and IDs
                lines_batch = token_split_texts[i: i+batch_size]
                
                ids_batch = [str(n) for n in range(i, i_end)]

                # create embeddings
                res = client.embeddings.create(input=lines_batch, model=self.MODEL)
                embeds = [record.embedding for record in res.data]

                # prep metadata and upsert batch
                meta = [{'text': line} for line in lines_batch]

                to_upsert = zip(ids_batch, embeds, meta)
                # upsert to Pinecone
                index.upsert(vectors=list(to_upsert))
            print(index.describe_index_stats())
        except Exception as e:
            raise e
        
    def handle_query(self, query):
        try:
            index = pinecone_client.Index(self.index_name)
            emb = client.embeddings.create(input=query, model=self.MODEL).data[0].embedding
            retrived_text = index.query(vector=[emb], top_k=2, include_metadata=True)
            output = [match['metadata']['text'] for match in retrived_text["matches"]]

            return "\n".join(output)


        except Exception as e:
            raise e
        

# if __name__ == "__main__":
#     index_name = "support1-knowledgebase"
#     pinecone_db =vectorDB(index_name)
#     pinecone_db.upload_embeddings()
#     print(pinecone_db.handle_query("waht is your company name?"))