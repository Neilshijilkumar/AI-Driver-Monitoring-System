from RAG_system.rag import retrieve_response

# Test query

query = "phone usage while driving"

# Retrieve response

response = retrieve_response(query)

# Print output

print("\nRetrieved Response:\n")

print(response)