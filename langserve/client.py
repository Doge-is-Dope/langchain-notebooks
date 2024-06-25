from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/")

# Invoke API
result = chain.invoke({"language": "italy", "text": "Hi"})
print(result)

# Batch API
results = chain.batch(
    [{"language": "italy", "text": "Hi"}, {"language": "italy", "text": "I'm a human."}]
)
for result in results:
    print(result, end=" ")


#
