# Collector AI Agent

## Workflow
1. User sends the query to the agent.
2. Agent sends the query along with some public user information to an LLM for making an optimized query.
3. The optimized query is used for getting RAG documents.
4. The retrieved documents (if any) are then sent to the LLM in a second pass-through to get relevant results.