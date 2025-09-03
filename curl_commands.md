# Ollama API Test Commands Reference
# Quick curl commands for manual testing

# Set server URL (change as needed)
export SERVER_URL="http://localhost:11434"

# 1. Check server version
curl -s "$SERVER_URL/api/version" | jq .

# 2. List available models
curl -s "$SERVER_URL/api/tags" | jq .

# 3. List currently running models
curl -s "$SERVER_URL/api/ps" | jq .

# 4. Show detailed model information
curl -s -X POST "$SERVER_URL/api/show" \
  -H "Content-Type: application/json" \
  -d '{"name": "gemma2:2b"}' | jq .

# 5. Generate text (non-streaming)
curl -s -X POST "$SERVER_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2:2b",
    "prompt": "Hello, how are you?",
    "stream": false
  }' | jq .

# 6. Generate text (streaming) - shows first few responses
curl -X POST "$SERVER_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2:2b",
    "prompt": "Tell me a short story",
    "stream": true
  }' | head -n 5

# 7. Chat completion (non-streaming)
curl -s -X POST "$SERVER_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2:2b",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false
  }' | jq .

# 8. Chat completion (streaming) - shows first few responses
curl -X POST "$SERVER_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2:2b",
    "messages": [
      {"role": "user", "content": "Tell me about artificial intelligence"}
    ],
    "stream": true
  }' | head -n 5

# 9. Chat with conversation history
curl -s -X POST "$SERVER_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2:2b",
    "messages": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there! How can I help you?"},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "stream": false
  }' | jq .

# 10. Generate with options
curl -s -X POST "$SERVER_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2:2b",
    "prompt": "Write a creative sentence",
    "stream": false,
    "options": {
      "temperature": 0.9,
      "top_p": 0.8,
      "top_k": 30,
      "max_tokens": 50
    }
  }' | jq .

# 11. Test error handling - non-existent model
curl -s -X POST "$SERVER_URL/api/show" \
  -H "Content-Type: application/json" \
  -d '{"name": "nonexistent:model"}' | jq .

# 12. Test error handling - invalid endpoint
curl -s "$SERVER_URL/api/invalid" | jq .

# 13. Check response headers (including CORS)
curl -I "$SERVER_URL/api/version"

# 14. Performance test - time a simple request
time curl -s "$SERVER_URL/api/version" > /dev/null

# 15. Load test - multiple concurrent requests (requires GNU parallel)
# echo -e "curl -s $SERVER_URL/api/version\ncurl -s $SERVER_URL/api/tags\ncurl -s $SERVER_URL/api/ps" | parallel -j 3

# 16. Validate JSON response structure for models
curl -s "$SERVER_URL/api/tags" | jq '.models[0] | keys'

# 17. Extract specific model information
curl -s "$SERVER_URL/api/tags" | jq '.models[0] | {name, size, format: .details.format, family: .details.family}'

# 18. Test with different model names (if you have multiple models)
# curl -s -X POST "$SERVER_URL/api/show" -H "Content-Type: application/json" -d '{"name": "gemma2:2b"}' | jq .

# 19. Monitor server logs while testing
# tail -f server.log

# 20. Test streaming with timeout
timeout 10s curl -X POST "$SERVER_URL/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma2:2b",
    "prompt": "Count from 1 to 100",
    "stream": true
  }'
