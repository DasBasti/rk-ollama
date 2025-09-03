#!/bin/bash

# Quick VS Code Compatibility Check
# This script performs a minimal check to verify VS Code compatibility
# Usage: ./vscode_check.sh [server_url]

SERVER_URL="${1:-http://localhost:11434}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 VS Code Ollama Compatibility Check${NC}"
echo "Server: $SERVER_URL"
echo "=================================="

# Function to check endpoint
check() {
    local name="$1"
    local test_cmd="$2"
    printf "%-30s" "$name:"
    
    if eval "$test_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

total=0
passed=0

# Check 1: Server responds
((total++))
if check "Server connectivity" "curl -s --max-time 5 '$SERVER_URL/api/version'"; then
    ((passed++))
fi

# Check 2: Version endpoint
((total++))
if check "Version endpoint" "curl -s --max-time 5 '$SERVER_URL/api/version' | jq -e '.version'"; then
    ((passed++))
fi

# Check 3: Models endpoint with proper structure
((total++))
if check "Models endpoint" "curl -s --max-time 5 '$SERVER_URL/api/tags' | jq -e '.models[0].name'"; then
    ((passed++))
fi

# Check 4: Model format is gguf
((total++))
format=$(curl -s --max-time 5 "$SERVER_URL/api/tags" 2>/dev/null | jq -r '.models[0].details.format // empty')
if [ "$format" = "gguf" ]; then
    echo -e "Model format (gguf):          ${GREEN}✅ PASS${NC}"
    ((passed++))
else
    echo -e "Model format (gguf):          ${RED}❌ FAIL${NC} (got: $format)"
fi

# Check 5: CORS headers
((total++))
if check "CORS headers" "curl -s -I --max-time 5 '$SERVER_URL/api/version' | grep -i 'access-control-allow-origin'"; then
    ((passed++))
fi

# Check 6: Content-Type header
((total++))
if check "JSON Content-Type" "curl -s -I --max-time 5 '$SERVER_URL/api/version' | grep -i 'content-type.*application/json'"; then
    ((passed++))
fi

# Check 7: Text generation works
((total++))
if check "Text generation" "curl -s --max-time 10 -X POST -H 'Content-Type: application/json' -d '{\"model\":\"gemma2:2b\",\"prompt\":\"Hi\",\"stream\":false}' '$SERVER_URL/api/generate' | jq -e '.response'"; then
    ((passed++))
fi

echo
echo "=================================="
echo "Results: $passed/$total tests passed"

if [ "$passed" -eq "$total" ]; then
    echo -e "${GREEN}🎉 SUCCESS: Your server is VS Code compatible!${NC}"
    echo
    echo "✅ VS Code Setup Instructions:"
    echo "1. Install an Ollama extension in VS Code"
    echo "2. Set Ollama URL to: $SERVER_URL"
    echo "3. Select model: gemma2:2b"
    echo "4. Start using AI assistance in VS Code!"
    exit 0
else
    echo -e "${RED}❌ ISSUES FOUND: Fix the failing checks above${NC}"
    echo
    echo "🔧 Common fixes:"
    echo "• Ensure server is running and accessible"
    echo "• Check model format is 'gguf' not 'rkllm'"
    echo "• Verify CORS headers are enabled"
    echo "• Test endpoints manually with curl"
    exit 1
fi
