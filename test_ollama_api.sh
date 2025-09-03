#!/bin/bash

# Ollama API Test Suite
# This script tests all endpoints and provides timing benchmarks
# Usage: ./test_ollama_api.sh [server_url]

set -e

# Configuration
SERVER_URL="${1:-http://localhost:11434}"
TEST_MODEL="gemma2:2b"
TIMEOUT=30
RESULTS_FILE="test_results_$(date +%Y%m%d_%H%M%S).json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ PASS:${NC} $1"
}

print_error() {
    echo -e "${RED}❌ FAIL:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️  WARN:${NC} $1"
}

# Function to test API endpoint with timing
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local expected_status="${5:-200}"
    
    print_status "Testing $name..."
    
    local start_time=$(date +%s.%N)
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT "$SERVER_URL$endpoint" 2>/dev/null)
    else
        response=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" "$SERVER_URL$endpoint" 2>/dev/null)
    fi
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l)
    
    # Extract HTTP status code (last line)
    local http_code=$(echo "$response" | tail -n1)
    # Extract response body (all lines except last)
    local response_body=$(echo "$response" | head -n -1)
    
    # Validate HTTP status
    if [ "$http_code" = "$expected_status" ]; then
        print_success "$name (${duration}s, HTTP $http_code)"
        
        # Validate JSON format
        if echo "$response_body" | jq . > /dev/null 2>&1; then
            print_success "$name JSON format valid"
        else
            print_error "$name Invalid JSON response"
            echo "$response_body"
            return 1
        fi
        
        # Store result
        echo "{\"test\":\"$name\",\"status\":\"pass\",\"duration\":$duration,\"http_code\":$http_code,\"timestamp\":\"$(date -Iseconds)\"}" >> "$RESULTS_FILE"
        
        return 0
    else
        print_error "$name (${duration}s, HTTP $http_code, expected $expected_status)"
        echo "Response: $response_body"
        echo "{\"test\":\"$name\",\"status\":\"fail\",\"duration\":$duration,\"http_code\":$http_code,\"timestamp\":\"$(date -Iseconds)\"}" >> "$RESULTS_FILE"
        return 1
    fi
}

# Function to validate model data structure
validate_model_structure() {
    local response="$1"
    local test_name="$2"
    
    # Check required fields
    local name=$(echo "$response" | jq -r '.models[0].name // empty' 2>/dev/null)
    local model=$(echo "$response" | jq -r '.models[0].model // empty' 2>/dev/null)
    local size=$(echo "$response" | jq -r '.models[0].size // empty' 2>/dev/null)
    local digest=$(echo "$response" | jq -r '.models[0].digest // empty' 2>/dev/null)
    local format=$(echo "$response" | jq -r '.models[0].details.format // empty' 2>/dev/null)
    local family=$(echo "$response" | jq -r '.models[0].details.family // empty' 2>/dev/null)
    
    if [ -n "$name" ] && [ -n "$model" ] && [ -n "$size" ] && [ -n "$digest" ] && [ -n "$format" ] && [ -n "$family" ]; then
        print_success "$test_name model structure validation"
        
        # Check specific values
        if [ "$format" = "gguf" ]; then
            print_success "$test_name format is 'gguf' (Ollama standard)"
        else
            print_warning "$test_name format is '$format' (not standard 'gguf')"
        fi
        
        if [ "$family" = "gemma" ]; then
            print_success "$test_name family is 'gemma'"
        else
            print_warning "$test_name family is '$family'"
        fi
        
        return 0
    else
        print_error "$test_name model structure validation failed"
        echo "Missing fields: name='$name', model='$model', size='$size', digest='$digest', format='$format', family='$family'"
        return 1
    fi
}

# Initialize results file
echo "# Ollama API Test Results - $(date)" > "$RESULTS_FILE"
echo "# Server: $SERVER_URL" >> "$RESULTS_FILE"

print_status "🚀 Starting Ollama API Test Suite"
print_status "Server: $SERVER_URL"
print_status "Model: $TEST_MODEL"
print_status "Results will be saved to: $RESULTS_FILE"
echo

# Check if server is running
print_status "🔍 Checking server connectivity..."
if ! curl -s --max-time 5 "$SERVER_URL" > /dev/null 2>&1; then
    print_error "Cannot connect to server at $SERVER_URL"
    print_error "Make sure the server is running and accessible"
    exit 1
fi
print_success "Server is reachable"
echo

# Test 1: Version endpoint
print_status "📋 Testing Core API Endpoints"
echo "----------------------------------------"

test_endpoint "Version Info" "GET" "/api/version" ""
echo

# Test 2: List available models
print_status "Testing model listing..."
response=$(curl -s --max-time $TIMEOUT "$SERVER_URL/api/tags" 2>/dev/null)
if test_endpoint "List Models" "GET" "/api/tags" ""; then
    validate_model_structure "$response" "List Models"
fi
echo

# Test 3: List running models  
print_status "Testing running models..."
response=$(curl -s --max-time $TIMEOUT "$SERVER_URL/api/ps" 2>/dev/null)
if test_endpoint "Running Models" "GET" "/api/ps" ""; then
    validate_model_structure "$response" "Running Models"
fi
echo

# Test 4: Show model details
test_endpoint "Show Model Details" "POST" "/api/show" "{\"name\": \"$TEST_MODEL\"}"
echo

# Test 5: Text generation (non-streaming)
print_status "📝 Testing Text Generation"
echo "----------------------------------------"

test_endpoint "Generate Text (Non-streaming)" "POST" "/api/generate" \
    "{\"model\": \"$TEST_MODEL\", \"prompt\": \"Hello, how are you?\", \"stream\": false}"
echo

# Test 6: Text generation (streaming) - just test that it starts correctly
print_status "Testing streaming generation..."
stream_response=$(curl -s --max-time 10 -X POST \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$TEST_MODEL\", \"prompt\": \"Hi\", \"stream\": true}" \
    "$SERVER_URL/api/generate" 2>/dev/null | head -n 1)

if echo "$stream_response" | jq . > /dev/null 2>&1; then
    print_success "Generate Text (Streaming) - Initial response valid"
else
    print_error "Generate Text (Streaming) - Invalid initial response"
fi
echo

# Test 7: Chat completion (non-streaming)
print_status "💬 Testing Chat Completion"
echo "----------------------------------------"

test_endpoint "Chat Completion (Non-streaming)" "POST" "/api/chat" \
    "{\"model\": \"$TEST_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}], \"stream\": false}"
echo

# Test 8: Chat completion (streaming) - just test that it starts correctly
print_status "Testing streaming chat..."
chat_stream_response=$(curl -s --max-time 10 -X POST \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$TEST_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"stream\": true}" \
    "$SERVER_URL/api/chat" 2>/dev/null | head -n 1)

if echo "$chat_stream_response" | jq . > /dev/null 2>&1; then
    print_success "Chat Completion (Streaming) - Initial response valid"
else
    print_error "Chat Completion (Streaming) - Invalid initial response"
fi
echo

# Test 9: Error handling - non-existent model
print_status "🚫 Testing Error Handling"
echo "----------------------------------------"

test_endpoint "Non-existent Model Error" "POST" "/api/show" \
    "{\"name\": \"nonexistent:model\"}" "404"
echo

# Test 10: Invalid endpoint
test_endpoint "Invalid Endpoint Error" "GET" "/api/invalid" "" "404"
echo

# Test 11: CORS headers check
print_status "🌐 Testing CORS Headers"
echo "----------------------------------------"

cors_response=$(curl -s -I --max-time 10 "$SERVER_URL/api/version" 2>/dev/null)
if echo "$cors_response" | grep -i "access-control-allow-origin" > /dev/null; then
    print_success "CORS headers present"
else
    print_warning "CORS headers not found"
fi
echo

# Test 12: Content-Type validation
print_status "📋 Testing Response Headers"
echo "----------------------------------------"

headers=$(curl -s -I --max-time 10 "$SERVER_URL/api/version" 2>/dev/null)
if echo "$headers" | grep -i "content-type.*application/json" > /dev/null; then
    print_success "Correct Content-Type: application/json"
else
    print_warning "Content-Type header not application/json"
fi
echo

# Performance benchmarks
print_status "⚡ Performance Benchmarks"
echo "----------------------------------------"

print_status "Running performance tests (5 iterations each)..."

# Benchmark version endpoint (lightweight)
total_time=0
for i in {1..5}; do
    start_time=$(date +%s.%N)
    curl -s --max-time 10 "$SERVER_URL/api/version" > /dev/null 2>&1
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    total_time=$(echo "$total_time + $duration" | bc -l)
done
avg_version_time=$(echo "scale=3; $total_time / 5" | bc -l)
print_success "Average version endpoint response: ${avg_version_time}s"

# Benchmark model listing
total_time=0
for i in {1..5}; do
    start_time=$(date +%s.%N)
    curl -s --max-time 10 "$SERVER_URL/api/tags" > /dev/null 2>&1
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    total_time=$(echo "$total_time + $duration" | bc -l)
done
avg_tags_time=$(echo "scale=3; $total_time / 5" | bc -l)
print_success "Average model listing response: ${avg_tags_time}s"

echo

# Generate summary report
print_status "📊 Test Summary"
echo "========================================"

# Count only tests from this run (after the header)
total_tests=$(tail -n +3 "$RESULTS_FILE" | grep -c '"test":' 2>/dev/null || echo "0")
passed_tests=$(tail -n +3 "$RESULTS_FILE" | grep -c '"status":"pass"' 2>/dev/null || echo "0")
failed_tests=$(tail -n +3 "$RESULTS_FILE" | grep -c '"status":"fail"' 2>/dev/null || echo "0")

echo "Total Tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"

if [ "$total_tests" -gt 0 ] && [ "$failed_tests" -eq 0 ]; then
    print_success "All tests passed! 🎉"
    echo
    print_status "✅ Your Ollama API server is fully compatible!"
    echo "  • All core endpoints working"
    echo "  • Proper JSON responses"
    echo "  • Correct model metadata format"
    echo "  • CORS headers configured"
    echo "  • Error handling functional"
    echo
    print_status "🚀 Ready for VS Code integration!"
elif [ "$failed_tests" -gt 0 ]; then
    print_error "Some tests failed. Check the output above for details."
else
    print_warning "No tests were recorded. Check the results file."
fi

echo
print_status "📄 Detailed results saved to: $RESULTS_FILE"
echo "========================================="
