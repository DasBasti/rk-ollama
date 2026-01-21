#!/bin/bash

# Ollama API Performance Benchmark
# This script focuses on performance testing and benchmarking
# Usage: ./benchmark_ollama.sh [server_url] [iterations]

set -e

# Configuration
SERVER_URL="${1:-http://localhost:11434}"
ITERATIONS="${2:-10}"
TEST_MODEL="gemma2:2b"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_result() {
    echo -e "${GREEN}$1${NC}"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

# Function to calculate statistics
calculate_stats() {
    local times=("$@")
    local sum=0
    local min=${times[0]}
    local max=${times[0]}
    
    for time in "${times[@]}"; do
        sum=$(echo "$sum + $time" | bc -l)
        if (( $(echo "$time < $min" | bc -l) )); then
            min=$time
        fi
        if (( $(echo "$time > $max" | bc -l) )); then
            max=$time
        fi
    done
    
    local avg=$(echo "scale=3; $sum / ${#times[@]}" | bc -l)
    
    echo "Average: ${avg}s"
    echo "Min: ${min}s"
    echo "Max: ${max}s"
}

# Function to benchmark endpoint
benchmark_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    
    print_header "$name Benchmark ($ITERATIONS iterations)"
    
    local times=()
    
    for ((i=1; i<=ITERATIONS; i++)); do
        local start_time=$(date +%s.%N)
        
        if [ "$method" = "GET" ]; then
            curl -s --max-time 30 "$SERVER_URL$endpoint" > /dev/null 2>&1
        else
            curl -s --max-time 30 -X "$method" \
                -H "Content-Type: application/json" \
                -d "$data" "$SERVER_URL$endpoint" > /dev/null 2>&1
        fi
        
        local end_time=$(date +%s.%N)
        local duration=$(echo "scale=3; $end_time - $start_time" | bc -l)
        times+=($duration)
        
        printf "Iteration %2d: %ss\n" $i $duration
    done
    
    echo
    print_result "Statistics:"
    calculate_stats "${times[@]}"
    echo
}

print_header "Ollama API Performance Benchmark"
print_info "Server: $SERVER_URL"
print_info "Model: $TEST_MODEL" 
print_info "Iterations per test: $ITERATIONS"
echo

# Check server connectivity
print_info "Checking server connectivity..."
if ! curl -s --max-time 5 "$SERVER_URL/api/version" > /dev/null 2>&1; then
    echo "❌ Cannot connect to server at $SERVER_URL"
    exit 1
fi
echo "✅ Server is reachable"
echo

# Benchmark lightweight endpoints1"

benchmark_endpoint "List Models" "GET" "/api/tags" ""

benchmark_endpoint "Running Models" "GET" "/api/ps" ""

# Benchmark generation endpoints (shorter prompts for faster testing)
benchmark_endpoint "Text Generation" "POST" "/api/generate" \
    "{\"model\": \"$TEST_MODEL\", \"prompt\": \"Hi\", \"stream\": false}"

benchmark_endpoint "Chat Completion" "POST" "/api/chat" \
    "{\"model\": \"$TEST_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"stream\": false}"

print_header "Benchmark Complete"
echo "Use this data to:"
echo "• Monitor performance over time"
echo "• Identify performance regressions"
echo "• Compare against other Ollama implementations"
echo "• Optimize server configuration"
