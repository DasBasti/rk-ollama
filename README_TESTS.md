# Ollama API Test Suite

This directory contains comprehensive testing and benchmarking tools for your Ollama-compatible API server.

## Files Overview

### 🧪 `test_ollama_api.sh`
**Comprehensive Test Suite** - Full API validation and compatibility testing

**Usage:**
```bash
./test_ollama_api.sh [server_url]
```

**Examples:**
```bash
# Test local server
./test_ollama_api.sh

# Test remote server
./test_ollama_api.sh http://192.168.1.100:11434

# Test with custom port
./test_ollama_api.sh http://localhost:8080
```

**What it tests:**
- ✅ All API endpoints (`/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/generate`, `/api/chat`)
- ✅ JSON response format validation
- ✅ Model metadata structure compliance
- ✅ CORS headers
- ✅ Error handling (404s, invalid models)
- ✅ Streaming vs non-streaming responses
- ✅ Performance benchmarks
- ✅ VS Code compatibility validation

**Output:**
- Console report with pass/fail status
- Detailed JSON results file (`test_results_YYYYMMDD_HHMMSS.json`)
- Performance timing data

---

### ⚡ `benchmark_ollama.sh` 
**Performance Benchmark Suite** - Focused on timing and performance metrics

**Usage:**
```bash
./benchmark_ollama.sh [server_url] [iterations]
```

**Examples:**
```bash
# Default: 10 iterations on localhost
./benchmark_ollama.sh

# Custom iterations
./benchmark_ollama.sh http://localhost:11434 20

# Stress test
./benchmark_ollama.sh http://localhost:11434 100
```

**What it measures:**
- Response times for each endpoint
- Statistical analysis (min, max, average)
- Performance trends over multiple iterations
- Identifies performance bottlenecks

---

### 📝 `curl_commands.md`
**Quick Reference** - Manual testing commands

Contains 20+ ready-to-use curl commands for:
- Testing individual endpoints
- Debugging specific issues
- Performance testing
- Error condition testing
- Header validation

**Usage:**
```bash
# Set your server URL
export SERVER_URL="http://localhost:11434"

# Copy and paste any command from the file
curl -s "$SERVER_URL/api/version" | jq .
```

---

## Quick Start

1. **Start your Ollama API server**
   ```bash
   cargo run
   ```

2. **Run the full test suite**
   ```bash
   ./test_ollama_api.sh
   ```

3. **If all tests pass:** Your server is VS Code compatible! 🎉

4. **If tests fail:** Check the output for specific issues to fix

## Test Categories

### 🔧 **Core Functionality**
- API endpoint availability
- JSON response format
- Model metadata structure

### 🚀 **Performance**
- Response time benchmarks
- Concurrent request handling
- Memory usage patterns

### 🔒 **Compatibility**
- Ollama API specification compliance
- VS Code extension compatibility
- CORS header validation

### 🚨 **Error Handling**
- 404 error responses
- Invalid model handling
- Malformed request handling

## Interpreting Results

### ✅ **All Tests Pass**
Your server is fully compatible with:
- VS Code Ollama extensions
- Official Ollama clients
- Third-party Ollama tools

### ❌ **Some Tests Fail**
Common issues and fixes:

1. **JSON Format Errors**
   - Check response structure matches Ollama API spec
   - Ensure proper Content-Type headers

2. **Model Structure Validation Fails**
   - Verify model metadata includes all required fields
   - Check `format`, `family`, and `quantization_level` values

3. **Endpoint Not Found (404)**
   - Ensure all required endpoints are implemented
   - Check routing configuration

4. **CORS Issues**
   - Add proper CORS headers for web-based tools
   - Enable cross-origin requests

## Continuous Integration

### **Regression Testing**
Run the test suite after any code changes:
```bash
# Before committing changes
./test_ollama_api.sh > test_results.log
```

### **Performance Monitoring**
Track performance over time:
```bash
# Daily performance check
./benchmark_ollama.sh http://production-server:11434 50 >> performance_log.txt
```

### **Automated Testing**
Add to your CI/CD pipeline:
```yaml
# Example GitHub Actions step
- name: Test Ollama API
  run: |
    ./test_ollama_api.sh http://localhost:11434
    if [ $? -eq 0 ]; then
      echo "✅ All tests passed"
    else
      echo "❌ Tests failed"
      exit 1
    fi
```

## Troubleshooting

### **Server Not Responding**
```bash
# Check if server is running
curl -I http://localhost:11434/api/version

# Check server logs
tail -f server.log
```

### **Performance Issues**
```bash
# Run focused performance test
./benchmark_ollama.sh http://localhost:11434 5

# Monitor system resources
htop  # or top
```

### **VS Code Integration Issues**
1. Run full test suite: `./test_ollama_api.sh`
2. Check VS Code settings point to correct server URL
3. Verify model names match between server and VS Code
4. Check CORS headers are present

## Adding New Tests

To add a new test to the suite:

1. **Add test function** in `test_ollama_api.sh`:
   ```bash
   test_endpoint "My New Test" "GET" "/api/new-endpoint" ""
   ```

2. **Add validation** if needed:
   ```bash
   validate_new_response "$response" "My New Test"
   ```

3. **Update documentation** in this README

## Support

If you encounter issues:
1. Check server logs for errors
2. Run individual curl commands from `curl_commands.md`
3. Compare your responses with the Ollama API specification
4. Test against a real Ollama server for comparison

---

**Happy Testing! 🚀**
