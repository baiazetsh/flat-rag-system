#!/bin/bash
# backend/run_tests.sh
# Script to run automated tests for the RAG project

set -e

echo "🧪 Running RAG Project Tests"
echo "=============================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions for colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if pytest is installed
if ! python -m pytest --version > /dev/null 2>&1; then
    print_warning "Installing test dependencies..."
    pip install -r requirements-test.txt --break-system-packages

fi

# Create directory for test reports
mkdir -p test_reports

# Function to run a test suite
run_tests() {
    local test_type=$1
    local test_path=$2
    local markers=$3
    
    echo ""
    echo "Running $test_type tests..."
    echo "----------------------------"
    
    if [ -n "$markers" ]; then
        pytest $test_path -m "$markers" -v --tb=short \
            --cov=app --cov-report=term-missing \
            --cov-report=html:test_reports/coverage_$test_type \
            --junitxml=test_reports/junit_$test_type.xml
    else
        pytest $test_path -v --tb=short \
            --cov=app --cov-report=term-missing \
            --cov-report=html:test_reports/coverage_$test_type \
            --junitxml=test_reports/junit_$test_type.xml
    fi
    
    if [ $? -eq 0 ]; then
        print_status "$test_type tests passed"
    else
        print_error "$test_type tests failed"
        return 1
    fi
}

# Argument parsing
case "${1:-all}" in
    unit)
        print_status "Running unit tests only..."
        run_tests "unit" "tests/" "unit"
        ;;
    integration)
        print_status "Running integration tests only..."
        run_tests "integration" "tests/" "integration"
        ;;
    api)
        print_status "Running API tests only..."
        run_tests "api" "tests/" "api"
        ;;
    fast)
        print_status "Running fast tests only (excluding slow ones)..."
        pytest tests/ -v --tb=short -m "not slow" \
            --cov=app --cov-report=term-missing
        ;;
    coverage)
        print_status "Running all tests with detailed coverage..."
        pytest tests/ -v --tb=short \
            --cov=app --cov-report=term-missing \
            --cov-report=html:test_reports/coverage \
            --cov-report=xml:test_reports/coverage.xml
        print_status "Coverage report generated at: test_reports/coverage/index.html"
        ;;
    specific)
        if [ -z "$2" ]; then
            print_error "Please specify a test file or function to run"
            echo "Usage: ./run_tests.sh specific tests/test_utils.py::test_function"
            exit 1
        fi
        print_status "Running specific test: $2"
        pytest "$2" -v --tb=short
        ;;
    all|*)
        print_status "Running all tests..."
        run_tests "all" "tests/" ""
        
        echo ""
        echo "=============================="
        print_status "All tests completed successfully!"
        echo ""
        print_status "Coverage report available at: test_reports/coverage_all/index.html"
        ;;
esac

echo ""
echo "=============================="
echo "Test run finished"
echo "=============================="
