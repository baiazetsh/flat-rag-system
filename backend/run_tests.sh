#!/bin/bash
# backend/run_tests.sh
# Script to run automated tests for the RAG project

set -e

echo "🧪 Running RAG Project Tests"
echo "=============================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Check pytest
if ! python -m pytest --version > /dev/null 2>&1; then
    print_warning "Installing pytest..."
    pip install pytest pytest-asyncio pytest-cov httpx
fi

mkdir -p test_reports

run_tests() {
    local type=$1
    local path="app/tests"
    local markers=$2

    echo ""
    echo "Running $type tests..."
    echo "----------------------------"

    if [ -n "$markers" ]; then
        pytest "$path" -m "$markers" -v --tb=short \
            --cov=app --cov-report=term-missing \
            --cov-report=html:test_reports/coverage_$type \
            --junitxml=test_reports/junit_$type.xml
    else
        pytest "$path" -v --tb=short \
            --cov=app --cov-report=term-missing \
            --cov-report=html:test_reports/coverage_$type \
            --junitxml=test_reports/junit_$type.xml
    fi

    print_status "$type tests completed"
}

case "${1:-all}" in
    unit)        run_tests "unit" "unit" ;;
    integration) run_tests "integration" "integration" ;;
    api)         run_tests "api" "api" ;;
    fast)        pytest app/tests -v --tb=short -m "not slow" ;;
    specific)
        if [ -z "$2" ]; then
            print_error "Specify a test target"
            exit 1
        fi
        pytest "$2" -v --tb=short
        ;;
    all|*)
        run_tests "all" ""
        ;;
esac

echo ""
echo "=============================="
echo "All done."
echo "=============================="
