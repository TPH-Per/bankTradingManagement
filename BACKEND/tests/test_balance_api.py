"""
Test Cases for Balance Update API
=================================
Test: /accounts/{account_id}/balance/update

Run: python tests/test_balance_api.py
"""

import requests
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
TEST_ACCOUNT_ID = "0005478"  # Real account


def get_balance(account_id):
    """Get current balance"""
    r = requests.get(f"{BASE_URL}/accounts/{account_id}/balance")
    return r.json() if r.status_code == 200 else None


def update_balance(account_id, amount, operation):
    """Update balance"""
    return requests.post(
        f"{BASE_URL}/accounts/{account_id}/balance/update",
        json={"amount": amount, "operation": operation}
    )


def find_test_account():
    """Find an existing account for testing"""
    try:
        r = requests.get(f"{BASE_URL}/customers/search?query=&search_type=all")
        if r.status_code == 200:
            items = r.json().get("items", [])
            if items:
                return items[0].get("account_id", TEST_ACCOUNT_ID)
    except:
        pass
    return TEST_ACCOUNT_ID


# =============================================================================
# TEST CASES
# =============================================================================

def test_1_add_balance(account_id):
    """Test: Add money"""
    print("\n[TEST 1] Add Balance")
    print("-" * 40)
    
    initial = get_balance(account_id)
    initial_balance = initial.get("balance", 0) if initial else 0
    add_amount = 1000
    
    r = update_balance(account_id, add_amount, "add")
    
    if r.status_code != 200:
        print(f"❌ FAILED: status {r.status_code}")
        print(f"   Response: {r.text}")
        return False
    
    data = r.json()
    expected = initial_balance + add_amount
    actual = data.get("balance", 0)
    
    if abs(actual - expected) < 0.01:
        print(f"✅ PASSED: {initial_balance:,.0f} + {add_amount:,.0f} = {actual:,.0f}")
        return True
    else:
        print(f"❌ FAILED: Expected {expected:,.0f}, got {actual:,.0f}")
        return False


def test_2_subtract_balance(account_id):
    """Test: Subtract money"""
    print("\n[TEST 2] Subtract Balance")
    print("-" * 40)
    
    # Ensure we have balance
    update_balance(account_id, 5000, "add")
    
    initial = get_balance(account_id)
    initial_balance = initial.get("balance", 0) if initial else 0
    subtract_amount = 500
    
    r = update_balance(account_id, subtract_amount, "subtract")
    
    if r.status_code != 200:
        print(f"❌ FAILED: status {r.status_code}")
        return False
    
    data = r.json()
    expected = initial_balance - subtract_amount
    actual = data.get("balance", 0)
    
    if abs(actual - expected) < 0.01:
        print(f"✅ PASSED: {initial_balance:,.0f} - {subtract_amount:,.0f} = {actual:,.0f}")
        return True
    else:
        print(f"❌ FAILED: Expected {expected:,.0f}, got {actual:,.0f}")
        return False


def test_3_set_balance(account_id):
    """Test: Set balance"""
    print("\n[TEST 3] Set Balance")
    print("-" * 40)
    
    set_amount = 10000
    r = update_balance(account_id, set_amount, "set")
    
    if r.status_code != 200:
        print(f"❌ FAILED: status {r.status_code}")
        return False
    
    data = r.json()
    actual = data.get("balance", 0)
    
    if actual == set_amount:
        print(f"✅ PASSED: balance = {actual:,.0f}")
        return True
    else:
        print(f"❌ FAILED: Expected {set_amount:,.0f}, got {actual:,.0f}")
        return False


def test_4_insufficient_balance(account_id):
    """Test: Subtract more than available - should fail"""
    print("\n[TEST 4] Insufficient Balance")
    print("-" * 40)
    
    # Set to small amount
    update_balance(account_id, 100, "set")
    
    # Try to subtract more
    r = update_balance(account_id, 500, "subtract")
    
    if r.status_code == 400:
        print(f"✅ PASSED: Rejected with 400 - {r.json().get('detail', '')[:50]}...")
        return True
    else:
        print(f"❌ FAILED: Expected 400, got {r.status_code}")
        return False


def test_5_invalid_operation(account_id):
    """Test: Invalid operation"""
    print("\n[TEST 5] Invalid Operation")
    print("-" * 40)
    
    r = update_balance(account_id, 100, "multiply")  # invalid
    
    if r.status_code in [400, 422]:
        print(f"✅ PASSED: Rejected with {r.status_code}")
        return True
    else:
        print(f"❌ FAILED: Expected 400/422, got {r.status_code}")
        return False


def test_6_negative_amount(account_id):
    """Test: Negative set amount"""
    print("\n[TEST 6] Negative Amount")
    print("-" * 40)
    
    r = update_balance(account_id, -1000, "set")
    
    if r.status_code == 400:
        print(f"✅ PASSED: Rejected negative amount")
        return True
    else:
        print(f"❌ FAILED: Expected 400, got {r.status_code}")
        return False


def test_7_cache_invalidation(account_id):
    """Test: Cache invalidated after update"""
    print("\n[TEST 7] Cache Invalidation")
    print("-" * 40)
    
    set_amount = 7777
    update_balance(account_id, set_amount, "set")
    
    # Immediately get balance
    balance = get_balance(account_id)
    actual = balance.get("balance", 0) if balance else 0
    
    if actual == set_amount:
        print(f"✅ PASSED: GET returns {actual:,.0f} immediately (cache cleared)")
        return True
    else:
        print(f"❌ FAILED: Expected {set_amount:,.0f}, got {actual:,.0f}")
        return False


def test_8_response_structure(account_id):
    """Test: Response has required fields"""
    print("\n[TEST 8] Response Structure")
    print("-" * 40)
    
    r = update_balance(account_id, 100, "add")
    
    if r.status_code != 200:
        print(f"❌ FAILED: status {r.status_code}")
        return False
    
    data = r.json()
    required = ["status", "account_id", "balance", "updated_at"]
    missing = [f for f in required if f not in data]
    
    if not missing:
        print(f"✅ PASSED: All fields present {list(data.keys())}")
        return True
    else:
        print(f"❌ FAILED: Missing fields: {missing}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def run_all_tests():
    print("=" * 60)
    print("BALANCE UPDATE API - TEST SUITE")
    print("=" * 60)
    
    # Find test account
    account_id = find_test_account()
    print(f"\nUsing test account: {account_id}")
    print(f"Base URL: {BASE_URL}")
    
    # Check server
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server status: {'OK' if r.status_code == 200 else r.status_code}")
    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("Make sure backend is running at http://localhost:8000")
        return
    
    # Run tests
    tests = [
        test_1_add_balance,
        test_2_subtract_balance,
        test_3_set_balance,
        test_4_insufficient_balance,
        test_5_invalid_operation,
        test_6_negative_amount,
        test_7_cache_invalidation,
        test_8_response_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test(account_id):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
