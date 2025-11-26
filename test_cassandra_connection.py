import requests
import json

print("Testing Backend → Cassandra Connection")
print("="*60)

try:
    # Test health endpoint
    print("\n1. Testing /health/detailed...")
    response = requests.get("http://localhost:8000/health/detailed", timeout=5)
    
    print(f"   Status Code: {response.status_code}")
    
    data = response.json()
    print(f"   Overall Status: {data.get('status')}")
    
    # Check Cassandra service
    cassandra = data.get('services', {}).get('cassandra', {})
    cassandra_status = cassandra.get('status', 'unknown')
    
    print(f"\n2. Cassandra Service:")
    print(f"   Status: {cassandra_status}")
    
    if cassandra_status == 'healthy':
        print(f"   ✅ CONNECTED!")
        print(f"   Keyspace: {cassandra.get('keyspace', 'N/A')}")
        print(f"   Contact Points: {cassandra.get('contact_points', [])}")
        print(f"\n🎉 Backend is connected to Cassandra!")
    else:
        print(f"   ❌ NOT CONNECTED")
        print(f"   Error: {cassandra.get('error', 'Unknown error')}")
        print(f"\n⚠️ Backend cannot connect to Cassandra.")
        print(f"   Solution: Restart backend with:")
        print(f"   Ctrl+C → python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    
    # Print full response for debugging
    print(f"\n3. Full Response:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
except requests.exceptions.ConnectionError:
    print("   ❌ Cannot connect to backend!")
    print("   Make sure backend is running on http://localhost:8000")
except requests.exceptions.Timeout:
    print("   ❌ Request timeout!")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
