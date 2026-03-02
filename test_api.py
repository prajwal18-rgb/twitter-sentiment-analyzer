"""
Test API - Send Requests to the API
====================================
This script tests the API endpoints by sending HTTP requests.
Run this AFTER starting the API server.
"""

import requests
import json
import time


API_URL = "http://localhost:8000"


def test_health():
    """
    Test the health check endpoint.
    """
    print("\n" + "=" * 70)
    print("🔍 Testing /health endpoint...")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/health")
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            return True
        else:
            print("❌ Health check failed!")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API!")
        print("   Make sure the API server is running: python run_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_predict(text):
    """
    Test the prediction endpoint.
    """
    print("\n" + "=" * 70)
    print(f"🎯 Testing /predict endpoint with text:")
    print(f"   '{text}'")
    print("=" * 70)
    
    try:
        # Prepare request
        payload = {"text": text}
        
        # Send POST request
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 Prediction Results:")
            print(f"   Sentiment: {result['sentiment'].upper()}")
            print(f"   Confidence: {result['confidence']:.2f}%")
            print(f"\n   Probability Distribution:")
            for sentiment, prob in result['probabilities'].items():
                print(f"      {sentiment.capitalize()}: {prob:.2f}%")
            print("\n✅ Prediction successful!")
            return True
        else:
            print(f"\n❌ Prediction failed!")
            print(f"   Error: {response.json()}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API!")
        print("   Make sure the API server is running: python run_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_info():
    """
    Test the info endpoint.
    """
    print("\n" + "=" * 70)
    print("ℹ️  Testing /info endpoint...")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/info")
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Info endpoint works!")
            return True
        else:
            print("❌ Info endpoint failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """
    Run all API tests.
    """
    print("=" * 70)
    print("🧪 API TESTING SCRIPT")
    print("=" * 70)
    print("\nℹ️  Make sure the API server is running before testing!")
    print("   Start it with: python run_api.py\n")
    
    input("Press Enter to start testing...")
    
    # Test health endpoint
    health_ok = test_health()
    
    if not health_ok:
        print("\n❌ API is not running or not healthy. Stopping tests.")
        return
    
    time.sleep(1)
    
    # Test info endpoint
    test_info()
    
    time.sleep(1)
    
    # Test prediction with different texts
    test_cases = [
        "I love this product! It's absolutely amazing!",
        "This is terrible. Worst purchase ever!",
        "It's okay, nothing special.",
        "Best day ever! So happy right now!",
        "Very disappointed. Not worth the money."
    ]
    
    print("\n" + "=" * 70)
    print("🎯 TESTING PREDICTIONS WITH MULTIPLE EXAMPLES")
    print("=" * 70)
    
    for text in test_cases:
        test_predict(text)
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\n📚 Next Steps:")
    print("   1. Try the interactive API docs at: http://localhost:8000/")
    print("   2. Test with Postman or curl")
    print("   3. Move to Phase 2: Build the Streamlit frontend!")
    print("\n")


if __name__ == "__main__":
    main()
