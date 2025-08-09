"""Script to test the FastAPI endpoints."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import requests
from PIL import Image
import numpy as np


class APITester:
    """Class for testing the API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
    
    def test_health(self) -> dict[str, any]:
        """Test the health endpoint."""
        print("Testing health endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Health check successful: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {"error": str(e)}
    
    def test_model_info(self) -> dict[str, any]:
        """Test the model info endpoint."""
        print("Testing model info endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/model_info")
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Model info retrieved: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Model info failed: {e}")
            return {"error": str(e)}
    
    def create_test_image(self, size: tuple = (512, 512)) -> bytes:
        """Create a test image for testing.
        
        Args:
            size: Image size (width, height)
            
        Returns:
            Image bytes
        """
        # Create a random RGB image
        image_array = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        return img_buffer.getvalue()
    
    def test_single_prediction(self, image_path: str | None = None) -> dict[str, any]:
        """Test single image prediction.
        
        Args:
            image_path: Path to test image (optional)
            
        Returns:
            Prediction result
        """
        print("Testing single prediction endpoint...")
        
        try:
            # Use provided image or create test image
            if image_path and Path(image_path).exists():
                with open(image_path, "rb") as f:
                    image_data = f.read()
                filename = Path(image_path).name
            else:
                image_data = self.create_test_image()
                filename = "test_image.png"
            
            # Prepare request
            files = {"file": (filename, image_data, "image/png")}
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/predict", files=files)
            request_time = time.time() - start_time
            
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Single prediction successful (took {request_time:.3f}s):")
            print(json.dumps(result, indent=2))
            return result
            
        except Exception as e:
            print(f"❌ Single prediction failed: {e}")
            return {"error": str(e)}
    
    def test_batch_prediction(self, image_paths: list[str] | None = None, num_images: int = 3) -> dict[str, any]:
        """Test batch prediction.
        
        Args:
            image_paths: List of image paths (optional)
            num_images: Number of test images to create if no paths provided
            
        Returns:
            Batch prediction result
        """
        print(f"Testing batch prediction endpoint with {num_images} images...")
        
        try:
            files = []
            
            # Use provided images or create test images
            if image_paths:
                for i, image_path in enumerate(image_paths[:num_images]):
                    if Path(image_path).exists():
                        with open(image_path, "rb") as f:
                            image_data = f.read()
                        filename = Path(image_path).name
                        files.append(("files", (filename, image_data, "image/png")))
            else:
                for i in range(num_images):
                    image_data = self.create_test_image()
                    filename = f"test_image_{i}.png"
                    files.append(("files", (filename, image_data, "image/png")))
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/predict_batch", files=files)
            request_time = time.time() - start_time
            
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Batch prediction successful (took {request_time:.3f}s):")
            print(json.dumps(result, indent=2))
            return result
            
        except Exception as e:
            print(f"❌ Batch prediction failed: {e}")
            return {"error": str(e)}
    
    def run_all_tests(self, image_path: str | None = None, image_paths: list[str] | None = None) -> dict[str, dict[str, any]]:
        """Run all available tests.
        
        Args:
            image_path: Path to test image for single prediction
            image_paths: List of image paths for batch prediction
            
        Returns:
            Dictionary containing all test results
        """
        print("=" * 50)
        print("Running API Tests")
        print("=" * 50)
        
        results = {}
        
        # Test health endpoint
        results["health"] = self.test_health()
        print()
        
        # Test model info endpoint
        results["model_info"] = self.test_model_info()
        print()
        
        # Test single prediction
        results["single_prediction"] = self.test_single_prediction(image_path)
        print()
        
        # Test batch prediction
        results["batch_prediction"] = self.test_batch_prediction(image_paths)
        print()
        
        print("=" * 50)
        print("Test Summary")
        print("=" * 50)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if "error" not in result else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        return results


def main():
    """Main function to run API tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the FastAPI endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--image-path", help="Path to test image for single prediction")
    parser.add_argument("--image-paths", nargs="+", help="Paths to test images for batch prediction")
    parser.add_argument("--test", choices=["health", "model_info", "single", "batch", "all"], 
                       default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = APITester(args.base_url)
    
    # Run specific test
    if args.test == "health":
        tester.test_health()
    elif args.test == "model_info":
        tester.test_model_info()
    elif args.test == "single":
        tester.test_single_prediction(args.image_path)
    elif args.test == "batch":
        tester.test_batch_prediction(args.image_paths)
    else:  # all
        tester.run_all_tests(args.image_path, args.image_paths)


if __name__ == "__main__":
    main()
