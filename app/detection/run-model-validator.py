from app.detection.model_validator import ModelValidator
from datetime import datetim

# Create validator
validator = ModelValidator()

# Validate setup
success, errors = validator.validate_setup()
if not success:
    print("Model setup validation failed:")
    for error in errors:
        print(f"- {error}")

# Test with a sample image
test_image = "path/to/test/car/image.jpg"
success, errors, results = validator.validate_inference(test_image)
if success:
    print("Inference test passed!")
    print("Predictions:", results['predictions'])
else:
    print("Inference test failed:", errors)

# Generate full validation report
report = validator.generate_validation_report()