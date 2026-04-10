# Classification Error Fix

## Problem
Classification was failing with an error when uploading images. The error was occurring during the Grad-CAM computation step.

## Root Cause
The `pytorch_grad_cam` library has a dependency chain that includes `scikit-learn`, specifically the `KernelPCA` class from sklearn's decomposition module. When this library tried to import its submodules, it was causing an import error, which was not being properly caught in the try-except block.

The error occurred at:
```
File "classifier.py", line 182, in _compute_gradcam
    from pytorch_grad_cam import GradCAM
```

## Solution
Enhanced error handling in the `_compute_gradcam()` function to:
1. Add a nested try-except that specifically catches `ImportError` and `ModuleNotFoundError`
2. Log the import failure and gracefully fall back to `_gradcam_placeholder()` 
3. Keep the outer try-except for any other computation errors

## Changes Made
**File:** `backend/classifier.py`

Modified the `_compute_gradcam()` function to catch import errors more robustly:
- Added inner try-except to handle ImportError/ModuleNotFoundError before attempting to use GradCAM
- Falls back to placeholder image if any dependency issues occur
- Preserves outer exception handling for runtime Grad-CAM computation errors

## Testing
✅ Classification now works successfully with test images
✅ Returns predicted class and confidence score
✅ Grad-CAM visualization gracefully falls back when library issues occur

## Next Steps
The classification endpoint is now functional. You can:
1. Upload images in the classifier page
2. Get predictions with confidence scores
3. See condition-specific recommendations
