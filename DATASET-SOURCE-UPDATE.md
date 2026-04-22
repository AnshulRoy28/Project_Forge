# Dataset Source Update

## Summary
Updated the CLI Neural Network Builder to support both custom datasets and standard PyTorch/torchvision datasets (like MNIST, CIFAR10, etc.).

## Changes Made

### 1. Updated Gemini Client (`nnb/gemini_brain/client.py`)
- **Migrated from deprecated `google-generativeai` to `google-genai`** package
- Updated to use `gemini-2.5-flash` model
- Fixed deprecation warnings

### 2. Updated Project Specification (`nnb/models/project_spec.py`)
- Added `dataset_source` field to `ProjectSpec` model
- Default value: `"custom"`
- Possible values: `"custom"`, `"torchvision"`, or specific dataset names like `"MNIST"`, `"CIFAR10"`

### 3. Updated Scoping Stage (`nnb/stages/stage_02_scoping.py`)
- Modified interviewer prompt to detect dataset source from user description
- Captures `dataset_source` from Gemini response
- Stores it in project specification

### 4. Updated Data Requirements Stage (`nnb/stages/stage_03_data_requirements.py`)
- Modified prompt to handle both custom and torchvision datasets
- For torchvision datasets:
  - Confirms auto-download capability
  - Provides example code for loading
  - Skips manual validation requirement
- For custom datasets:
  - Provides full folder structure requirements
  - Requires manual validation with `nnb data validate`
- Updated next step instructions based on dataset source

### 5. Updated Data Validation (`nnb/cli.py`, `nnb/orchestrator/project.py`)
- **CLI command now detects torchvision datasets automatically**
- For torchvision datasets:
  - `nnb data validate` (no --path needed) confirms dataset will be auto-downloaded
  - Automatically transitions to DATA_VALIDATED state
  - Tells user to run `nnb env build` next
- For custom datasets:
  - `nnb data validate --path /your/data` performs full validation
  - Requires --path parameter

### 6. Updated Dependencies (`requirements.txt`)
- Changed from `google-generativeai>=0.3.1` to `google-genai>=0.1.0`

## Usage

### For Standard Datasets (MNIST, CIFAR10, etc.)
When you describe your project and mention using a standard dataset like MNIST from torchvision:

```bash
nnb start
# Answer questions mentioning "MNIST from torchvision"
# System will detect dataset_source as "torchvision" or "MNIST"

# Option 1: Skip validation entirely
nnb env build

# Option 2: Run validation (it will auto-skip)
nnb data validate
# Output: "✓ Using torchvision dataset: MNIST"
#         "Dataset will be automatically downloaded during training"
#         "No validation needed!"
```

### For Custom Datasets
When using your own data:

```bash
nnb start
# Answer questions about your custom data
# System will detect dataset_source as "custom"

# Must provide --path for validation
nnb data validate --path /path/to/your/data
nnb env build
```

## How It Works

1. **During scoping**: Gemini detects if you mention "MNIST", "torchvision", etc. and sets `dataset_source`
2. **During data requirements**: Different instructions generated based on `dataset_source`
3. **During validation**:
   - System checks `project._spec.dataset_source`
   - If torchvision dataset: Auto-passes validation, no path needed
   - If custom dataset: Requires --path and performs full validation

## Testing
- All 15 tests passing
- No diagnostic errors
- Backward compatible with existing projects

## Next Steps
When implementing stages 5-8, the code generation stage will need to:
1. Check `project._spec.dataset_source`
2. Generate appropriate data loading code:
   - For torchvision: Use `torchvision.datasets.MNIST(root='/data', download=True, ...)`
   - For custom: Use `ImageFolder` or custom data loaders with `/data` mount point
