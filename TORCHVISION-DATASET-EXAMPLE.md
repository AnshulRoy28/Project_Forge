# Using Torchvision Datasets - Quick Example

## How to Use MNIST (or any torchvision dataset)

### Step 1: Start a new project
```bash
nnb start
```

### Step 2: Answer the conversation questions
When asked about your project, mention the torchvision dataset:

**Example answers:**
- "I want to build a simple neural network to classify MNIST handwritten digits"
- "I plan on using the standard public distribution provided by torchvision"
- "Just a basic feedforward network with 2 hidden layers"

The system will automatically detect `dataset_source: "MNIST"` or `"torchvision"`

### Step 3: Skip or run validation (both work!)

**Option A - Skip validation entirely:**
```bash
nnb env build
```

**Option B - Run validation (it auto-skips):**
```bash
nnb data validate
```

Output:
```
✓ Using torchvision dataset: MNIST
  Dataset will be automatically downloaded during training
  No validation needed!

Next step:
  Run: nnb env build
```

### Step 4: Continue with environment setup
```bash
nnb env build
```

## Supported Torchvision Datasets

The system automatically recognizes these dataset names:
- `MNIST`
- `CIFAR10`
- `CIFAR100`
- `FashionMNIST`
- `ImageNet`
- Or just say `"torchvision"` for any torchvision dataset

## What Happens Behind the Scenes

1. **Scoping Stage**: Gemini detects "MNIST from torchvision" in your answers
2. **Spec Creation**: Sets `dataset_source: "MNIST"` in project spec
3. **Data Requirements**: Generates instructions for auto-downloading
4. **Validation**: Auto-passes without needing a data path
5. **Code Generation** (Stage 6): Will generate code like:
   ```python
   from torchvision import datasets, transforms
   
   train_dataset = datasets.MNIST(
       root='/data',
       train=True,
       download=True,
       transform=transforms.ToTensor()
   )
   ```

## Key Benefits

✅ No need to manually download datasets  
✅ No need to organize folder structures  
✅ No need to run data validation  
✅ Dataset automatically downloaded in Docker container  
✅ Standard preprocessing applied automatically  

## For Custom Datasets

If you want to use your own data instead:
1. Organize your data in the required folder structure
2. Run `nnb data validate --path /your/data/path`
3. System will validate format and structure
