# Model-Aware Preprocessing Pipeline

## Overview

Transform Forge's preprocessing workflow to be model-aware, hardware-optimized, and follow ML engineering best practices. The new workflow integrates `forge plan` and `forge prepare` commands to create a seamless, intelligent preprocessing pipeline that adapts to the target model and hardware configuration.

## Problem Statement

The current `forge prepare` command has several limitations:
- Hardcoded ChatML format incompatible with many models (Llama, Gemma, etc.)
- No awareness of target model or hardware constraints
- Heavy venv sandbox instead of efficient Docker containers
- Fixed preprocessing pipeline without ML engineering best practices
- No data quality validation or optimization features

## User Stories

### 1. Model-Aware Workflow
**As a** ML practitioner  
**I want** the preprocessing to automatically adapt to my target model  
**So that** my data is formatted correctly for the specific model I'm training

**Acceptance Criteria:**
- `forge plan` must run before `forge prepare` to establish model context
- `forge prepare` reads plan configuration to determine target model
- Preprocessing uses correct chat template for the selected model
- Data format matches model's expected input structure

### 2. Hardware-Optimized Processing
**As a** user with specific GPU hardware  
**I want** preprocessing optimized for my hardware capabilities  
**So that** I can efficiently process large datasets within my memory constraints

**Acceptance Criteria:**
- Preprocessing adapts batch sizes based on available VRAM
- Memory-efficient streaming for datasets larger than RAM
- GPU-accelerated preprocessing when beneficial
- Hardware-aware chunking and parallel processing

### 3. Docker-Based Sandbox
**As a** developer  
**I want** preprocessing to run in the same Docker environment as training  
**So that** I have consistent, reproducible preprocessing without dependency conflicts

**Acceptance Criteria:**
- Preprocessing runs in GPU-optimized Docker containers
- Uses same container architecture as training (blackwell, ada, ampere, hopper)
- No venv creation or management required
- Faster execution than current venv approach

### 4. ML Engineering Best Practices
**As a** ML engineer  
**I want** professional-grade data processing features  
**So that** I can ensure data quality and optimize training performance

**Acceptance Criteria:**
- Data quality validation and profiling
- Stratified train/validation splits with configurable ratios
- Dataset statistics and distribution analysis
- Checkpoint/resume capability for large datasets
- Output format validation and quality checks

### 5. Flexible Chat Templates
**As a** user training different model types  
**I want** support for multiple chat formats  
**So that** I can work with any model architecture

**Acceptance Criteria:**
- Support for ChatML, Llama, Alpaca, Gemma, and custom formats
- Template selection based on target model from plan
- Configurable template parameters
- Validation of template compatibility with model

## Technical Requirements

### 1. Plan-Prepare Integration

#### 1.1 Configuration Sharing
- `forge plan` creates `forge.yaml` with model and hardware configuration
- `forge prepare` reads `forge.yaml` to determine preprocessing parameters
- Shared configuration includes:
  - Target model name and architecture
  - Chat template format
  - Hardware constraints (VRAM, compute capability)
  - Batch size and memory optimizations

#### 1.2 Workflow Validation
- `forge prepare` checks for existing `forge.yaml`
- If no plan exists, prompt user to run `forge plan` first
- Validate plan compatibility with dataset characteristics

### 2. Docker Sandbox Implementation

#### 2.1 Container Selection
- Use same GPU-optimized containers as training
- Auto-detect architecture (blackwell, ada, ampere, hopper)
- Mount data directories and configuration files
- Pass environment variables for API keys

#### 2.2 Execution Environment
- Run preprocessing scripts inside Docker container
- Stream output and progress to host terminal
- Handle container lifecycle (start, execute, cleanup)
- Error handling and container restart on failures

### 3. Model-Aware Templates

#### 3.1 Template System
```python
CHAT_TEMPLATES = {
    "chatml": "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>",
    "llama": "<s>[INST] {query} [/INST] {response} </s>",
    "alpaca": "### Instruction:\n{query}\n\n### Response:\n{response}",
    "gemma": "<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>",
    "custom": "{custom_template}"
}
```

#### 3.2 Model Detection
- Map model names to appropriate templates
- Support for custom template definitions
- Template validation and testing

### 4. ML Engineering Features

#### 4.1 Data Quality Validation
- Missing value detection and handling
- Duplicate detection and removal
- Text quality metrics (length, encoding, language)
- Distribution analysis and outlier detection

#### 4.2 Advanced Splitting
- Stratified splits based on data characteristics
- Configurable train/validation/test ratios
- Temporal splits for time-series data
- Balanced sampling for imbalanced datasets

#### 4.3 Memory Management
- Streaming processing for large datasets
- Configurable chunk sizes based on available memory
- Progress tracking and ETA estimation
- Checkpoint saving for resumable processing

#### 4.4 Output Validation
- JSONL format validation
- Template application verification
- Sample quality checks
- Statistics reporting

### 5. Hardware Optimization

#### 5.1 Memory-Aware Processing
- Dynamic batch sizing based on available VRAM
- Streaming for datasets exceeding RAM capacity
- GPU acceleration for text processing when beneficial
- Parallel processing optimization

#### 5.2 Performance Monitoring
- Processing speed metrics
- Memory usage tracking
- GPU utilization monitoring
- Bottleneck identification

## Implementation Architecture

### 1. Command Flow
```
forge plan "goal" → forge.yaml created
forge prepare data.csv → reads forge.yaml → Docker preprocessing
```

### 2. Configuration Schema
```yaml
# forge.yaml
model:
  name: "microsoft/DialoGPT-medium"
  architecture: "gpt2"
  chat_template: "chatml"
  max_length: 2048

hardware:
  gpu_arch: "ada"
  vram_gb: 16
  batch_size: 4
  use_gpu_preprocessing: true

preprocessing:
  train_split: 0.9
  validation_split: 0.1
  chunk_size: 1000
  quality_checks: true
```

### 3. Docker Integration
- Extend existing Docker containers with preprocessing tools
- Mount configuration and data directories
- Stream logs and progress to host
- Handle GPU passthrough for accelerated processing

## Success Metrics

1. **Compatibility**: Support for 5+ major model architectures
2. **Performance**: 50% faster preprocessing than current venv approach
3. **Memory Efficiency**: Process datasets 2x larger than available RAM
4. **Quality**: 95% reduction in preprocessing-related training failures
5. **Usability**: Single command workflow with automatic configuration

## Dependencies

- Existing Docker infrastructure
- `forge plan` command functionality
- Gemini AI integration for script generation
- Hardware detection capabilities
- Configuration management system

## Risks and Mitigations

1. **Docker Complexity**: Mitigate with comprehensive error handling and fallback options
2. **Template Compatibility**: Extensive testing with popular models
3. **Memory Management**: Conservative defaults with user override options
4. **Performance Regression**: Benchmarking and optimization during development

## Future Enhancements

- Multi-modal data support (images, audio)
- Distributed preprocessing across multiple GPUs
- Integration with popular ML frameworks (Weights & Biases, MLflow)
- Advanced data augmentation techniques
- Real-time preprocessing monitoring dashboard