"""
Forge Trainer - Training engine optimized for RTX 50-series (Blackwell).

Optimizations:
- BF16 native precision (Blackwell excels at this)
- TF32 for Tensor Core speedup
- Unsloth gradient checkpointing ("unsloth" mode)
- adamw_8bit optimizer for VRAM savings
- FP8 support when available
"""

from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass

from forge.core.config import ForgeConfig


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""
    step: int
    loss: float
    learning_rate: float
    epoch: float


class ForgeTrainer:
    """
    Training engine for Forge.
    
    Optimized for RTX 50-series (Blackwell) with:
    - BF16 precision (native to 5080)
    - TF32 Tensor Core acceleration
    - Unsloth 2x speedup + 70% VRAM savings
    """
    
    def __init__(self, config: ForgeConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Enable Blackwell optimizations
        self._setup_blackwell_optimizations()
        self._load_model()
    
    def _setup_blackwell_optimizations(self):
        """Configure PyTorch for optimal RTX 50-series performance."""
        import torch
        
        # Enable TF32 for 3x speedup on FP32 math using Tensor Cores
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set optimal CUDA settings
        if torch.cuda.is_available():
            # Enable flash attention if available
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    def _load_model(self):
        """Load the base model with Unsloth optimizations."""
        training = self.config.training
        model_name = training.base_model
        
        # Check if model is pre-quantized
        is_prequantized = "bnb-4bit" in model_name.lower() or "bnb-8bit" in model_name.lower()
        
        try:
            # Use Unsloth for 2x speed + 70% VRAM savings
            from unsloth import FastLanguageModel
            import torch
            
            # Load model - let Unsloth handle dtype for best compatibility
            if is_prequantized:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=training.max_seq_length,
                    dtype=None,  # Auto-detect best dtype
                    load_in_4bit=True,
                )
            else:
                load_in_4bit = training.quantization == "4bit"
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=training.max_seq_length,
                    load_in_4bit=load_in_4bit,
                    dtype=None,
                )
            
            # Apply LoRA with Unsloth's optimized gradient checkpointing
            lora = training.lora
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=lora.rank,
                lora_alpha=lora.alpha,
                lora_dropout=lora.dropout,
                target_modules=lora.target_modules,
                # Use Unsloth's special gradient checkpointing for massive VRAM savings
                use_gradient_checkpointing="unsloth" if training.use_gradient_checkpointing else False,
                random_state=42,
                bias="none",
            )
            
            self._using_unsloth = True
            
        except ImportError:
            # Fallback to standard transformers + PEFT
            self._load_model_transformers()
            self._using_unsloth = False
    
    def _load_model_transformers(self):
        """Load model using transformers + PEFT (fallback)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        training = self.config.training
        
        # 4-bit quantization config optimized for 16GB VRAM
        bnb_config = None
        if training.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # BF16 for Blackwell
                bnb_4bit_use_double_quant=True,
            )
        elif training.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(training.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with BF16 for Blackwell
        self.model = AutoModelForCausalLM.from_pretrained(
            training.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Native to RTX 5080
        )
        
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        lora = training.lora
        lora_config = LoraConfig(
            r=lora.rank,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        if training.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def train(
        self,
        dataset: Any,
        callback: Optional[Callable[[int, float, float], None]] = None,
        resume_from: Optional[Path] = None,
    ):
        """
        Run the training loop with Blackwell optimizations.
        """
        from trl import SFTTrainer
        
        training = self.config.training
        
        # Create optimized training config
        try:
            from trl import SFTConfig
            
            sft_config = SFTConfig(
                output_dir=self.config.output.checkpoint_dir,
                num_train_epochs=training.num_epochs,
                
                # Batch settings for 16GB VRAM
                per_device_train_batch_size=training.batch_size,
                gradient_accumulation_steps=training.gradient_accumulation_steps,
                
                # Learning rate
                learning_rate=training.learning_rate,
                weight_decay=training.weight_decay,
                warmup_ratio=training.warmup_ratio,
                lr_scheduler_type=training.lr_scheduler,
                
                # Optimizer: adamw_8bit saves ~2GB VRAM
                optim=training.optim,
                
                # Logging and checkpoints
                logging_steps=training.logging_steps,
                save_steps=training.save_steps,
                save_total_limit=3,
                
                # Precision: BF16 is native to RTX 5080 Blackwell
                # Prevents NaN errors that FP16 can cause
                fp16=False,
                bf16=True,  # Native to Blackwell - 3x speedup
                
                # Disable reporting
                report_to="none",
                
                # Sequence settings
                max_length=training.max_seq_length,
                dataset_text_field="text",
                packing=False,
                
                # Speed optimizations
                dataloader_pin_memory=True,
                dataloader_num_workers=0,
                
                # Gradient checkpointing handled by Unsloth
                gradient_checkpointing=training.use_gradient_checkpointing,
            )
            
            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=dataset,
                processing_class=self.tokenizer,
                args=sft_config,
            )
            
        except ImportError:
            # Fallback for older TRL
            from transformers import TrainingArguments, DataCollatorForLanguageModeling
            
            training_args = TrainingArguments(
                output_dir=self.config.output.checkpoint_dir,
                num_train_epochs=training.num_epochs,
                per_device_train_batch_size=training.batch_size,
                gradient_accumulation_steps=training.gradient_accumulation_steps,
                learning_rate=training.learning_rate,
                weight_decay=training.weight_decay,
                warmup_ratio=training.warmup_ratio,
                lr_scheduler_type=training.lr_scheduler,
                optim=training.optim,
                logging_steps=training.logging_steps,
                save_steps=training.save_steps,
                save_total_limit=3,
                fp16=False,
                bf16=True,
                report_to="none",
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                args=training_args,
                data_collator=data_collator,
                max_seq_length=training.max_seq_length,
                dataset_text_field="text",
                packing=False,
            )
        
        # Progress callback
        if callback:
            from transformers import TrainerCallback
            
            class ForgeCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and "loss" in logs:
                        callback(
                            state.global_step,
                            logs.get("loss", 0),
                            logs.get("learning_rate", 0),
                        )
            
            self.trainer.add_callback(ForgeCallback())
        
        # Start training
        self.trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    
    def save(self, output_path: Path):
        """Save the trained LoRA adapter."""
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
    
    def merge_and_save(self, output_path: Path):
        """Merge LoRA weights and save full model."""
        import json
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get model_type from base model config before merging
        base_model_type = None
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'model_type'):
            base_model_type = self.model.config.model_type
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'config'):
            base_model_type = getattr(self.model.base_model.config, 'model_type', None)
        
        if self._using_unsloth:
            from unsloth import FastLanguageModel
            self.model.save_pretrained_merged(
                output_path,
                self.tokenizer,
                save_method="merged_16bit",
            )
        else:
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)
        
        # Ensure config.json has model_type (required by transformers)
        config_path = output_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'model_type' not in config and base_model_type:
                config['model_type'] = base_model_type
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
    
    def export_gguf(self, output_path: Path, quantization: str = "q4_k_m"):
        """
        Export model to GGUF format for Ollama/llama.cpp.
        
        Args:
            output_path: Directory to save GGUF file
            quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16)
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self._using_unsloth:
            from unsloth import FastLanguageModel
            
            # Unsloth has built-in GGUF export
            gguf_path = output_path / f"model-{quantization}.gguf"
            self.model.save_pretrained_gguf(
                str(output_path),
                self.tokenizer,
                quantization_method=quantization,
            )
            return gguf_path
        else:
            # Fallback: save merged model and instruct user to convert
            merged_path = output_path / "merged"
            self.merge_and_save(merged_path)
            
            # Create conversion instructions
            instructions = output_path / "CONVERT_TO_GGUF.md"
            with open(instructions, 'w') as f:
                f.write(f"""# Convert to GGUF

Your model has been saved to `{merged_path}`.

To convert to GGUF format, use llama.cpp's convert script:

```bash
# Clone llama.cpp if you haven't
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install requirements
pip install -r requirements.txt

# Convert to GGUF
python convert.py {merged_path} --outfile {output_path}/model.gguf --outtype {quantization}
```
""")
            return merged_path
    
    def register_ollama(self, output_path: Path, model_name: str, system_prompt: str = None):
        """
        Register a GGUF model with Ollama.
        
        Args:
            output_path: Path to GGUF file or directory containing it
            model_name: Name to register the model as in Ollama
            system_prompt: Optional system prompt to bake into the model
        """
        import subprocess
        
        # Find GGUF file
        if output_path.is_file() and output_path.suffix == '.gguf':
            gguf_file = output_path
        else:
            gguf_files = list(output_path.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF file found in {output_path}")
            gguf_file = gguf_files[0]
        
        # Create Modelfile
        modelfile_path = output_path / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(f"FROM {gguf_file.absolute()}\n")
            if system_prompt:
                f.write(f'\nSYSTEM """{system_prompt}"""\n')
            
            # Add reasonable defaults
            f.write("\nPARAMETER temperature 0.7\n")
            f.write("PARAMETER top_p 0.9\n")
            f.write("PARAMETER stop \"<|im_end|>\"\n")
            f.write("PARAMETER stop \"</s>\"\n")
        
        # Register with Ollama
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to register with Ollama: {result.stderr}")
        
        return model_name

