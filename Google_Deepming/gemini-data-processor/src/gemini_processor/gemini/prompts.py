"""Prompt templates for Gemini AI interactions."""

from typing import Any, Dict, List

from ..models.data import DataAnalysis, DataSnapshot, ProcessingContext


class PromptManager:
    """Manages prompt templates for Gemini AI."""
    
    DATA_ANALYSIS_PROMPT = """You are a data analysis expert. Analyze the following data sample and provide insights.

## Data Information
- **File Format**: {file_format}
- **Total Rows**: {total_rows:,}
- **Sample Size**: {sample_size}
- **Columns**: {num_columns}

## Schema
{schema_text}

## Sample Data (first 5 rows)
{sample_data}

## Your Task
Analyze this data and provide:
1. **Data Quality Issues**: List any problems like missing values, inconsistent formats, duplicates, etc.
2. **Suggested Operations**: What processing steps would improve this data?
3. **Column Insights**: Brief insight about each column's content and purpose.
4. **Processing Recommendations**: Specific recommendations for cleaning/transforming this data.
5. **Complexity Estimate**: Rate the complexity of processing as "low", "medium", or "high".

Respond in JSON format with the following structure:
{{
    "data_quality_issues": ["issue1", "issue2"],
    "suggested_operations": ["operation1", "operation2"],
    "column_insights": {{"column_name": "insight"}},
    "processing_recommendations": ["recommendation1", "recommendation2"],
    "estimated_complexity": "low|medium|high",
    "sensitive_data_detected": true|false
}}
"""

    SCRIPT_GENERATION_PROMPT = """You are a Python data processing expert. Generate a script to process data.

## Context
{context}

## Data Schema
{schema_text}

## Required Operation
{operation}

## Previous Processing Steps
{previous_steps}

## Guidelines
1. The input file is available at the path in environment variable INPUT_FILE
2. Write output to the directory in environment variable OUTPUT_DIR
3. Use pandas for data manipulation
4. Include proper error handling
5. Print progress messages to stdout
6. The script will run in an isolated Docker container

Generate a complete, runnable Python script that:
- Reads the input data
- Performs the requested operation
- Saves the processed data to the output directory
- Prints a summary of the changes made

Respond with ONLY the Python code, no explanations or markdown.
"""

    # Common JSON safety guidelines for all EDA scripts
    JSON_GUIDELINES = """
## CRITICAL: JSON Serialization
When saving JSON files, you MUST:
1. Replace NaN/Infinity with None before serializing: use `.replace([np.inf, -np.inf], np.nan).fillna(None)` or convert to Python native types
2. Convert numpy types to Python types: use `.item()` or `int()`, `float()`
3. Use `json.dump(data, f, indent=4, default=str)` to handle any remaining edge cases
4. Always close files properly or use `with open()` context manager
5. Handle DataFrames: convert to dict with `.to_dict()` before JSON serialization
"""

    # Modular EDA script prompts - each generates a focused script
    MODULAR_EDA_PROMPTS = {
        "missing_values": """Generate a Python script that analyzes MISSING VALUES in the dataset.

## Data Info
- File: {file_format} format, ~{total_rows:,} rows, {num_columns} columns
- Schema: {schema_text}

## Script Requirements
1. Count missing values per column
2. Calculate percentage of missing values
3. Generate a heatmap visualization of missing patterns
4. Save results to OUTPUT_DIR/missing_values_report.json
5. Save heatmap to OUTPUT_DIR/missing_values_heatmap.png

## CRITICAL: JSON Serialization
- Replace NaN/Infinity with None before JSON: use `float()` conversion and check for `math.isnan()`
- Use `json.dump(data, f, indent=4, default=str)` with default=str for safety
- Always use `with open(...) as f` context manager to ensure file is properly closed

## Guidelines
- Input: os.environ.get("INPUT_FILE")
- Output: os.environ.get("OUTPUT_DIR")
- Use pandas, matplotlib, seaborn
- Print summary findings to stdout

Respond with ONLY Python code, no markdown.""",

        "descriptive_stats": """Generate a Python script that computes DESCRIPTIVE STATISTICS for the dataset.

## Data Info
- File: {file_format} format, ~{total_rows:,} rows, {num_columns} columns
- Schema: {schema_text}

## Script Requirements
1. For numerical columns: mean, std, min, max, quartiles, skewness, kurtosis
2. For categorical columns: value counts, cardinality, mode
3. Generate distribution histograms for numerical columns
4. Save results to OUTPUT_DIR/descriptive_stats.json
5. Save plots to OUTPUT_DIR/

## CRITICAL: JSON Serialization
- Convert numpy types to Python: use `float(value)`, `int(value)` before adding to dict
- Handle NaN: check with `pd.isna(value)` and replace with None
- Handle Infinity: check with `np.isinf(value)` and replace with None
- Use `json.dump(data, f, indent=4, default=str)` for safety
- Always use `with open(...) as f` context manager

## Guidelines
- Input: os.environ.get("INPUT_FILE")
- Output: os.environ.get("OUTPUT_DIR")
- Use pandas, matplotlib, seaborn
- Print key statistics to stdout

Respond with ONLY Python code, no markdown.""",

        "outlier_detection": """Generate a Python script that detects OUTLIERS in numerical columns.

## Data Info
- File: {file_format} format, ~{total_rows:,} rows, {num_columns} columns
- Schema: {schema_text}

## Script Requirements
1. Use IQR method to detect outliers
2. Generate box plots for each numerical column
3. Report count and percentage of outliers per column
4. Save results to OUTPUT_DIR/outliers_report.json
5. Save box plots to OUTPUT_DIR/

## CRITICAL: JSON Serialization
- Convert all numbers to Python native types: `int(count)`, `float(percentage)`
- Use `json.dump(data, f, indent=4, default=str)` for safety
- Always use `with open(...) as f` context manager

## Guidelines
- Input: os.environ.get("INPUT_FILE")
- Output: os.environ.get("OUTPUT_DIR")
- Use pandas, matplotlib
- Print outlier summary to stdout

Respond with ONLY Python code, no markdown.""",

        "correlation_analysis": """Generate a Python script that performs CORRELATION ANALYSIS.

## Data Info
- File: {file_format} format, ~{total_rows:,} rows, {num_columns} columns
- Schema: {schema_text}

## Script Requirements
1. Compute correlation matrix for numerical columns
2. Identify highly correlated pairs (|r| > 0.7)
3. Generate correlation heatmap
4. Save results to OUTPUT_DIR/correlation_report.json
5. Save heatmap to OUTPUT_DIR/correlation_heatmap.png

## CRITICAL: JSON Serialization
- Correlation values can be NaN if column has no variance - replace with None
- Convert numpy floats to Python: `float(corr_value)` if not pd.isna()
- Use `json.dump(data, f, indent=4, default=str)` for safety
- Always use `with open(...) as f` context manager

## Guidelines
- Input: os.environ.get("INPUT_FILE")
- Output: os.environ.get("OUTPUT_DIR")
- Use pandas, matplotlib, seaborn
- Print key correlations to stdout

Respond with ONLY Python code, no markdown.""",

        "duplicate_detection": """Generate a Python script that detects DUPLICATE ROWS.

## Data Info
- File: {file_format} format, ~{total_rows:,} rows, {num_columns} columns
- Schema: {schema_text}

## Script Requirements
1. Identify exact duplicate rows
2. Identify near-duplicates (if applicable)
3. Report count and percentage of duplicates
4. Save duplicate rows to OUTPUT_DIR/duplicates.csv
5. Save summary to OUTPUT_DIR/duplicates_report.json

## CRITICAL: JSON Serialization
- Convert counts to Python int: `int(count)`
- Convert percentages to Python float: `float(percentage)`
- Use `json.dump(data, f, indent=4, default=str)` for safety
- Always use `with open(...) as f` context manager

## Guidelines
- Input: os.environ.get("INPUT_FILE")
- Output: os.environ.get("OUTPUT_DIR")
- Use pandas
- Print duplicate summary to stdout

Respond with ONLY Python code, no markdown.""",

        "data_quality": """Generate a Python script that creates a DATA QUALITY REPORT.

## Data Info
- File: {file_format} format, ~{total_rows:,} rows, {num_columns} columns
- Schema: {schema_text}

## Script Requirements
1. Check data type consistency
2. Detect mixed-type columns
3. Check for empty strings vs nulls
4. Validate value ranges for numerical columns
5. Save comprehensive report to OUTPUT_DIR/data_quality_report.json

## CRITICAL: JSON Serialization
- Convert all numpy types to Python native types before JSON
- Handle NaN in numerical ranges: replace with None
- Convert int64/float64 to int/float using `.item()` or built-in converters
- Use `json.dump(data, f, indent=4, default=str)` for safety
- Always use `with open(...) as f` context manager

## Guidelines
- Input: os.environ.get("INPUT_FILE")
- Output: os.environ.get("OUTPUT_DIR")
- Use pandas
- Print quality issues to stdout

Respond with ONLY Python code, no markdown.""",

        "save_processed_data": """Generate a Python script that SAVES THE PROCESSED DATA to the output directory.

## Data Info
- File: {file_format} format, ~{total_rows:,} rows, {num_columns} columns
- Schema: {schema_text}

## Script Requirements
1. Read the input data
2. Apply any basic cleaning (remove duplicates if any, strip whitespace from string columns)
3. Save the data to OUTPUT_DIR/processed_data.csv
4. Print summary of the saved data (rows, columns)

## Guidelines
- Input: os.environ.get("INPUT_FILE")
- Output: os.environ.get("OUTPUT_DIR")
- Use pandas
- Print confirmation message with row/column counts

Respond with ONLY Python code, no markdown.""",
    }

    # List of modular EDA steps to generate
    EDA_STEPS = [
        ("missing_values", "Missing Values Analysis"),
        ("descriptive_stats", "Descriptive Statistics"),
        ("outlier_detection", "Outlier Detection"),
        ("correlation_analysis", "Correlation Analysis"),
        ("duplicate_detection", "Duplicate Detection"),
        ("data_quality", "Data Quality Check"),
        ("save_processed_data", "Save Processed Data"),
    ]

    # Prompt for validating script outputs
    OUTPUT_VALIDATION_PROMPT = """You are a data validation expert. Check if the following script output is valid and complete.

## Script Description
{script_description}

## Expected Output Files
{expected_files}

## Actual Output Files Found
{actual_files}

## File Contents to Validate
{file_contents}

## Script's Console Output
{console_output}

## Validation Checks
1. Are all expected JSON files valid (parseable, complete, not truncated)?
2. Are numerical values reasonable (no unexpected NaN, Infinity as strings)?
3. Are all expected files present?
4. Does the console output indicate errors?

Respond in JSON format:
{{
    "is_valid": true|false,
    "issues": ["issue1", "issue2"],
    "suggestions": ["fix1", "fix2"],
    "severity": "none|minor|major|critical"
}}
"""

    # Prompt for self-healing a failed script
    SELF_HEAL_PROMPT = """You are a Python debugging expert. Fix this script that failed to produce valid output.

## Original Script Description
{script_description}

## Original Script
```python
{original_script}
```

## Validation Issues Found
{issues}

## Console Output (including errors)
{console_output}

## File Contents (if any, may be truncated/invalid)
{file_contents}

## Your Task
Fix the script to:
1. Address all validation issues
2. Ensure proper JSON serialization (handle NaN, Infinity, numpy types)
3. Use `json.dump(data, f, indent=4, default=str)` for safety
4. Use `with open(...) as f` context manager
5. Add try-except around JSON writing to catch and log errors

Generate the COMPLETE fixed Python script.
Respond with ONLY the fixed Python code, no markdown or explanations.
"""

    @classmethod
    def format_data_analysis_prompt(cls, snapshot: DataSnapshot) -> str:
        """Format the data analysis prompt with snapshot data."""
        # Format schema
        schema_lines = [f"- **{col}**: {dtype}" for col, dtype in snapshot.schema.items()]
        schema_text = "\n".join(schema_lines) if schema_lines else "No schema available"
        
        # Format sample data (first 5 rows)
        sample_rows = snapshot.rows[:5]
        if sample_rows:
            sample_data = "\n".join([str(row) for row in sample_rows])
        else:
            sample_data = "No sample data available"
        
        return cls.DATA_ANALYSIS_PROMPT.format(
            file_format=snapshot.file_format.upper(),
            total_rows=snapshot.total_rows,
            sample_size=snapshot.sample_size,
            num_columns=len(snapshot.schema),
            schema_text=schema_text,
            sample_data=sample_data,
        )
    
    @classmethod
    def format_modular_eda_prompt(
        cls,
        snapshot: DataSnapshot,
        step_key: str,
    ) -> str:
        """Format a modular EDA script prompt for a specific analysis step."""
        # Format schema
        schema_lines = [f"- {col}: {dtype}" for col, dtype in snapshot.schema.items()]
        schema_text = "\n".join(schema_lines) if schema_lines else "No schema available"
        
        prompt_template = cls.MODULAR_EDA_PROMPTS.get(step_key, "")
        if not prompt_template:
            raise ValueError(f"Unknown EDA step: {step_key}")
        
        return prompt_template.format(
            file_format=snapshot.file_format.upper(),
            total_rows=snapshot.total_rows,
            num_columns=len(snapshot.schema),
            schema_text=schema_text,
        )
    
    @classmethod
    def format_script_generation_prompt(
        cls,
        analysis: DataAnalysis,
        operation: str,
        context: ProcessingContext = None,
        schema: Dict[str, str] = None,
    ) -> str:
        """Format the script generation prompt."""
        # Format context
        context_lines = []
        if analysis.data_quality_issues:
            context_lines.append("Data Quality Issues:")
            context_lines.extend([f"  - {issue}" for issue in analysis.data_quality_issues])
        
        if analysis.processing_recommendations:
            context_lines.append("\nRecommendations:")
            context_lines.extend([f"  - {rec}" for rec in analysis.processing_recommendations])
        
        context_text = "\n".join(context_lines) if context_lines else "No additional context"
        
        # Format schema
        schema = schema or {}
        schema_lines = [f"- {col}: {dtype}" for col, dtype in schema.items()]
        schema_text = "\n".join(schema_lines) if schema_lines else "No schema available"
        
        # Format previous steps
        previous_steps = []
        if context and context.executed_scripts:
            previous_steps = [f"- {s.description}" for s in context.executed_scripts]
        previous_steps_text = "\n".join(previous_steps) if previous_steps else "None"
        
        return cls.SCRIPT_GENERATION_PROMPT.format(
            context=context_text,
            schema_text=schema_text,
            operation=operation,
            previous_steps=previous_steps_text,
        )
    
    @classmethod
    def format_validation_prompt(
        cls,
        script_description: str,
        expected_files: List[str],
        actual_files: List[str],
        file_contents: Dict[str, str],
        console_output: str,
    ) -> str:
        """Format the output validation prompt."""
        # Format file contents for display
        content_lines = []
        for filename, content in file_contents.items():
            # Truncate long content
            display_content = content[:2000] + "..." if len(content) > 2000 else content
            content_lines.append(f"### {filename}\n```\n{display_content}\n```")
        
        return cls.OUTPUT_VALIDATION_PROMPT.format(
            script_description=script_description,
            expected_files=", ".join(expected_files),
            actual_files=", ".join(actual_files),
            file_contents="\n\n".join(content_lines) if content_lines else "No files to validate",
            console_output=console_output[:3000] if console_output else "No output",
        )
    
    @classmethod
    def format_self_heal_prompt(
        cls,
        script_description: str,
        original_script: str,
        issues: List[str],
        console_output: str,
        file_contents: Dict[str, str],
    ) -> str:
        """Format the self-healing prompt."""
        # Format file contents for display
        content_lines = []
        for filename, content in file_contents.items():
            display_content = content[:1500] + "..." if len(content) > 1500 else content
            content_lines.append(f"### {filename}\n```\n{display_content}\n```")
        
        return cls.SELF_HEAL_PROMPT.format(
            script_description=script_description,
            original_script=original_script,
            issues="\n".join([f"- {issue}" for issue in issues]),
            console_output=console_output[:2000] if console_output else "No output",
            file_contents="\n\n".join(content_lines) if content_lines else "No files generated",
        )
