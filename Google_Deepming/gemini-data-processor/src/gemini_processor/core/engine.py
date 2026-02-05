"""Main processing engine that orchestrates the data processing workflow."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.syntax import Syntax

from ..models.config import AppConfig
from ..models.data import (
    DataAnalysis,
    ExecutionResult,
    ProcessingContext,
    ProcessingSession,
)
from ..models.enums import SessionStatus
from ..security import SecurityManager
from .data_analyzer import DataAnalyzer
from .script_coordinator import ScriptCoordinator

console = Console()


class ProcessingEngine:
    """Coordinates the data processing workflow."""
    
    def __init__(self, config: AppConfig):
        """
        Initialize the processing engine.
        
        Args:
            config: Application configuration.
        """
        self.config = config
        self.script_coordinator = ScriptCoordinator()
        self._current_session: Optional[ProcessingSession] = None
        self._docker_manager = None  # Lazy initialization, reused across scripts
        self._security_manager = SecurityManager()
        self._gemini_client = None  # Track Gemini client for cleanup
    
    def process_file(
        self,
        input_file: str,
        output_dir: Optional[str] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> ExecutionResult:
        """
        Process a data file using Gemini AI.
        
        Args:
            input_file: Path to the input data file.
            output_dir: Directory for output files.
            dry_run: If True, show analysis without executing scripts.
            verbose: If True, show detailed output.
            
        Returns:
            ExecutionResult with processing outcome.
        """
        input_path = Path(input_file)
        
        # Set up output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = input_path.parent / f"output_{timestamp}"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create session
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{input_path.stem}"
        self._current_session = ProcessingSession(
            session_id=session_id,
            input_file=str(input_path),
            output_directory=str(output_path),
        )
        
        try:
            # Step 1: Analyze data structure
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Analyzing data structure...", total=None)
                
                analyzer = DataAnalyzer(input_file)
                snapshot = analyzer.create_snapshot()
                self._current_session.data_snapshot = snapshot
                
                progress.update(task, description="[green]✓[/green] Data analysis complete")
            
            # Display snapshot info
            console.print()
            console.print(Panel.fit(
                f"[bold]File Format:[/bold] {snapshot.file_format.upper()}\n"
                f"[bold]Total Rows:[/bold] {snapshot.total_rows:,}\n"
                f"[bold]Sample Size:[/bold] {snapshot.sample_size}\n"
                f"[bold]Columns:[/bold] {len(snapshot.schema)}",
                title="Data Summary",
                border_style="cyan"
            ))
            
            # Display schema
            if verbose and snapshot.schema:
                console.print("\n[bold]Schema:[/bold]")
                for col, dtype in snapshot.schema.items():
                    console.print(f"  • {col}: [dim]{dtype}[/dim]")
            
            # Step 2: Get AI analysis
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Getting AI analysis...", total=None)
                
                analysis = self._get_ai_analysis(snapshot)
                self._current_session.current_context.previous_analyses.append(analysis)
                
                progress.update(task, description="[green]✓[/green] AI analysis complete")
            
            # Display analysis
            console.print()
            console.print(Panel.fit(
                "\n".join([f"• {issue}" for issue in analysis.data_quality_issues[:5]]) 
                if analysis.data_quality_issues else "No issues detected",
                title="Data Quality Issues",
                border_style="yellow"
            ))
            
            console.print()
            console.print(Panel.fit(
                "\n".join([f"• {rec}" for rec in analysis.processing_recommendations[:5]])
                if analysis.processing_recommendations else "No recommendations",
                title="Processing Recommendations",
                border_style="green"
            ))
            
            if dry_run:
                console.print("\n[yellow]Dry run mode - no scripts will be executed.[/yellow]")
                self._current_session.status = SessionStatus.COMPLETED
                return ExecutionResult(
                    success=True,
                    output_data=json.dumps({
                        "analysis": {
                            "issues": analysis.data_quality_issues,
                            "recommendations": analysis.processing_recommendations,
                        }
                    }),
                )
            
            # Step 3: Generate and execute scripts
            scripts = self._generate_scripts(analysis)
            
            if not scripts:
                console.print("\n[yellow]No processing scripts generated.[/yellow]")
                self._current_session.status = SessionStatus.COMPLETED
                return ExecutionResult(success=True)
            
            console.print(f"\n[bold]Generated {len(scripts)} processing script(s)[/bold]\n")
            
            output_files = []
            
            for i, script in enumerate(scripts, 1):
                console.print(f"\n[bold cyan]Script {i}/{len(scripts)}:[/bold cyan] {script.description}")
                
                # Show script content
                console.print()
                syntax = Syntax(script.content, "python", theme="monokai", line_numbers=True)
                console.print(Panel(syntax, title="Script Content", border_style="blue"))
                
                # Show validation status
                if script.security_level == "caution":
                    console.print("[yellow]⚠ This script has security warnings[/yellow]")
                
                # Request approval
                if not Confirm.ask("\nExecute this script?", default=True):
                    console.print("[dim]Skipped[/dim]")
                    continue
                
                # Execute script with self-healing validation
                max_attempts = 2  # Original + 1 retry after self-heal
                current_script = script
                
                for attempt in range(max_attempts):
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        execution_desc = "Executing script..." if attempt == 0 else "Re-running fixed script..."
                        task = progress.add_task(execution_desc, total=None)
                        
                        result = self._execute_script(current_script, output_path)
                        
                        if result.success:
                            progress.update(task, description="[green]✓[/green] Script completed")
                        else:
                            progress.update(task, description="[red]✗[/red] Script failed")
                            console.print(f"\n[red]Error:[/red] {result.error_message}")
                            break  # Don't retry on execution failure
                    
                    # Validate output if execution succeeded
                    if result.success:
                        validation = self._validate_and_display_results(
                            script=current_script,
                            result=result,
                            output_path=output_path,
                        )
                        
                        if validation.get("is_valid", True):
                            output_files.extend(result.output_files)
                            self.script_coordinator.record_execution(current_script)
                            break  # Success, move to next script
                        
                        elif attempt < max_attempts - 1:
                            # Try self-healing
                            console.print(f"\n[yellow]⚠ Validation issues found:[/yellow]")
                            for issue in validation.get("issues", []):
                                console.print(f"  [dim]•[/dim] {issue}")
                            
                            console.print("\n[cyan]🔧 Attempting self-healing fix...[/cyan]")
                            
                            try:
                                fixed_script = self._generate_self_healed_script(
                                    script=current_script,
                                    validation=validation,
                                    result=result,
                                    output_path=output_path,
                                )
                                current_script = fixed_script
                                console.print("[green]✓[/green] Generated fixed script, retrying...")
                            except Exception as e:
                                console.print(f"[red]Self-healing failed:[/red] {e}")
                                output_files.extend(result.output_files)
                                break
                        else:
                            # Max attempts reached
                            console.print(f"\n[yellow]⚠ Validation issues remain after retry[/yellow]")
                            output_files.extend(result.output_files)
            
            self._current_session.status = SessionStatus.COMPLETED
            
            # Save processing summary
            summary_file = output_path / "processing_summary.json"
            self._save_summary(summary_file)
            output_files.append(str(summary_file))
            
            return ExecutionResult(
                success=True,
                output_files=output_files,
            )
            
        except Exception as e:
            self._current_session.status = SessionStatus.FAILED
            return ExecutionResult(
                success=False,
                error_message=str(e),
            )
        finally:
            # Always cleanup Docker session and API keys at end
            self._cleanup_docker_session()
            self._cleanup_security()
    
    def _get_ai_analysis(self, snapshot) -> DataAnalysis:
        """Get AI analysis of the data snapshot."""
        # Import Gemini integration
        try:
            from ..gemini import GeminiIntegration
            
            if self.config.gemini:
                self._gemini_client = GeminiIntegration(self.config.gemini)
                return self._gemini_client.analyze_data_snapshot(snapshot)
        except ImportError:
            pass
        except Exception as e:
            console.print(f"[yellow]Warning: AI analysis failed: {e}[/yellow]")
        
        # Fallback to basic analysis
        return self._basic_analysis(snapshot)
    
    def _basic_analysis(self, snapshot) -> DataAnalysis:
        """Perform basic analysis without AI."""
        issues = []
        recommendations = []
        column_insights = {}
        
        # Analyze schema
        for col, dtype in snapshot.schema.items():
            column_insights[col] = f"Type: {dtype}"
            
            if dtype == "null":
                issues.append(f"Column '{col}' appears to have many null values")
                recommendations.append(f"Consider handling null values in '{col}'")
        
        # Check for potential issues in sample data
        if snapshot.rows:
            # Check for empty strings
            for col in snapshot.schema:
                empty_count = sum(
                    1 for row in snapshot.rows 
                    if col in row and (row[col] == "" or row[col] is None)
                )
                if empty_count > len(snapshot.rows) * 0.1:
                    issues.append(f"Column '{col}' has {empty_count} empty/null values in sample")
        
        if not issues:
            issues.append("No obvious data quality issues detected in sample")
        
        if not recommendations:
            recommendations.append("Data appears clean - consider exploring further transformations")
        
        return DataAnalysis(
            data_quality_issues=issues,
            suggested_operations=["Basic data cleaning", "Data type validation"],
            column_insights=column_insights,
            processing_recommendations=recommendations,
            estimated_complexity="low",
        )
    
    def _generate_scripts(self, analysis: DataAnalysis):
        """Generate processing scripts based on analysis, starting with modular EDA."""
        scripts = []
        
        try:
            from ..gemini import GeminiIntegration
            
            if self.config.gemini and self._current_session and self._current_session.data_snapshot:
                gemini = GeminiIntegration(self.config.gemini)
                
                # First, generate modular EDA scripts (each focused on one task)
                console.print("\n[dim]Generating modular EDA scripts for full dataset analysis...[/dim]")
                try:
                    eda_scripts = gemini.generate_modular_eda_scripts(
                        self._current_session.data_snapshot
                    )
                    scripts.extend(eda_scripts)
                    console.print(f"[dim]Generated {len(eda_scripts)} EDA analysis scripts[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Warning: EDA script generation failed: {e}[/yellow]")
                
                # Then generate processing scripts based on AI recommendations
                console.print("[dim]Generating data processing scripts...[/dim]")
                processing_scripts = gemini.generate_processing_scripts(
                    analysis, 
                    self._current_session.current_context if self._current_session else None
                )
                scripts.extend(processing_scripts)
                
        except ImportError:
            pass
        except Exception as e:
            console.print(f"[yellow]Warning: Script generation failed: {e}[/yellow]")
        
        return scripts
    
    def _execute_script(self, script, output_path: Path) -> ExecutionResult:
        """Execute a script in a Docker container, reusing session across scripts."""
        try:
            from ..docker import DockerManager
            
            # Reuse docker manager to keep session alive
            if self._docker_manager is None:
                self._docker_manager = DockerManager(self.config.resources)
            
            return self._docker_manager.execute_script(
                script=script,
                input_file=self._current_session.input_file if self._current_session else "",
                output_dir=str(output_path),
            )
        except ImportError:
            return ExecutionResult(
                success=False,
                error_message="Docker module not available"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    def _cleanup_docker_session(self) -> None:
        """Clean up Docker session at end of processing."""
        if self._docker_manager:
            self._docker_manager.end_session()
            self._docker_manager = None
    
    def _cleanup_security(self) -> None:
        """Clean up API keys and sensitive data at end of processing."""
        try:
            # Clean up Gemini client
            if self._gemini_client:
                self._security_manager.cleanup_gemini_client(self._gemini_client)
                self._gemini_client = None
            
            # Wipe all API keys from storage and memory
            success = self._security_manager.wipe_all_api_keys()
            if success:
                console.print("[dim]🔒 API keys securely wiped from memory and storage[/dim]")
            else:
                console.print("[yellow]⚠ Warning: API key cleanup may have been incomplete[/yellow]")
            
            # Perform secure memory cleanup
            self._security_manager.secure_memory_cleanup()
            
        except Exception as e:
            console.print(f"[yellow]Warning: Security cleanup failed: {e}[/yellow]")
    
    def _save_summary(self, summary_file: Path) -> None:
        """Save processing summary to file."""
        if not self._current_session:
            return
        
        summary = {
            "session_id": self._current_session.session_id,
            "input_file": self._current_session.input_file,
            "output_directory": self._current_session.output_directory,
            "created_at": self._current_session.created_at.isoformat(),
            "status": self._current_session.status.value,
            "scripts_executed": len(self.script_coordinator.get_executed_scripts()),
            "data_snapshot": {
                "total_rows": self._current_session.data_snapshot.total_rows if self._current_session.data_snapshot else 0,
                "sample_size": self._current_session.data_snapshot.sample_size if self._current_session.data_snapshot else 0,
                "file_format": self._current_session.data_snapshot.file_format if self._current_session.data_snapshot else "unknown",
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _validate_and_display_results(
        self,
        script,
        result: ExecutionResult,
        output_path: Path,
    ) -> dict:
        """Validate script output and display results in terminal."""
        from rich.table import Table
        from rich.json import JSON
        
        # Find JSON files created by this script
        json_files = list(output_path.glob("*.json"))
        png_files = list(output_path.glob("*.png"))
        csv_files = list(output_path.glob("*.csv"))
        
        # Read file contents for validation
        file_contents = {}
        for json_file in json_files:
            try:
                content = json_file.read_text()
                file_contents[json_file.name] = content
                
                # Try to parse JSON to check validity
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    file_contents[json_file.name] = f"PARSE ERROR: {e}\n\nContent:\n{content}"
            except Exception as e:
                file_contents[json_file.name] = f"READ ERROR: {e}"
        
        # Display results in terminal
        console.print("\n[bold green]📊 Script Results:[/bold green]")
        
        # Show console output summary
        if result.output_data:
            output_lines = result.output_data.strip().split('\n')[-10:]  # Last 10 lines
            console.print(Panel(
                "\n".join(output_lines),
                title="Console Output (last 10 lines)",
                border_style="dim"
            ))
        
        # Show generated files
        if json_files or png_files or csv_files:
            table = Table(title="Generated Files", show_header=True)
            table.add_column("File", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Size", justify="right")
            
            for f in json_files:
                table.add_row(f.name, "JSON", f"{f.stat().st_size:,} bytes")
            for f in png_files:
                table.add_row(f.name, "PNG", f"{f.stat().st_size:,} bytes")
            for f in csv_files:
                table.add_row(f.name, "CSV", f"{f.stat().st_size:,} bytes")
            
            console.print(table)
        
        # Show JSON content preview for small files
        for json_file, content in file_contents.items():
            if "PARSE ERROR" not in content and len(content) < 1500:
                try:
                    console.print(Panel(
                        JSON(content),
                        title=f"📄 {json_file}",
                        border_style="blue"
                    ))
                except Exception:
                    pass  # Skip if can't render
        
        # Validate using Gemini
        try:
            from ..gemini import GeminiIntegration
            
            if self.config.gemini:
                gemini = GeminiIntegration(self.config.gemini)
                validation = gemini.validate_script_output(
                    script_description=script.description,
                    expected_files=script.output_files,
                    actual_files=[f.name for f in json_files + png_files + csv_files],
                    file_contents=file_contents,
                    console_output=result.output_data or "",
                )
                
                # Display validation result
                if validation.get("is_valid"):
                    console.print("[green]✓ Output validated successfully[/green]")
                else:
                    severity = validation.get("severity", "minor")
                    severity_color = {"minor": "yellow", "major": "orange", "critical": "red"}.get(severity, "yellow")
                    console.print(f"[{severity_color}]⚠ Validation: {severity.upper()}[/{severity_color}]")
                
                return validation
        except ImportError:
            pass
        except Exception as e:
            console.print(f"[dim]Validation skipped: {e}[/dim]")
        
        return {"is_valid": True, "issues": [], "suggestions": [], "severity": "none"}
    
    def _generate_self_healed_script(
        self,
        script,
        validation: dict,
        result: ExecutionResult,
        output_path: Path,
    ):
        """Generate a fixed script using Gemini AI."""
        from ..gemini import GeminiIntegration
        from ..models.data import ProcessingScript
        
        if not self.config.gemini:
            raise RuntimeError("Gemini not configured for self-healing")
        
        # Collect file contents for context
        file_contents = {}
        for json_file in output_path.glob("*.json"):
            try:
                file_contents[json_file.name] = json_file.read_text()[:2000]
            except Exception:
                pass
        
        gemini = GeminiIntegration(self.config.gemini)
        fixed_content = gemini.generate_fixed_script(
            script_description=script.description,
            original_script=script.content,
            issues=validation.get("issues", []),
            console_output=result.output_data or "",
            file_contents=file_contents,
        )
        
        # Create new script with fixed content
        return ProcessingScript(
            script_id=f"{script.script_id}_fixed",
            content=fixed_content,
            description=f"{script.description} (Fixed)",
            required_packages=script.required_packages,
            input_files=script.input_files,
            output_files=script.output_files,
            validation_status=script.validation_status,
        )
