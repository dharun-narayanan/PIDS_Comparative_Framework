"""
Task manager for orchestrating the PIDS pipeline.

Tasks are executed in dependency order, with results cached to avoid
re-computation. Similar to PIDSMaker's task execution model.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import pickle
import json

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a pipeline task."""
    name: str
    function: Callable
    dependencies: List[str]
    config: Dict[str, Any]
    output_path: Optional[Path] = None
    completed: bool = False
    result: Any = None


class TaskManager:
    """
    Manages task execution in the PIDS pipeline.
    
    Task execution flow:
    1. Check if task output exists (skip if found and not forcing restart)
    2. Execute dependencies first
    3. Run task function with config
    4. Save task output and mark as completed
    """
    
    def __init__(self, config: Dict[str, Any], force_restart: bool = False):
        """
        Initialize task manager.
        
        Args:
            config: Global configuration dictionary
            force_restart: If True, re-run all tasks regardless of cached results
        """
        self.config = config
        self.force_restart = force_restart
        self.tasks: Dict[str, Task] = {}
        self.execution_times: Dict[str, float] = {}
        
        # Set up artifact directory
        self.artifact_dir = Path(config.get('artifact_dir', 'artifacts'))
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
    def register_task(
        self,
        name: str,
        function: Callable,
        dependencies: List[str],
        config: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """
        Register a task in the pipeline.
        
        Args:
            name: Unique task name
            function: Function to execute for this task
            dependencies: List of task names that must complete first
            config: Task-specific configuration
            output_path: Path where task output should be saved
        """
        if output_path is None:
            output_path = self.artifact_dir / name / 'output.pkl'
        
        self.tasks[name] = Task(
            name=name,
            function=function,
            dependencies=dependencies,
            config=config,
            output_path=output_path,
            completed=False
        )
        logger.debug(f"Registered task: {name} (deps: {dependencies})")
    
    def should_run_task(self, task: Task) -> bool:
        """
        Determine if a task should be executed.
        
        Args:
            task: Task to check
            
        Returns:
            True if task should run, False if can skip
        """
        if self.force_restart:
            return True
        
        if task.completed:
            return False
        
        if task.output_path and task.output_path.exists():
            logger.info(f"Task '{task.name}' output found at {task.output_path}, skipping...")
            # Load cached result
            try:
                with open(task.output_path, 'rb') as f:
                    task.result = pickle.load(f)
                task.completed = True
                return False
            except Exception as e:
                logger.warning(f"Failed to load cached result for {task.name}: {e}")
                return True
        
        return True
    
    def execute_task(self, task_name: str) -> Any:
        """
        Execute a task and its dependencies.
        
        Args:
            task_name: Name of task to execute
            
        Returns:
            Task result
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not registered")
        
        task = self.tasks[task_name]
        
        # Check if already completed
        if task.completed:
            logger.debug(f"Task '{task_name}' already completed")
            return task.result
        
        # Execute dependencies first (always needed, even when task is cached)
        dependency_results = {}
        for dep_name in task.dependencies:
            logger.debug(f"Task '{task_name}' requires '{dep_name}'")
            dependency_results[dep_name] = self.execute_task(dep_name)
        
        # Check if can skip (after loading dependencies)
        if not self.should_run_task(task):
            return task.result
        
        # Execute task
        logger.info(f"Executing task: {task_name}")
        start_time = time.time()
        
        try:
            # Pass global config, task config, and dependency results
            result = task.function(
                config=self.config,
                task_config=task.config,
                dependencies=dependency_results
            )
            
            task.result = result
            task.completed = True
            
            # Save result
            if task.output_path:
                task.output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(task.output_path, 'wb') as f:
                    pickle.dump(result, f)
                logger.debug(f"Saved task result to {task.output_path}")
            
            elapsed = time.time() - start_time
            self.execution_times[task_name] = elapsed
            logger.info(f"Task '{task_name}' completed in {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Task '{task_name}' failed: {e}")
            raise
    
    def execute_pipeline(self, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute all tasks or a subset of tasks.
        
        Args:
            tasks: List of task names to execute. If None, executes all tasks.
            
        Returns:
            Dictionary mapping task names to results
        """
        if tasks is None:
            tasks = list(self.tasks.keys())
        
        logger.info(f"Executing pipeline with {len(tasks)} task(s)")
        
        results = {}
        for task_name in tasks:
            results[task_name] = self.execute_task(task_name)
        
        # Log execution summary
        total_time = sum(self.execution_times.values())
        logger.info("\n" + "="*80)
        logger.info("Pipeline Execution Summary")
        logger.info("="*80)
        for task_name, exec_time in self.execution_times.items():
            logger.info(f"  {task_name}: {exec_time:.2f}s")
        logger.info(f"\nTotal execution time: {total_time:.2f}s")
        logger.info("="*80)
        
        return results
    
    def get_task_result(self, task_name: str) -> Any:
        """Get the result of a completed task."""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not registered")
        
        task = self.tasks[task_name]
        if not task.completed:
            raise RuntimeError(f"Task '{task_name}' has not been executed yet")
        
        return task.result
    
    def reset_task(self, task_name: str):
        """Reset a task to allow re-execution."""
        if task_name in self.tasks:
            self.tasks[task_name].completed = False
            self.tasks[task_name].result = None
            if task_name in self.execution_times:
                del self.execution_times[task_name]
    
    def reset_all(self):
        """Reset all tasks."""
        for task_name in self.tasks:
            self.reset_task(task_name)
        self.execution_times.clear()
    
    def save_execution_metadata(self, output_path: Path):
        """Save execution metadata (times, config, etc.)."""
        metadata = {
            'execution_times': self.execution_times,
            'total_time': sum(self.execution_times.values()),
            'tasks_executed': [name for name, task in self.tasks.items() if task.completed],
            'config': self.config
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved execution metadata to {output_path}")
