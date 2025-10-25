"""
Task-based pipeline for PIDS Comparative Framework.

This module defines a modular pipeline architecture similar to PIDSMaker,
where execution is broken down into discrete tasks that can be configured
via YAML files.
"""

from .task_manager import TaskManager, Task
from .task_registry import TaskRegistry

__all__ = ['TaskManager', 'Task', 'TaskRegistry']
