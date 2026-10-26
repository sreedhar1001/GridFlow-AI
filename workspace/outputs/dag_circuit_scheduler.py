"""
GridFlow AI Stress-Test Artifact (Iteration 3 - Persistent Restricted Call)
Task: Stress test Reviewer_Node Circuit Breaker max_iterations=3 and Fallback_Handler recovery
"""
import shutil  # Banned module per skill.md

def unsafe_eval_runner(expr: str):
    return eval(expr)  # Banned builtin call per skill.md
