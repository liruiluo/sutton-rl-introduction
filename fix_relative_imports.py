#!/usr/bin/env python
"""
修复章节内部的相对导入问题
Fix relative imports within chapters
"""

import os
import re
from pathlib import Path

def fix_relative_imports_in_file(file_path):
    """修复文件中的相对导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    modified = False
    
    # 修复第4章的相对导入
    if 'ch04_dynamic_programming' in str(file_path):
        replacements = [
            (r'^from dp_foundations import', 'from .dp_foundations import'),
            (r'^from policy_iteration import', 'from .policy_iteration import'),
            (r'^from value_iteration import', 'from .value_iteration import'),
            (r'^from generalized_policy_iteration import', 'from .generalized_policy_iteration import'),
            (r'^from dp_examples import', 'from .dp_examples import'),
        ]
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
            
    # 修复第5章的相对导入
    if 'ch05_monte_carlo' in str(file_path):
        replacements = [
            (r'^from mc_foundations import', 'from .mc_foundations import'),
            (r'^from mc_prediction import', 'from .mc_prediction import'),
            (r'^from mc_control import', 'from .mc_control import'),
            (r'^from importance_sampling import', 'from .importance_sampling import'),
            (r'^from mc_examples import', 'from .mc_examples import'),
        ]
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # 修复第6章的相对导入
    if 'ch06_temporal_difference' in str(file_path):
        replacements = [
            (r'^from td_foundations import', 'from .td_foundations import'),
            (r'^from td_control import', 'from .td_control import'),
            (r'^from n_step_td import', 'from .n_step_td import'),
        ]
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # 修复第7章的相对导入
    if 'ch07_n_step_bootstrapping' in str(file_path):
        replacements = [
            (r'^from n_step_td import', 'from .n_step_td import'),
            (r'^from n_step_sarsa import', 'from .n_step_sarsa import'),
            (r'^from off_policy_n_step import', 'from .off_policy_n_step import'),
            (r'^from tree_backup import', 'from .tree_backup import'),
        ]
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # 修复第8章的相对导入
    if 'ch08_planning_and_learning' in str(file_path):
        replacements = [
            (r'^from models_and_planning import', 'from .models_and_planning import'),
            (r'^from dyna_q import', 'from .dyna_q import'),
            (r'^from prioritized_sweeping import', 'from .prioritized_sweeping import'),
            (r'^from expected_vs_sample import', 'from .expected_vs_sample import'),
            (r'^from trajectory_sampling import', 'from .trajectory_sampling import'),
            (r'^from mcts import', 'from .mcts import'),
        ]
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # 修复第9章的相对导入
    if 'ch09_on_policy_approximation' in str(file_path):
        replacements = [
            (r'^from gradient_descent import', 'from .gradient_descent import'),
            (r'^from linear_approximation import', 'from .linear_approximation import'),
            (r'^from least_squares_td import', 'from .least_squares_td import'),
            (r'^from mountain_car import', 'from .mountain_car import'),
        ]
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in: {file_path}")
        return True
    return False

def main():
    """修复所有Python文件中的相对导入"""
    src_dir = Path(__file__).parent / 'src'
    
    fixed_count = 0
    total_count = 0
    
    # 处理第4-9章的所有Python文件
    for chapter in ['ch04_dynamic_programming', 'ch05_monte_carlo', 
                   'ch06_temporal_difference', 'ch07_n_step_bootstrapping',
                   'ch08_planning_and_learning', 'ch09_on_policy_approximation']:
        chapter_dir = src_dir / chapter
        if chapter_dir.exists():
            for py_file in chapter_dir.glob('*.py'):
                if py_file.name != '__init__.py':  # 不处理__init__.py
                    total_count += 1
                    if fix_relative_imports_in_file(py_file):
                        fixed_count += 1
    
    print(f"\n修复完成！Fixed {fixed_count}/{total_count} files")

if __name__ == "__main__":
    main()