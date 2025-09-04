#!/usr/bin/env python
"""
修复所有章节的导入问题
Fix all import issues in all chapters
"""

import os
import re
from pathlib import Path

def fix_chapter_imports(file_path):
    """修复文件中的章节导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复错误的章节引用
    replacements = [
        # 第5章：修复 ch04_monte_carlo -> ch05_monte_carlo 或相对导入
        (r'from ch04_monte_carlo\.mc_foundations', 'from .mc_foundations'),
        (r'from ch04_monte_carlo\.mc_prediction', 'from .mc_prediction'),
        (r'from ch04_monte_carlo\.mc_control', 'from .mc_control'),
        (r'from ch04_monte_carlo\.importance_sampling', 'from .importance_sampling'),
        (r'from ch04_monte_carlo\.mc_examples', 'from .mc_examples'),
        (r'import ch04_monte_carlo', 'import src.ch05_monte_carlo'),
        
        # 第6章：修复 ch05_temporal_difference -> ch06_temporal_difference
        (r'from ch05_temporal_difference\.', 'from .'),
        (r'import ch05_temporal_difference', 'import src.ch06_temporal_difference'),
        
        # 第7章：修复 ch06_n_step -> ch07_n_step_bootstrapping
        (r'from ch06_n_step\.', 'from .'),
        
        # 第8章：修复 ch07_planning -> ch08_planning_and_learning  
        (r'from ch07_planning\.', 'from .'),
        
        # 第9章：修复 ch08_on_policy -> ch09_on_policy_approximation
        (r'from ch08_on_policy\.', 'from .'),
        
        # 修复没有点号的相对导入（在章节内部）
        (r'^from (mc_foundations|mc_prediction|mc_control|importance_sampling|mc_examples) import', 
         r'from .\1 import'),
        (r'^from (td_foundations|td_control|n_step_td) import', 
         r'from .\1 import'),
        (r'^from (n_step_td|n_step_sarsa|off_policy_n_step|tree_backup) import', 
         r'from .\1 import'),
        (r'^from (models_and_planning|dyna_q|prioritized_sweeping|expected_vs_sample|trajectory_sampling|mcts) import', 
         r'from .\1 import'),
        (r'^from (gradient_descent|linear_approximation|least_squares_td|mountain_car) import', 
         r'from .\1 import'),
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
    """修复所有Python文件中的导入"""
    src_dir = Path(__file__).parent / 'src'
    
    fixed_count = 0
    total_count = 0
    
    # 处理所有章节的Python文件
    for py_file in src_dir.rglob('*.py'):
        if py_file.name != '__init__.py' and 'test_' not in py_file.name:
            total_count += 1
            if fix_chapter_imports(py_file):
                fixed_count += 1
    
    print(f"\n修复完成！Fixed {fixed_count}/{total_count} files")

if __name__ == "__main__":
    main()