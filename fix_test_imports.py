#!/usr/bin/env python
"""
修复所有测试文件中的导入问题
Fix all import issues in test files
"""

import os
import re
from pathlib import Path

def fix_test_imports(file_path):
    """修复测试文件中的导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复所有错误的导入映射
    replacements = [
        # ch02_mdp -> ch03_finite_mdp
        (r'from src\.ch02_mdp\.', 'from src.ch03_finite_mdp.'),
        (r'import src\.ch02_mdp', 'import src.ch03_finite_mdp'),
        
        # ch04_mc -> ch05_monte_carlo  
        (r'from src\.ch04_mc\.', 'from src.ch05_monte_carlo.'),
        (r'import src\.ch04_mc', 'import src.ch05_monte_carlo'),
        
        # ch05_td -> ch06_temporal_difference
        (r'from src\.ch05_td\.', 'from src.ch06_temporal_difference.'),
        (r'import src\.ch05_td', 'import src.ch06_temporal_difference'),
        
        # ch05_temporal_difference -> ch06_temporal_difference（修复测试文件中的章节号错误）
        (r'from src\.ch05_temporal_difference\.', 'from src.ch06_temporal_difference.'),
        
        # 修复测试文件本身的错误章节引用
        (r'from src\.ch06_n_step\.', 'from src.ch07_n_step_bootstrapping.'),
        (r'from src\.ch07_planning\.', 'from src.ch08_planning_and_learning.'),
        (r'from src\.ch08_on_policy\.', 'from src.ch09_on_policy_approximation.'),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed imports in: {file_path}")
        return True
    return False

def main():
    """修复所有测试文件中的导入"""
    src_dir = Path(__file__).parent / 'src'
    
    fixed_count = 0
    total_count = 0
    
    # 只处理test_*.py文件
    for test_file in src_dir.rglob('test_*.py'):
        total_count += 1
        if fix_test_imports(test_file):
            fixed_count += 1
    
    print(f"\n修复完成！Fixed {fixed_count}/{total_count} test files")

if __name__ == "__main__":
    main()