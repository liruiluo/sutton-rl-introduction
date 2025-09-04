#!/usr/bin/env python
"""
修复所有导入问题
Fix all import issues
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """修复单个文件中的导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 记录是否有修改
    modified = False
    original_content = content
    
    # 替换所有 ch02_mdp 为 src.ch03_finite_mdp
    if 'ch02_mdp' in content:
        # 处理 from ch02_mdp 导入
        content = re.sub(r'from ch02_mdp\.', 'from src.ch03_finite_mdp.', content)
        # 处理 import ch02_mdp
        content = re.sub(r'import ch02_mdp', 'import src.ch03_finite_mdp', content)
        modified = True
    
    # 替换所有 ch01_bandits 为 src.ch02_multi_armed_bandits  
    if 'ch01_bandits' in content:
        content = re.sub(r'from ch01_bandits\.', 'from src.ch02_multi_armed_bandits.', content)
        content = re.sub(r'import ch01_bandits', 'import src.ch02_multi_armed_bandits', content)
        modified = True
        
    # 替换所有 ch03_dp 为 src.ch04_dynamic_programming
    if 'ch03_dp' in content:
        content = re.sub(r'from ch03_dp\.', 'from src.ch04_dynamic_programming.', content)
        content = re.sub(r'import ch03_dp', 'import src.ch04_dynamic_programming', content)
        modified = True
        
    # 替换所有 ch04_mc 为 src.ch05_monte_carlo
    if 'ch04_mc' in content:
        content = re.sub(r'from ch04_mc\.', 'from src.ch05_monte_carlo.', content)
        content = re.sub(r'import ch04_mc', 'import src.ch05_monte_carlo', content)
        modified = True
        
    # 替换所有 ch05_td 为 src.ch06_temporal_difference
    if 'ch05_td' in content:
        content = re.sub(r'from ch05_td\.', 'from src.ch06_temporal_difference.', content)
        content = re.sub(r'import ch05_td', 'import src.ch06_temporal_difference', content)
        modified = True
    
    if modified:
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
    
    for py_file in src_dir.rglob('*.py'):
        total_count += 1
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"\n修复完成！Fixed {fixed_count}/{total_count} files")

if __name__ == "__main__":
    main()