# MATLAB中Git的使用方法

## 概述
MATLAB支持通过命令行调用Git命令来进行版本控制。本文档介绍在MATLAB环境中使用Git的完整方法。

## 前提条件
- 已安装Git并配置到系统PATH
- 已有Git仓库或准备初始化新仓库
- MATLAB 2014b或更高版本

## 基本语法

### 两种调用方式

1. **使用 `system()` 函数**（推荐用于需要捕获返回值的场景）
```matlab
[status, result] = system('git command');
```

2. **使用 `!` 操作符**（推荐用于交互式操作，更简洁）
```matlab
!git command
```

## 常用Git命令在MATLAB中的使用

### 1. 仓库状态查看
```matlab
% 查看仓库状态
!git status

% 查看提交历史
!git log --oneline

% 查看分支
!git branch
```

### 2. 文件操作
```matlab
% 添加文件到暂存区
!git add filename.m
!git add .                    % 添加所有文件

% 提交更改
!git commit -m "提交信息"

% 查看文件差异
!git diff filename.m
```

### 3. 远程仓库操作
```matlab
% 拉取最新代码
!git pull

% 推送到远程仓库
!git push

% 查看远程仓库
!git remote -v
```

### 4. 分支操作
```matlab
% 创建新分支
!git branch new-branch

% 切换分支
!git checkout branch-name

% 创建并切换到新分支
!git checkout -b new-branch

% 合并分支
!git merge branch-name
```

## 高级用法

### 捕获命令结果进行处理
```matlab
% 获取当前分支名
[status, branch] = system('git branch --show-current');
branch = strtrim(branch);  % 去除换行符
fprintf('当前分支: %s\n', branch);

% 检查是否有未提交的更改
[status, result] = system('git status --porcelain');
if isempty(strtrim(result))
    disp('工作目录干净');
else
    disp('有未提交的更改');
end
```

### 批量操作脚本
```matlab
function gitAutoCommit(message)
    % 自动添加、提交和推送
    if nargin < 1
        message = sprintf('Auto commit at %s', datestr(now));
    end
    
    % 添加所有更改
    [status1, ~] = system('git add .');
    
    % 提交更改
    cmd = sprintf('git commit -m "%s"', message);
    [status2, ~] = system(cmd);
    
    % 推送到远程
    [status3, ~] = system('git push');
    
    if status1 == 0 && status2 == 0 && status3 == 0
        fprintf('成功提交并推送: %s\n', message);
    else
        fprintf('操作失败，请检查Git状态\n');
    end
end
```

### 目录管理
```matlab
% 切换到项目目录并执行Git操作
function switchAndGit(projectPath, gitCommand)
    currentDir = pwd;
    try
        cd(projectPath);
        system(['git ' gitCommand]);
    catch ME
        fprintf('错误: %s\n', ME.message);
    end
    cd(currentDir);  % 恢复原目录
end

% 使用示例
switchAndGit('D:\MyProject', 'status');
```

## MATLAB内置Git支持（2014b+）

### 使用matlab.git函数
```matlab
% 注意：这些函数可能在某些版本中不可用
try
    matlab.git.pull();
    matlab.git.push();
catch
    % 降级到system调用
    system('git pull');
    system('git push');
end
```

## 常见问题及解决方案

### 1. "git"无法识别
**原因**: Git路径未添加到MATLAB环境变量
```matlab
% 检查PATH
getenv('PATH')

% 手动添加Git路径
gitPath = 'C:\Program Files\Git\cmd';
currentPath = getenv('PATH');
if ~contains(currentPath, gitPath)
    setenv('PATH', [currentPath ';' gitPath]);
end
```

### 2. 中文字符问题
```matlab
% 设置Git配置处理中文
!git config --global core.quotepath false
```

### 3. 长路径问题
```matlab
% 启用长路径支持
!git config --global core.longpaths true
```

## 最佳实践

### 1. 工作流程建议
```matlab
% 标准工作流程
!git pull                    % 1. 拉取最新代码
% 进行代码修改...
!git add .                   % 2. 添加更改
!git status                  % 3. 确认状态
!git commit -m "描述更改"     % 4. 提交
!git push                    % 5. 推送
```

### 2. 安全检查
```matlab
% 提交前检查
[status, result] = system('git status --porcelain');
if ~isempty(strtrim(result))
    disp('准备提交的文件:');
    disp(result);
    response = input('确认提交? (y/n): ', 's');
    if lower(response) == 'y'
        !git commit -m "MATLAB auto commit"
        !git push
    end
end
```

### 3. 项目组织
- 将MATLAB项目文件(.m, .mat, .fig)加入版本控制
- 使用.gitignore忽略临时文件和大型数据文件
- 定期提交，保持清晰的提交历史

## .gitignore建议（MATLAB项目）
```
# MATLAB临时文件
*.asv
*.m~

# MATLAB数据文件（根据需要选择性忽略）
*.mat

# MATLAB代码生成
codegen/
slprj/

# 系统文件
.DS_Store
Thumbs.db
```

## 总结
在MATLAB中使用Git主要通过`!`操作符或`system()`函数调用命令行Git命令。掌握这些基本操作后，可以有效地在MATLAB开发环境中进行版本控制。