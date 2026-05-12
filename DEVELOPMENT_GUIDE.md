# 项目开发与多端协作指南 (人类开发者必读)

这份指南专门为你（人类开发者/项目管理者）编写。因为本项目跨越了 **3 台电脑**（两台 Win，一台 Mac）并且使用了 **3 种 AI 工具**（Claude Code、Codex、Antigravity），如果没有一个铁的纪律，极易发生代码冲突、进度相互覆盖的灾难。

为了避免这些问题，**你是唯一的任务调度中枢**，AI 绝对不能擅自决定下一步做什么。

---

## 🔑 核心法则：状态锁与单点修改

所有的多端协作冲突都源于“两个人同时修改了同一个文件”。我们通过 `.tasks.yaml` 结合 Git 来实现**状态锁**。

**你的核心责任是：在任何 AI 开始写代码之前，先在 `.tasks.yaml` 中把任务分配给它，并同步给所有机器。**

---

## 🛠️ 标准开发操作流（SOP）

请在每次开发新功能时，**严格按以下 6 步操作**：

### 步骤 1：获取最新状态 (所有操作的起点)
在你当前要使用的电脑上，打开终端：
```bash
git pull
```
*如果不执行这步，你可能会覆盖另一台电脑上 AI 刚刚写好的代码。*

### 步骤 2：分配任务并上锁 (人类操作)
打开项目根目录的 `.tasks.yaml`，找到下一步要做的任务。
将任务的状态从 `free` 改为 `in_progress`，并指定你要使用的 AI 工具（`claude` / `codex` / `antigravity`）：

```yaml
# 修改前：
- id: task_09
  description: "实现特征提取模块"
  status: free
  handler: null

# 修改后：
- id: task_09
  description: "实现特征提取模块"
  status: in_progress     # <-- 加锁
  handler: antigravity    # <-- 指定干活的 AI
```

### 步骤 3：全网广播这个锁 (人类操作)
保存 `.tasks.yaml`，然后提交并推送到 GitHub：
```bash
git add .tasks.yaml
git commit -m "chore: assign task_09 to antigravity"
git push
```
*这一步极其关键。它告诉另外两台电脑和另外两个 AI：“这个活儿已经被 Antigravity 承包了，你们别碰”。*

### 步骤 4：呼叫 AI 开始干活
在指定的机器上，唤醒你分配好的 AI（比如唤醒 Antigravity 或 Claude Code），只对它说一句话：
> *"请读取 `.tasks.yaml`，执行分配给你（状态为 `in_progress`）的任务。做完后告诉我。"*

AI 会自动读取 `AI_CONTEXT.md` 了解背景，然后去写代码。

### 步骤 5：验收并提交代码 (人类/AI 操作)
AI 写完代码后，你跑一下测试或检查一下代码。如果满意，让 AI 或你自己把代码提交：
```bash
git add .
git commit -m "feat: complete feature extraction module"
```

### 步骤 6：释放锁 (人类操作)
活干完了，再次打开 `.tasks.yaml`，把状态改成 `done`：
```yaml
- id: task_09
  description: "实现特征提取模块"
  status: done           # <-- 解锁并标记完成
  handler: antigravity
```
然后最后一次推送：
```bash
git add .tasks.yaml
git commit -m "chore: mark task_09 as done"
git push
```

---

## 💻 3 台电脑的职责划分建议

为了最大化利用你的硬件，建议这样分配工作：

1. **MacBook M1 Pro (日常开发机)**
   - **工具**: 主要使用 Cursor (集成 Claude) 或 Antigravity
   - **工作内容**: 写数据预处理脚本（ECG/PPG/SCG 滤波）、写测试用例、跑 CPU 级别的小批量数据验证、写文档。
   - **依赖**: 跑 `pip install -r requirements-mac.txt`。

2. **Windows 1 (带有 WSL2 和 Nvidia GPU)**
   - **工具**: Codex 或 Claude Code (CLI)
   - **工作内容**: 跑深度学习模型（ResNet-SE-LSTM）的正式训练、调参、超参数搜索。
   - **依赖**: 跑 `pip install -r requirements-wsl.txt`。

3. **Windows 2 (备用机/文档机)**
   - **工作内容**: 数据整理、查阅论文、代码 Review。

---

## 🤖 3 个 AI 的脾气与指南文件

为了让不同的 AI 守规矩，项目里准备了三个对应的紧箍咒（文件）。你不需要管这些文件，AI 启动时会自动看：

- **Antigravity**: 直接读取 `AI_CONTEXT.md`。
- **Claude Code**: 启动时会读取 `CLAUDE.md`（里面强制要求它去读 AI_CONTEXT）。
- **Codex**: 启动时会读取 `AGENTS.md`（里面强制要求它去读 AI_CONTEXT）。

## ⚠️ 绝对禁止的危险操作

1. 🚫 **禁止在没有 `git pull` 的情况下直接开始写代码**（这会产生 Git 冲突地狱）。
2. 🚫 **禁止在没有修改 `.tasks.yaml` 并 `git push` 的情况下让 AI 跨端干活**。
3. 🚫 **禁止把绝对路径写死在代码里**（如 `C:/Users/...` 或 `/Users/...`）。所有路径必须使用相对路径并写在 `configs/default.yaml` 里。
4. 🚫 **禁止在 `git commit` 前不跑单元测试**。

---
*记住：你不是在写代码，你是一个调度了三个超级程序员的项目经理。维护好 `.tasks.yaml` 你的项目就能顺利完成。*
