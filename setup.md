
# TODO: ARGO RAG System Contributor's Guide

## The Orchestrator Workflow

This guide outlines a new, highly automated workflow for populating the ARGO RAG system's vector database. Your task as a team member is to use a single, powerful "Orchestrator" prompt in the Gemini CLI. This prompt will guide the entire process of generating, testing, and refining prompts for a given category, one batch at a time.

### Your Task: Run the Orchestrator

Your entire task is now encapsulated in a single master prompt. You will paste this prompt into the Gemini CLI, and it will guide you through the process.

**Instructions:**

1.  **Open the Gemini CLI in the project directory** (see setup instructions below).
2.  **Copy the entire "Orchestrator Master Prompt"** you can find that in seperate masterprompt.md .
3.  **Paste it into the Gemini CLI** and press Enter.
4.  The Gemini CLI will then ask you for the query category you are working on. Type it in and press Enter.
5.  The CLI will then execute the entire workflow for the first batch of 10 prompts.
6.  Once a batch is complete and has a good similarity score, you will be prompted to paste the master prompt again to start the next batch.


## Setup and Reference

### Setting up the Gemini CLI

**Important Note on Accounts:** The Gemini CLI is free for personal/unofficial Google accounts. Please **do not use your official SRM organizational account**.

To interact with the project's AI assistant, you will need to have the Gemini CLI installed and configured on your system.

#### Windows Installation

**Reference Link:** [Download Gemini CLI for Windows](https://your-download-link-for-windows.com/gemini.exe)

**Installation Flow:**

```
┌──────────────────────────┐
│ 1. Download gemini.exe   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 2. Create a new folder   │
│    (e.g., C:\GeminiCLI)   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 3. Move gemini.exe to    │
│    the new folder        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 4. Add the folder to     │
│    your system's PATH    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 5. Open a new terminal   │
│    and run 'gemini --version' │
└──────────────────────────┘
```

#### macOS Installation

**Reference Link:** [Download Gemini CLI for macOS](https://your-download-link-for-macos.com/gemini)

**Installation Flow:**

```
┌──────────────────────────┐
│ 1. Download the 'gemini' │
│    binary                │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 2. Open a terminal and   │
│    navigate to the       │
│    download location     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 3. Make the binary       │
│    executable:           │
│    'chmod +x gemini'     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 4. Move the binary to    │
│    /usr/local/bin:       │
│    'sudo mv gemini /usr/local/bin/'│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 5. Open a new terminal   │
│    and run 'gemini --version' │
└──────────────────────────┘
```

### Running Gemini CLI in the Project Directory

It is important to run the Gemini CLI from within the project\'s root directory.

#### Windows

1.  Open File Explorer and navigate to the cloned repository folder.
2.  Click on the address bar at the top of the File Explorer window.
3.  Type `cmd` and press Enter. This will open a Command Prompt window in the correct directory.
4.  You can now run `gemini` commands.

#### macOS

1.  Open the Terminal application.
2.  Use the `cd` command to navigate to the cloned repository folder.
    ```bash
    cd /path/to/your/cloned/repo
    ```
3.  You can now run `gemini` commands.
