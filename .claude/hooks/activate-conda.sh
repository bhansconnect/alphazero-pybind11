#!/bin/bash
# Activate the torch conda environment for Claude Code sessions
if [ -n "$CLAUDE_ENV_FILE" ]; then
  cat >> "$CLAUDE_ENV_FILE" << 'EOF'
export PATH="/opt/homebrew/Caskroom/miniconda/base/envs/torch/bin:$PATH"
export CONDA_DEFAULT_ENV="torch"
export CONDA_PREFIX="/opt/homebrew/Caskroom/miniconda/base/envs/torch"
EOF
fi
exit 0
