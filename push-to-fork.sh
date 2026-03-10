#!/bin/bash
# Push documentation fixes to your fork and prepare for PR
# Uses gh CLI to create fork if needed, then pushes

set -e
cd "$(dirname "$0")"

UPSTREAM="genieincodebottle/generative-ai"
FORK_REPO="git@github.com:ljluestc/generative-ai.git"

echo "=== generative-ai: Fork & Push ==="
echo ""

# Ensure we're on main with clean state
git checkout main 2>/dev/null || true
if [[ -n $(git status --porcelain) ]]; then
  echo "Error: Working tree is dirty. Commit or stash changes first."
  exit 1
fi

# Step 1: Create fork via gh (creates fork if not exists, adds remote)
echo "Step 1: Ensuring fork exists..."
if gh repo fork "$UPSTREAM" --remote=true --remote-name=fork 2>/dev/null; then
  echo "  Fork ready."
else
  # Fork may already exist; add/update remote manually
  if ! git remote get-url fork &>/dev/null; then
    git remote add fork "$FORK_REPO"
    echo "  Added remote: fork"
  else
    git remote set-url fork "$FORK_REPO"
    echo "  Updated remote: fork"
  fi
fi

# Step 2: Show what will be pushed
echo ""
echo "Step 2: Commits to push:"
git log origin/main..main --oneline 2>/dev/null || git log fork/main..main --oneline 2>/dev/null || true
if [[ $(git rev-list origin/main..main 2>/dev/null | wc -l) -eq 0 ]] && [[ $(git rev-list fork/main..main 2>/dev/null | wc -l) -eq 0 ]]; then
  echo "  (no new commits - already in sync?)"
fi
echo ""

# Step 3: Push to fork (rewrite author if email privacy blocks)
echo "Step 3: Pushing to fork..."
PUSH_OUT=$(mktemp)
trap "rm -f $PUSH_OUT" EXIT
if ! git push fork main 2>&1 | tee "$PUSH_OUT"; then
  if grep -q "private email" "$PUSH_OUT" 2>/dev/null; then
    echo "  Push blocked by email privacy. Rewriting commits with GitHub no-reply..."
    NO_REPLY=$(gh api user -q '.login + "@users.noreply.github.com"' 2>/dev/null || echo "USER@users.noreply.github.com")
    UNAME=$(git config user.name 2>/dev/null || echo "User")
    FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter "
      export GIT_AUTHOR_EMAIL=\"$NO_REPLY\"
      export GIT_AUTHOR_NAME=\"$UNAME\"
      export GIT_COMMITTER_EMAIL=\"$NO_REPLY\"
      export GIT_COMMITTER_NAME=\"$UNAME\"
    " origin/main..HEAD 2>/dev/null
    git push fork main
  else
    cat "$PUSH_OUT"
    exit 1
  fi
fi

echo ""
echo "=== Success ==="
echo ""
echo "Open PR: https://github.com/ljluestc/generative-ai/compare/main...genieincodebottle:main"
echo ""
if command -v gh &>/dev/null; then
  echo "Create PR via CLI: gh pr create --repo genieincodebottle/generative-ai --base main --head ljluestc:main"
fi
