#!/usr/bin/env bash
set -e

echo "Checking for forbidden tracked files..."

# Exact match patterns
FORBIDDEN_FILES_REGEX='(^|/)\.DS_Store$|(^|/)\.coverage$|(^|/)htmlcov/'

if git ls-files | grep -E "$FORBIDDEN_FILES_REGEX"; then
  echo "❌ Forbidden tracked files detected!"
  exit 1
fi

echo "✅ Repository clean."