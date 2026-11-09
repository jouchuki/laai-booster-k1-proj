#!/usr/bin/env bash
# Shared Linux/macOS path lookup. Source this file; pass the project root explicitly.
# Prefer a top-level checkout, reuse the older nested layout, or choose a top-level
# path for a new clone. The project directory itself can have any name.

booster_checkout_path() {
    local repo_root="$1" repo_name="$2"
    if [ -d "$repo_root/$repo_name" ]; then
        printf '%s\n' "$repo_root/$repo_name"
    elif [ -d "$repo_root/booster/$repo_name" ]; then
        printf '%s\n' "$repo_root/booster/$repo_name"
    else
        printf '%s\n' "$repo_root/$repo_name"
    fi
}
