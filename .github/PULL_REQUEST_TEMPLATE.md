## Summary
<!-- Provide a clear and concise description of what this PR changes and why it's needed -->

Closes #(issue_number) <!-- Link related issues with "Closes #123" or "Fixes #456" -->

### What Changed?
<!-- Describe the changes in this PR -->

### Why?
<!-- Explain the motivation behind these changes -->

## Type of Change
<!-- Check all that apply -->
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring (code improvement without behavior change)
- [ ] Style/UI changes
- [ ] Performance improvement
- [ ] Test addition or update
- [ ] Configuration/build changes
- [ ] Security fix

## Testing
<!-- Describe how you tested these changes -->

### Local Testing Checklist
- [ ] `cargo fmt --all -- --check` - Code formatting
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` - Linting
- [ ] `cargo test --workspace -- --test-threads=1` - All tests pass
- [ ] `cargo audit` - No security vulnerabilities
- [ ] `python -m ruff check nexum_ai/` - Python linting (if applicable)
- [ ] `python -m compileall nexum_ai` - Python compilation (if applicable)
- [ ] Manual testing performed (describe below)

### Testing Details
<!-- Describe your testing approach, edge cases covered, etc. -->

## Screenshots/Examples
<!-- If applicable, add screenshots or code examples demonstrating the changes -->

## Performance Impact
<!-- Describe any performance implications of these changes -->
- [ ] No performance impact
- [ ] Performance improvement (describe below)
- [ ] Potential performance concern (describe below)

## Security Considerations
<!-- Have you considered security implications? -->
- [ ] No security concerns
- [ ] Security improvement (describe below)
- [ ] Requires security review (describe below)

## Documentation
- [ ] Updated inline code comments/documentation
- [ ] Updated README.md (if applicable)
- [ ] Updated CHANGELOG.md (will be auto-updated on release)
- [ ] Updated TESTING.md (if test procedures changed)
- [ ] Added/updated examples in `demo.sh` (if applicable)

## Pre-Merge Checklist
- [ ] Self-reviewed my own code
- [ ] Added tests covering my changes
- [ ] All existing tests pass locally
- [ ] Updated relevant documentation
- [ ] No uncommitted debug code or console logs
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] Follows [Code of Conduct](CODE_OF_CONDUCT.md)
- [ ] Branch is up to date with `main`

## Additional Notes
<!-- Any additional context, deployment notes, follow-up tasks, or breaking changes details -->

**Reviewers**: Please check that all checkboxes are ticked before approving
