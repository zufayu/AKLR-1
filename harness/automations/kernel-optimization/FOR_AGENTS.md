# AKLR kernel automation — start here

This is the shared entry for setting up, maintaining, or running this automation pack.

The workflow entry is
[skills/kernel-automation/SKILL.md](skills/kernel-automation/SKILL.md).
For setup or configuration changes, follow its
[setup guide](skills/kernel-automation/references/setup.md).
For a runtime round, follow the entry skill and its four responsibility skills without repeating setup.

Use [templates/automation-prompt.md](templates/automation-prompt.md) as the prompt
in the owner's selected scheduler. That prompt reads this file on each trigger, which routes runtime
invocations to the entry skill. Keep the workflow
rules in the skills rather than copying them into the prompt.

Keep the pack's source in this directory. Read its skills directly by repository
path and reuse shared kernel skills from `harness/skills/`.
Preserve `.kernel-optimizer/configuration.yaml` and other user changes;
use the configuration template only when creating a new configuration. Keep the actual configuration
local and ignored by Git. Shared skills and templates must not contain fixed absolute paths, internal
hostnames, account identities or credentials; use repository-relative references and configuration values.
Use the owner's recorded model, environment, budgets, and permissions.

Configure one scheduled workflow. Update the existing instance if it is already present.
Bind the intended checkout and execution session, and record the effective revision and model settings.
If the scheduler checks out source afresh, make the required files available on that revision first.
