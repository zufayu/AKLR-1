<!-- Use this prompt in the configured scheduler. -->

Run one round of the AKLR kernel optimization automation in this repository.

Read and follow `harness/automations/kernel-optimization/FOR_AGENTS.md`.
This is a runtime invocation, not setup. Follow its runtime route to the entry skill.
Use `.kernel-optimizer/configuration.yaml`. Report missing configuration or
prerequisites; follow the entry skill's rules for any work that can still proceed.

Follow the configured mode, budget, model and permissions. Do not manage schedules or reinitialize
existing state. Save a concise Chinese progress report and release the round lock in final cleanup.
End this invocation after one round; the existing scheduler supplies the next wakeup.
