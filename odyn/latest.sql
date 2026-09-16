-- MIGRATION v2 -> v3
--
-- CHANGES:
-- - Add method_calls.ended_at, set when a recorded call returns or raises.
--
-- NOTES:
-- - NULL means the call is still running, or its process died before it could
--   finish. `migrate` refuses while any recent call has no ended_at, so a
--   migration cannot pull a schema out from under an overnight job.
-- - Calls recorded before this migration stay NULL.
-- - Additive, so calls already running under the old code finish normally.

ALTER TABLE method_calls
    ADD COLUMN ended_at TEXT CHECK(ended_at IS NULL OR datetime(ended_at) IS NOT NULL);
