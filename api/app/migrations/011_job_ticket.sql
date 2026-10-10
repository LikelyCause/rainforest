-- Migration 011: queue order for background jobs. Each job takes a ticket when it's submitted; a waiting job takes a
-- free slot only when no older ticket is still waiting.

CREATE SEQUENCE IF NOT EXISTS meta.job_ticket AS integer CYCLE;
