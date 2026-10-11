-- Migration 015: accounts defined in .env (a stable key per account), a profile picture, and the Viewer role.
ALTER TABLE meta.users ADD COLUMN IF NOT EXISTS key TEXT UNIQUE, ADD COLUMN IF NOT EXISTS avatar TEXT;
ALTER TABLE meta.users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE meta.users ADD CONSTRAINT users_role_check CHECK (role IN ('lead', 'member', 'viewer'));
