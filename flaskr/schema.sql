-- Initialize the database.
-- Drop any existing data and create empty tables.

DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;
DROP TABLE IF EXISTS results;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

CREATE TABLE post (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  FOREIGN KEY (author_id) REFERENCES user (id)
);

CREATE TABLE results (
  user_id INTEGER PRIMARY KEY,
  filename VARCHAR(100),
  fs_methods VARCHAR(100),
  col_method1 TEXT,
  col_method2 TEXT,
  col_method3 TEXT,
  col_overlapped TEXT,
  col_selected_method TEXT,
  selected_method VARCHAR(30),

  FOREIGN KEY (user_id) REFERENCES user (id)
);