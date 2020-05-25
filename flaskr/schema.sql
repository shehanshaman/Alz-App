-- Initialize the database.
-- Drop any existing data and create empty tables.

DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS post;
DROP TABLE IF EXISTS results;
DROP TABLE IF EXISTS modeling;
DROP TABLE IF EXISTS mail_template;
DROP TABLE IF EXISTS verify;
DROP TABLE IF EXISTS classifiers;
DROP TABLE IF EXISTS preprocess;
DROP TABLE IF EXISTS file;

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  given_name VARCHAR(20) NOT NULL,
  image_url TEXT,
  last_login DATETIME NOT NULL,
  is_verified INTEGER NOT NULL,
  want_tour INTEGER DEFAULT 1,
  is_admin INTEGER DEFAULT 0,
  disk_space INTEGER DEFAULT 100,
  is_sent_warning INTEGER DEFAULT 0,
  warning_sent_time DATETIME DEFAULT NULL
);

-- is_verified: 0-Unverified, 1-GeNet-User, 2-Google-User

CREATE TABLE post (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  FOREIGN KEY (author_id) REFERENCES user (id)
);

CREATE TABLE preprocess (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL ,
  file_name VARCHAR(40) NOT NULL ,
  file_path TEXT NOT NULL ,
  annotation_table VARCHAR(30) NOT NULL ,
  col_sel_method VARCHAR(30) NOT NULL ,
  merge_df_path TEXT DEFAULT NULL ,
  avg_symbol_df_path TEXT DEFAULT NULL ,
  reduce_df_path TEXT DEFAULT NULL ,
  scaling VARCHAR(30) DEFAULT NULL ,
  imputation VARCHAR(30) DEFAULT NULL ,
  classifiers VARCHAR(10) DEFAULT NULL ,

  FOREIGN KEY (user_id) REFERENCES user (id)
);

CREATE TABLE results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL ,
  filename VARCHAR(100) NOT NULL ,
  classifiers VARCHAR(10) DEFAULT '4,5,6',
  fs_methods VARCHAR(100) NOT NULL ,
  col_method1 TEXT NOT NULL ,
  col_method2 TEXT NOT NULL ,
  col_method3 TEXT NOT NULL ,
  col_overlapped TEXT,
  col_selected_method TEXT,
  selected_method VARCHAR(30),

  FOREIGN KEY (user_id) REFERENCES user (id)
);

CREATE TABLE modeling (
  user_id INTEGER PRIMARY KEY,
  trained_file VARCHAR(100),
  clasifier VARCHAR(50),
  features TEXT,
  model_path_name VARCHAR(100),
  accuracy VARCHAR(10) DEFAULT NULL,
  has_model INTEGER DEFAULT 1,

  FOREIGN KEY (user_id) REFERENCES user (id)
);

CREATE TABLE mail_template (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  subject VARCHAR(100),
  message TEXT
);

CREATE TABLE verify (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER,
  subject VARCHAR(50),
  verify_key VARCHAR(20),

  FOREIGN KEY (user_id) REFERENCES user (id)
);

CREATE TABLE classifiers (
  id INTEGER PRIMARY KEY,
  clf_name VARCHAR(50),
  short_name VARCHAR(10)
);

INSERT INTO classifiers (id, clf_name, short_name) VALUES
  (1, 'Gaussian Naive Bayes', 'GNB'),
  (2, 'Decision Tree', 'DT'),
  (3, 'Nearest Neighbors', 'NN'),
  (4, 'SVM + Gaussian kernel', 'GK'),
  (5, 'SVM + linear kernel', 'LK'),
  (6, 'Random Forest', 'RF');


CREATE TABLE file (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  file_name VARCHAR(100) NOT NULL ,
  file_type VARCHAR(10) NOT NULL,
  path TEXT NOT NULL,
  user_id int NOT NULL,
  is_annotation int NOT NULL,
  has_class int NOT NULL

--   FOREIGN KEY (user_id) REFERENCES user (id)
);

ALTER TABLE preprocess ADD after_norm_set varchar(500);
ALTER TABLE preprocess ADD volcano_hash MEDIUMTEXT;
ALTER TABLE preprocess ADD fold varchar(10);
ALTER TABLE preprocess ADD pvalue varchar(10);
ALTER TABLE preprocess ADD length varchar(10);
ALTER TABLE preprocess ADD fr_univariate_hash MEDIUMTEXT;
ALTER TABLE preprocess ADD classification_result_set varchar(500);
ALTER TABLE results ADD venn_data_set varchar(5000);
ALTER TABLE results ADD fs_hash MEDIUMTEXT;
ALTER TABLE results ADD an_overlap_hash MEDIUMTEXT;
ALTER TABLE results ADD an_cls_hash MEDIUMTEXT;
ALTER TABLE results ADD an_crr_hash MEDIUMTEXT;
ALTER TABLE results ADD an_crr_1_hash MEDIUMTEXT;
ALTER TABLE results ADD an_crr_2_hash MEDIUMTEXT;
ALTER TABLE results ADD corr_classification_accuracy varchar(500);
ALTER TABLE results ADD selected_roc_pic_hash MEDIUMTEXT;
ALTER TABLE results ADD all_roc_pic_hash MEDIUMTEXT;
ALTER TABLE results ADD result_data_1 varchar(500);
ALTER TABLE results ADD result_data_2 varchar(500);
ALTER TABLE preprocess ADD can_download INTEGER;
ALTER TABLE results ADD can_download_fs INTEGER;
ALTER TABLE results ADD can_download_anlz INTEGER;