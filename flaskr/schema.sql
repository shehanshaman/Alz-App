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

CREATE TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  given_name VARCHAR(20) NOT NULL,
  image_url TEXT,
  last_login DATETIME NOT NULL,
  is_verified INTEGER NOT NULL,
  want_tour INTEGER DEFAULT 1,
  is_admin INTEGER DEFAULT 0
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

INSERT INTO user (username, given_name, password, last_login, is_verified, is_admin) VALUES ('user', 'user', 'pbkdf2:sha256:150000$LuBMEbIc$f3e7ec7e9061bad12ffb9193a8740722cc96be7e193c520e3aa551dc43c78b7c', '2020-04-10 17:24:15.484058', 3, 1);
-- INSERT INTO results (user_id, filename, fs_methods, col_method1, col_method2,
--                      col_method3, col_overlapped, col_selected_method, selected_method) VALUES (1, 'GSE5281_DE_200.plk', 'PCA,Random Forest,Extra Tree Classifier,50', 'SST,CHGB,CALY,STAT4,AX747182,SERTM1,PCSK1,MT1M,LOC101929787,MAFF,KIFAP3,MAL2,AMPH,SLC39A12,ZIC2,PCYOX1L,TAC1,JPX,TMEM200A,SLIRP,SLC39A10,SPHKAP,CALB1,CDK7,SGIP1,AP2M1,CDK5,NAP1L5,LOC100507557,MRPL15,CTD-3092A11.2,PLK2,ZDHHC23,SMYD3,P2RY14,RP11-271C24.3,ZNF415,SCG2,EMX2,SERPINF1,ARPC1A,PVALB,ID3,THYN1,LOC101060510,PRO1804,LINC00889,DDIT4,PSMD8,FPGT-TNNI3K', 'NEAT1,MIR612,PCYOX1L,CTD-3092A11.2,SLC12A7,FIBP,MLLT11,CKMT1B,JPX,GNG3,SST,CKMT1A,NIT2,ATP6V1E1,MAFF,MGC12488,LDHA,FIG4,LOC101929787,LOC202181,MT1M,ATP6V1G2,ATP5B,TUBB4B,LOC100272216,REEP1,CHRM1,COPG2IT1,TUBB3,AK090844,PSMB3,PRO1804,MIF,MKKS,GFAP,RGS7,CDK5,IMMT,PSMA5,PLSCR4,LRP4,BSCL2,PRR34-AS1,RP11-271C24.3,GPI,DDIT4,SLC39A12,YAP1,ATP5C1,AC004951.6', 'NEAT1,MIR612,TUBB4B,CTD-3092A11.2,SLC12A7,MAFF,PCYOX1L,FIBP,ATP5B,SLC39A12,GNG3,MGC12488,JPX,MT1M,LDHA,LOC101929787,CKMT1A,ATP5C1,PSMA5,LOC202181,CKMT1B,ATP1A1,SST,BCAS2,NAA20,LOC100272216,GPI,FIG4,EMC4,ATP6V1G2,SNCA,MKKS,ZNF415,AP2M1,CALY,SCN3B,GFAP,GPRASP1,BSCL2,MLLT11,PLSCR4,ID3,NECAP1,RP11-271C24.3,PSMB3,APOO,SLIRP,CCNH,RPH3A,STMN2', 'RP11-271C24.3,CTD-3092A11.2,MAFF,PCYOX1L,MT1M,SST,LOC101929787,JPX,SLC39A12', 'CHGB,STAT4,AX747182,ZIC2,LOC101060510', 'PCA' );
INSERT INTO modeling (user_id, trained_file, clasifier, features, model_path_name, accuracy, has_model) VALUES (1, 'GSE5281_DE_200.plk', 'svmLinear', 'RP11-271C24.3,CTD-3092A11.2,MAFF,PCYOX1L,MT1M,SST,LOC101929787,JPX,SLC39A12,CHGB,STAT4,AX747182,ZIC2,LOC101060510', '_model.pkl', '80.1', 1 );