BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "file" (
	"id"	INTEGER PRIMARY KEY AUTOINCREMENT,
	"file_name"	VARCHAR(100) NOT NULL,
	"file_type"	VARCHAR(10) NOT NULL,
	"path"	TEXT NOT NULL,
	"user_id"	int NOT NULL,
	"is_annotation"	int NOT NULL,
	"has_class"	int NOT NULL
);
INSERT INTO "file" VALUES (1,'GPL570-55999.csv','csv','/AnnotationTbls/GPL570-55999.csv',0,1,0);
INSERT INTO "file" VALUES (2,'GPL96-57554.csv','csv','/AnnotationTbls/GPL96-57554.csv',0,1,0);
INSERT INTO "file" VALUES (3,'GPL97-17394.csv','csv','/AnnotationTbls/GPL97-17394.csv',0,1,0);
COMMIT;
