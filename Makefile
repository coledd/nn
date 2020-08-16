all:	train test predict README.pdf

data_prep.o: data_prep.c data_prep.h
	$(CC) -Wall data_prep.c -c -march=native -flto -Ofast

nn.o: nn.c nn.h
	$(CC) -Wall nn.c -c -march=native -flto -Ofast

train: train.c nn.o data_prep.o
	$(CC) -Wall train.c data_prep.o nn.o -o train -lm -march=native -Ofast

test: test.c nn.o data_prep.o
	$(CC) -Wall test.c data_prep.o nn.o -o test -lm -march=native -Ofast

predict: predict.c nn.o
	$(CC) -Wall predict.c nn.o -o predict -lm -march=native -Ofast

nn.png: nn.dot
	dot -Tpng nn.dot -o nn.png

README.pdf: README.md nn.png
	pandoc README.md -o README.pdf

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	rm -f data_prep.o nn.o train test predict model.txt tags nn.png README.pdf

