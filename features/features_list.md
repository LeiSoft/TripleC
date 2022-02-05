|features|details|structure|
|---|---|---|
|sentiment|From [MFDSL](https://github.com/FoVNull/MFDSL)|n-dim vector for token<br>|
|POS pattern|one-hot represent of spacy pos|n-dim vector for token|
|POS pattern1|From [deep-cenic](https://github.com/julianprester/deep-cenic). 5 seq pos pattern|repeat for every token|
|tf-idf|1-dim from tensorflow|tf_idf matrix|
|citation offsets|begin/end position in the context|\[begin_offset/len(context), end_offset/len(context)\]|