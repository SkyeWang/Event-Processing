1、cd camr

2、bash ./scripts/config.sh

3、python amr_parsing.py -m preprocess sentences.txt, 得到标记化的句子( .tok)、POS标签和名称实体( .prp) 以及依赖结构( .charniak.parse.dep)

4、python amr_parsing.py -m parse --model [model_file] sentences.txt, [model_file]即amr-anno-1.0.train.m.tar.gz解压后的路径. 然后我们会得到得到解析后的AMR文件( .parsed)，表示为[input_amr_file]

5、python amr_parsing.py -m preprocess --amrfmt amr [input_amr_file], 得到标记化的AMR文件( .amr.tok)，表示为[input_amr_tok_file]

6、bash ./setup

7、bash scripts/config.sh

8、./compile

9、${JAMR_HOME}/run Aligner -v 0 --print-nodes-and-edges < [input_amr_tok_file] > output_file

10、output_file即最终结果