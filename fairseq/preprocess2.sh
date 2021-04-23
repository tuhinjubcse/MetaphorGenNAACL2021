fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "metaphor/train.bpe" \
  --validpref "metaphor/val.bpe" \
  --destdir "metaphor/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
