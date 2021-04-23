fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "freebase/train.bpe" \
  --validpref "freebase/val.bpe" \
  --destdir "freebase/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
