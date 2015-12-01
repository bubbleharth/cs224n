ant && java -cp "classes:extlib/*" cs224n.deep.Baseline ../data

echo
echo "Train:"
../conlleval -r -d '\t' < train.out | head -n 2
echo "Dev:"
../conlleval -r -d '\t' < dev.out
